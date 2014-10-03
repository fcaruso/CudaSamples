#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <omp.h>

#include <Windows.h> // for Timing

using namespace std;
using std::string;
using std::fstream;
using std::vector;


struct  WavHeader{
    char                RIFF[4];        // RIFF Header      Magic header
    unsigned long       ChunkSize;      // RIFF Chunk Size  
    char                WAVE[4];        // WAVE Header      
    char                fmt[4];         // FMT header       
    unsigned long       Subchunk1Size;  // Size of the fmt chunk                                
    unsigned short      AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM 
    unsigned short      NumOfChan;      // Number of channels 1=Mono 2=Sterio                   
    unsigned long       SamplesPerSec;  // Sampling Frequency in Hz                             
    unsigned long       bytesPerSec;    // bytes per second 
    unsigned short      blockAlign;     // 2=16-bit mono, 4=16-bit stereo 
    unsigned short      bitsPerSample;  // Number of bits per sample      
    char                Subchunk2ID[4]; // "data"  string   
    unsigned long       Subchunk2Size;  // Sampled data length    
}; 

template<typename ValueType, typename TimeType>
struct TWavePoint
{
	ValueType	value;
	TimeType	time;
};

typedef TWavePoint< short, float> WavePoint;
typedef vector< WavePoint > Wave;

Wave gWavePoint;
Wave gWaveMinMax;
float gWaveTimeLength = 0;

// Function prototypes 
int getFileSize(FILE *inFile); 

void launch_kFindZeros(float* zeros, short* data, int N, int sampleRate);
void launch_kFindMinMax(unsigned* minmax, short* data, int N, int sampleRate);


/////////////////////////////////////////////////////
///// function that returns an array of values 
///// which are the times (in seconds) in which the signal is 0
///// 
void findZeros(float* zeros, int& n, short* data, int N, int samplerate)
{
	n=0;
	const float deltat= 1.f/samplerate;
	for (int i=0, j=1; j<N; i++, j++)
	{
		short cur = data[i];
		short next = data[j];
		
		if (cur*next < 0 ) // detect if it is negative
		{
			float m = (next - cur)/deltat;
			float x = -cur/m;
			zeros[n]= x+i*deltat;
			n++;
		}
	};
}


void findBottomPoints(Wave& waveMinMax, Wave& data )
{
	int i=0; int j=1;
	short cur = data[i].value;
	short next = data[j].value;
	char currentSlope = next > cur ? 1 : -1;
	do
	{
		i++, j++;
		cur = next;
		next = data[j].value;
		char nextSlope = next > cur ? 1 : -1;
		if ( ( currentSlope * nextSlope < 0 ) && ( currentSlope == - 1) )
		{
			//if ( data[i].value < -8000 ) 
			{
				waveMinMax.push_back(data[i]);
			}
			
		}
		currentSlope = nextSlope;
	}while ( j<data.size() - 1);
}

void findMinMax(Wave& waveMinMax, Wave& data)
{
	//int i=0; int j=1;
	short cur = data[0].value;
	short next = data[1].value;
	char currentSlope = next > cur ? 1 : -1;
#pragma omp parallel for
	for (int i=1, j=2; j<data.size() -1; i++, j++)
	{
		cur = next;
		next = data[j].value;
		char nextSlope = next > cur ? 1 : -1;
		if ( currentSlope * nextSlope < 0 )
		{
			waveMinMax.push_back(data[i]);
		}
		currentSlope = nextSlope;
	}
}


void	computeFrequencies(float* freq, float* zeros, int N)
{
	int i=0, j=1;
	do
	{
		float cur = zeros[i];
		float next = zeros[j];
		freq[i] = 1.f/(next - cur);
		i++; j++;
	} while(j<=N);

}

int main(int argc, char** argv)
{

	////////////////
	/// Read Header

	const char* szFileName = "../wav/new/A.wav";
	FILE*	fp = fopen(szFileName,"r");
	assert(fp);
	WavHeader wavHeader;
	fread(&wavHeader, 1, sizeof(wavHeader), fp);

	long iFileLength = getFileSize(fp);
	const int iDataLength = wavHeader.Subchunk2Size;
	const long iNumberOfSamples = iDataLength/( wavHeader.bitsPerSample/8);


	///////////////
	/// Read Bytes

	short*	bytes = new short[iDataLength/2];
	fseek(fp, sizeof(wavHeader), 0);
	//fread(&bytes[0],1,iDataLength, fp);
	const int offset = 0;
	const int sampleDim = iDataLength/2;// datalength in byte, n of sample = nOfByte/2
	const short threshold = MAXBYTE;
	fread(&bytes[offset],2,sampleDim, fp); 
	gWavePoint.clear();
	for (int i=0; i<sampleDim; i++)
	{
		WavePoint p;
		p.value = bytes[i + offset];
		p.time = (float) (i + offset)/wavHeader.SamplesPerSec;
		//if ( p.value < threshold )
			gWavePoint.push_back(p);
	}


	static LARGE_INTEGER last;
	static LARGE_INTEGER current;
	static LARGE_INTEGER frequency;
	
	QueryPerformanceFrequency(&frequency);	
	
	///============================================================
	/// ZerosCrossing 
	//////////////////////
	///   Compute the zeros on CPU
	float*	zeros = new float[sampleDim];
	int		nOfZeros;
	
	QueryPerformanceCounter(&last);
	//findZeros(zeros, nOfZeros, bytes, iNumberOfSamples, wavHeader.SamplesPerSec );
	findZeros(zeros, nOfZeros, &bytes[offset], sampleDim, wavHeader.SamplesPerSec );
	QueryPerformanceCounter( &current );
	float cpuElapsedTime = (float) (current.QuadPart - last.QuadPart) / frequency.QuadPart;
	
	/// compute zeros on GPU
	float* hostZeroArray = (float*) malloc(sampleDim*sizeof(float));
	float* gpuZeroArray;
	cudaMalloc(&gpuZeroArray, sampleDim*sizeof(float));
	cudaMemset(gpuZeroArray,0,sampleDim*sizeof(float));

	short* gpuDataArray;
	cudaMalloc(&gpuDataArray, sampleDim*sizeof(short));

	// reset the timer
	QueryPerformanceCounter(&last);

	// launch
	cudaMemcpy(gpuDataArray, &bytes[offset], sampleDim*sizeof(short), cudaMemcpyHostToDevice);
	launch_kFindZeros( gpuZeroArray, gpuDataArray, sampleDim, wavHeader.SamplesPerSec );
	cudaMemcpy(hostZeroArray, gpuZeroArray, sampleDim*sizeof(float), cudaMemcpyDeviceToHost );
	
	// get the time
	QueryPerformanceCounter( &current );
	float gpuElapsedTime = (float) (current.QuadPart - last.QuadPart) / frequency.QuadPart;


	// check the result
	cout << " chek GPU ZeroCross result...." ;
	float* ptr = hostZeroArray;
	for (int i=0; i<nOfZeros; i++)
	{
		while ( *ptr < 0)
		{
			ptr++;
		}
		assert( fabs(zeros[i] - *ptr)  < 1e-6 );
	/*	if ( fabs(zeros[i] - *ptr)  > 1e-6 )
			printf("%f %f\n",zeros[i], *ptr);*/
		ptr++;
	}
	cout << "PASSED" << endl;
	cout << " cpu ZeroCross computing time: " << cpuElapsedTime << endl;
	cout << " gpu ZeroCross computing time: " << gpuElapsedTime << endl;
	
	
	
	
	//float*	frequencies = new float[nOfZeros];
	//computeFrequencies(frequencies, zeros, nOfZeros);


	//=========================================================
	// MinMax 
	//=========================================================

	// MinMax on CPU
	
	gWaveMinMax.resize(0);
	QueryPerformanceCounter(&last);
	findMinMax(gWaveMinMax, gWavePoint);
	QueryPerformanceCounter(&current);
	cpuElapsedTime = (float) (current.QuadPart - last.QuadPart) / frequency.QuadPart;

	// MinMax on GPU
	unsigned* hostMinMaxArray = (unsigned*) malloc(sampleDim*sizeof(unsigned));
	unsigned* gpuMinMaxArray;
	cudaMalloc(&gpuMinMaxArray, sampleDim*sizeof(unsigned));
	cudaMemset(gpuMinMaxArray,-1,sampleDim*sizeof(unsigned));
	QueryPerformanceCounter(&last);
	launch_kFindMinMax( gpuMinMaxArray, gpuDataArray, sampleDim, wavHeader.SamplesPerSec );
	cudaMemcpy(hostMinMaxArray, gpuMinMaxArray, sampleDim*sizeof(unsigned), cudaMemcpyDeviceToHost );
	QueryPerformanceCounter(&current);
	gpuElapsedTime = (float) (current.QuadPart - last.QuadPart) / frequency.QuadPart;
	// check FindMinMax GPU result
	{
		cout << " chek FindMinMax GPU result...." ;
		unsigned* ptr = hostMinMaxArray;
		for (int i=0; i<gWaveMinMax.size(); i++)
		{
			while ( *ptr == -1)
			{
				ptr++;
			}
			assert( abs(gWaveMinMax[i].value - bytes[*ptr])  < 100 );
		/*	if ( fabs(zeros[i] - *ptr)  > 1e-6 )
				printf("%f %f\n",zeros[i], *ptr);*/
			ptr++;
		}
		cout << "PASSED" << endl;
	}
	cout << " cpu findMinMax computing time: " << cpuElapsedTime << endl;
	cout << " gpu findMinMax computing time: " << gpuElapsedTime << endl;

	cout << " Sample Dimension: " << sampleDim << endl;

	//findBottomPoints(gWaveMinMax, gWavePoint);
	//cout << "number of bottom points: " << gWaveMinMax.size() << endl;
	//float dt = gWavePoint.rbegin()->time - gWavePoint.begin()->time;
	//cout << "frequency: " << gWaveMinMax.size()/dt << endl;


	////////////////////
	/// Clean up
	delete[] bytes;
	fclose(fp);
	cudaFree(gpuDataArray);
	cudaFree(gpuZeroArray);
	return 0;
}


