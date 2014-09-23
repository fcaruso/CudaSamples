#include <iostream>

// CUDA utilities and system includes
#include <cuda_runtime.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_image.h>

#include <cuda_profiler_api.h>

char *image_filename = "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v6.5/common/data/lena.pgm";
unsigned int width, height;
unsigned char *h_img  = NULL;
unsigned char *d_img  = NULL;

#define BLOCK_WIDTH		32
#define BLOCK_HEIGHT	32

template<unsigned short RADIUS >
__global__ void kRadialBlur( unsigned char* img, unsigned width, unsigned height, size_t pitch)
{
	__shared__ unsigned char sh[BLOCK_WIDTH + 2*RADIUS][BLOCK_HEIGHT + 2*RADIUS];

	int g_x = blockDim.x*blockIdx.x + threadIdx.x;
	int g_y = blockDim.y*blockIdx.y + threadIdx.y;

	int pid_x = threadIdx.x + RADIUS;
	int pid_y = threadIdx.y + RADIUS;
	
	///////////////////////
	// gather shared memory
	///////////////////////
	sh[pid_y][pid_x] = img[ g_y*pitch + g_x];

	// halo 
	if ( ( threadIdx.x < RADIUS ) && ( g_x  >= RADIUS ) )
	{
		sh[pid_y][pid_x - RADIUS] = img[ g_y*pitch + g_x - RADIUS];

		if ( ( threadIdx.y < RADIUS ) && ( g_y >= RADIUS ) )
		{
			sh[pid_y - RADIUS][pid_x - RADIUS] = img[ (g_y - RADIUS)*pitch + g_x - RADIUS];
		}
		if ( ( threadIdx.y > (BLOCK_HEIGHT -1 - RADIUS) ) )
		{
			sh[pid_y + RADIUS][pid_x - RADIUS] = img[ (g_y + RADIUS)*pitch + g_x - RADIUS];
		}
	}
	if ( ( threadIdx.x > ( BLOCK_WIDTH -1 - RADIUS ) ) && ( g_x < ( width - RADIUS ) ) )
	{
		sh[pid_y][pid_x + RADIUS ] = img[ g_y*pitch + g_x + RADIUS];

		if ( ( threadIdx.y < RADIUS ) && ( g_y > RADIUS ) )
		{
			sh[pid_y - RADIUS][pid_x + RADIUS] = img[ (g_y - RADIUS)*pitch + g_x + RADIUS];
		}
		if ( (threadIdx.y > (BLOCK_HEIGHT -1 - RADIUS ) ) && ( g_y < ( height - RADIUS ) ) )
		{
			sh[pid_y + RADIUS][pid_x + RADIUS] = img[ (g_y + RADIUS)*pitch + g_x + RADIUS];
		}
	}

	if ( ( threadIdx.y < RADIUS ) && ( g_y >= RADIUS ) )
	{
		sh[pid_y - RADIUS][pid_x] = img[ (g_y - RADIUS)*pitch + g_x];
	}
	if ( ( threadIdx.y > ( BLOCK_HEIGHT -1 - RADIUS ) ) && ( g_y < ( height - RADIUS ) ) )
	{
		sh[pid_y + RADIUS][pid_x] = img[ ( g_y + RADIUS)*pitch + g_x ];
	}

	__syncthreads();

	//////////////////////
	// compute the blurred value
	//////////////////////

	unsigned val = 0;
	unsigned k = 0;
	for (int i=-RADIUS; i<= RADIUS; i++ )
		for ( int j=-RADIUS; j<=RADIUS ; j++ )
		{
			if ( ( ( g_x + j ) < 0 ) || ( ( g_x + j ) > ( width - 1) ) )
				continue;
			if ( ( ( g_y + i ) < 0 ) || ( ( g_y + i ) > ( height - 1) ) )
				continue;
			val += sh[pid_y + i][pid_x + j];
			k++;
		}

	val /= k;

	////////////////////
	// write into global memory
	///////////////

	img[ g_y*pitch + g_x ] = (unsigned char) val;
			
}

__global__ void kBlur(unsigned char* img, unsigned width, unsigned height, size_t pitch)
{
	__shared__ unsigned char sh[BLOCK_WIDTH][BLOCK_HEIGHT];

	int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
	int tid_y = blockDim.y*blockIdx.y + threadIdx.y;

	sh[threadIdx.x][threadIdx.y] = img[ tid_y*pitch + tid_x ];
	
	__syncthreads();

	unsigned int val = 0;
	for (int i=0; i< blockDim.y; i++ )
		for ( int j=0; j < blockDim.x; j++ )
			val += sh[i][j];

	val /= blockDim.x*blockDim.y;

	img[ tid_y*pitch + tid_x ] = (unsigned char) val;

}

int main(int argc, char* argv[])
{
	cudaProfilerStart();
    // load image (needed so we can get the width and height before we create the window
	sdkLoadPGM(image_filename, (unsigned char **) &h_img, &width, &height);
	printf("width: %d \t height: %d \n", width, height);

	// fill GPU 
	unsigned char* d_img = NULL;
	size_t pitch;
	cudaMallocPitch( (void**) &d_img, &pitch, width*sizeof(unsigned char), height );
	cudaMemcpy2D( d_img, pitch*sizeof(unsigned char), 
			h_img, width*sizeof(unsigned char), width*sizeof(unsigned char), height, 
			cudaMemcpyHostToDevice );

	// process image
	dim3 dGrid(width / BLOCK_WIDTH, height / BLOCK_HEIGHT);
	dim3 dBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
	kRadialBlur<4> <<< dGrid, dBlock >>> (d_img, width, height, pitch );
	cudaThreadSynchronize();
	// save image
	cudaMemcpy2D( h_img, width*sizeof(unsigned char), 
		d_img, pitch*sizeof(unsigned char), width*sizeof(unsigned char), height,
		cudaMemcpyDeviceToHost );
	sdkSavePGM("blurred.ppm", h_img, width, height );

	// free memory
	cudaFree( d_img );
	cudaProfilerStop();
	return 0;
}