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

			
}

__global__ void kBlur(unsigned char* img, unsigned width, unsigned height, size_t pitch)
{


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