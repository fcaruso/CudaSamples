#include <iostream>

// CUDA utilities and system includes
#include <cuda_runtime.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_image.h>

#include <cuda_profiler_api.h>

char *image_filename = "....";
unsigned int width, height;
unsigned char *h_img  = NULL;
unsigned char *d_img  = NULL;

#define BLOCK_WIDTH		32
#define BLOCK_HEIGHT	32

__global__ void kRGBtoGrayscale( ... )
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
	cudaMallocPitch( .... );
	cudaMemcpy2D( .... );

	// process image
	dim3 dGrid( ... );
	dim3 dBlock( ... );
	kRGBtoGrayScale <<< .... >>> ( .... );
	cudaThreadSynchronize();
	// save image
	cudaMemcpy2D( .... );
	sdkSavePGM("grayscale.ppm", h_img, width, height );

	// free memory
	cudaFree( d_img );
	cudaProfilerStop();
	return 0;
}
