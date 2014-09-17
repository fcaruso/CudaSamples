///////////////////////////////////////////////////////
/////////
/////////  Simple program to explain streams usage in CUDA
/////////  
/////////  
/////////	Author: Francesco Caruso
/////////	francescocaruso979@gmail.com


#include <iostream>

#include <cuda_runtime.h>

////////////////////////
/// Error Handling code
////////////////////////

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//--------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	int whichDevice = 0;
	cudaDeviceProp prop;
	
	HANDLE_ERROR( cudaSetDevice(whichDevice) );
	
	HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice );

	return 0;
}