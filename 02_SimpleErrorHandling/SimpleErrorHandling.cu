///////////////////////////////////////////////////////
/////////
/////////  Program to explain error handling in CUDA
/////////  
/////////	Author: Francesco Caruso
/////////	francescocaruso979@gmail.com


#include <iostream>
#include <cuda.h>

#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

const int N = 1023;

#define	REAL float


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

//////////////////////////////
/// End of Error Handling code
//////////////////////////////

REAL	a[N];
REAL	b[N];
REAL	c[N];

REAL	reference[N]; // 

void	randomInit(REAL* v, int n, float max=1.)
{
	for (int i=0; i<N; i++ )
	{
		v[i] = (float) rand()/RAND_MAX;
	}
}

////////////////////////
// CUDA kernel
//////////////////////
const	int BLOCK_SIZE = 256;
__global__ void kAdd(REAL* c, REAL* a, REAL* b, int N)
{
	int idx = blockIdx.x*BLOCK_SIZE + threadIdx.x ;
	
	// uncomment to show the behaviour of assert...
	//if ( idx >= N )
	//	return;

	/*
	void assert(int expression);
	stops the kernel execution if expressionis equal to zero. 
	If the program is run within a debugger, 
	this triggers a breakpoint. 
	Otherwise, prints a message to stderr. 
	Look at Appendix B.16 of CUDA programming guide
	*/
	assert( idx < N );

	c[idx] = a[idx] + b[idx];
}

///////////////////////////////
/// Reference Serial Function
//////////////////////////////
void serialAdd(REAL* c, REAL* a, REAL* b, int N)
{
	for (int i=0; i<N; i++)
	{
		c[i] = a[i] + b[i];
	}
}

bool checkArray(REAL* reference, REAL* v, int n)
{
	for (int i=0; i<n; i++)
	{
		REAL delta = reference[i] - v[i];
		if (fabs(delta) > 1e-1)
		{
			return false;
		}
	}
	return true;
}



////////////////////////////////////////////////////////////////////
///////////////         M A I N
//////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

/////////////////////////
////	Array Initialization
////////////////////////
	randomInit(a, N, 5);
	randomInit(b, N, 3);

////////////////////////////////
//// Compute the reference value
///////////////////////////////
	serialAdd(reference, a, b, N);


////////////////////////
//// CUDA kernel launch
///////////////////////
	cudaError_t err;
	HANDLE_ERROR ( cudaSetDevice(0) ); 

	REAL*	d_a = NULL;
	HANDLE_ERROR ( cudaMalloc(&d_a, N*sizeof(REAL) )) ;
	HANDLE_ERROR ( cudaMemcpy(d_a, a, N*sizeof(REAL), cudaMemcpyHostToDevice ) );

	REAL*	d_b = NULL;
	cudaMalloc(&d_b, N*sizeof(REAL) );
	cudaMemcpy(d_b, b, N*sizeof(REAL), cudaMemcpyHostToDevice );

	REAL*	d_c = NULL;
	cudaMalloc(&d_c, N*sizeof(REAL) );
	cudaMemset(d_c, 0, N*sizeof(REAL) );
	dim3 blockDim = (N + BLOCK_SIZE -1)/BLOCK_SIZE;
	kAdd<<<blockDim ,BLOCK_SIZE>>>(d_c, d_a, d_b, N);

	cudaMemcpy(c, d_c, N*sizeof(REAL), cudaMemcpyDeviceToHost );

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

///////////////////////
/// Verify the results
///////////////////////

	cudaError_t error = cudaGetLastError();

	if (error == cudaErrorAssert)
    {
        printf("Device assert failed as expected, "
               "CUDA error message is: %s\n\n",
               cudaGetErrorString(error));
		return(-1);
    }

	if ( checkArray(reference, c, N) )
	{
		printf("PASSED\n");
		return (-1);
	}
	else
	{
		printf("NOT PASSED\n");
		return(1);
	}
}