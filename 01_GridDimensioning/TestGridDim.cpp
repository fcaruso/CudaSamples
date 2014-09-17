///////////////////////////////////////////////////////
/////////
/////////  Simple functions to explain simple concepts 
/////////  regarding GPU Computing with NVIDIA CUDA
/////////  
/////////	Topics:
/////////	- dimensioning thread batches and grid of tbread blocks
/////////	- memory operations
/////////	- printf within kernels
/////////   - identify threads, warps and blocks 
/////////	
/////////	Author: Francesco Caruso
/////////	francescocaruso979@gmail.com

#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

const int N = 200;

//float arrays
float*	fa_a = NULL;
float*  fa_b = NULL;
float*  fa_c = NULL;
float*	fa_computed_on_gpu = NULL;

float*	gpu_fa_a = NULL;
float*	gpu_fa_b = NULL;
float*	gpu_fa_c = NULL;

const int ARRAY_SIZE_IN_BYTES = N*sizeof(float);

void launch_gpu_simple_kernel(float* a, float* b, float* c, int N);

void simple_cpu_func(float* a, float* b, float* c, int N)
{
	for (int i=0; i<N; i++)
	{
		c[i] = a[i]*b[i];
	}
}

/////////////////////////////////////////////////////
//// Simple function to initialize arrays with random float
//// on both CPU and GPU memory
void set_arrays(float*& a, float*& b, float*& c, 
		float*& gpu_a, float*& gpu_b, float*& gpu_c,
	int N)
{
	a = (float*) malloc(ARRAY_SIZE_IN_BYTES);
	b = (float*) malloc(ARRAY_SIZE_IN_BYTES);
	c = (float*) malloc(ARRAY_SIZE_IN_BYTES);

	for (int i=0; i<N; i++)
	{
		a[i] = 10*rand()/RAND_MAX ;
		b[i] = 100*rand()/RAND_MAX ;
		c[i] = 0.f; // consider using memset 
	}

	cudaError err;
	
	err = cudaMalloc((void**) &gpu_a, ARRAY_SIZE_IN_BYTES);
	assert (err==cudaSuccess);

	cudaMalloc((void**) &gpu_b, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**) &gpu_c, ARRAY_SIZE_IN_BYTES);

	cudaMemcpy(gpu_a, a, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy(gpu_b, b, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy(gpu_c, c, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );// conside using cudaMemset
}

///////////////////////////////////
//  release GPU and HOST memory
void dispose_arrays(
	float*& a, float*& b, float*& c,
	float*& gpu_a, float*& gpu_b, float*& gpu_c)
{
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);

	free(a);
	free(b);
	free(c);
}

int main(int argc, char* argv[])
{
	// Initialize arrays
	set_arrays(
		fa_a, fa_b, fa_c, 
		gpu_fa_a, gpu_fa_b, gpu_fa_c,
		N);


	// compute....
	launch_gpu_simple_kernel(gpu_fa_a, gpu_fa_b, gpu_fa_c, N);
	simple_cpu_func(fa_a, fa_b, fa_c, N);

	// get computation results
	fa_computed_on_gpu = (float*) malloc(ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(fa_computed_on_gpu, gpu_fa_c, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost); 
	// check them
	bool passed=true;
	for (int i=0; i<N; i++)
	{
		if( fabs(fa_computed_on_gpu[i] - fa_c[i] ) > 1e-3 )
		{
			passed=false;
			break;
		}
	}
	if ( passed )
		printf("TEST PASSED\n");
	else
		printf("TEST NOT PASSED\n");

	// clean up
	dispose_arrays(
		fa_a, fa_b, fa_c, 
		gpu_fa_a, gpu_fa_b, gpu_fa_c);
	free(fa_computed_on_gpu);


	return 0;
}