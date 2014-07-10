///////////////////////////////////////////////////////
/////////
/////////  Simple functions to explain shared memory usage in CUDA
/////////  
/////////  
/////////	Author: Francesco Caruso
/////////	francescocaruso979@gmail.com

#include <iostream>
#include <cuda_runtime.h>


const unsigned ARRAY_DIMENSION = 33*1024;
const int THREADS_PER_BLOCK = 128;

//float arrays
float*	fa_a = NULL;
float*  fa_b = NULL;

float*	gpu_fa_a = NULL;
float*	gpu_fa_b = NULL;

#define imin(a,b) (a<b?a:b)

__global__ void gpuDot(float* dot, float* a, float* b, int N)
{
	__shared__ float cache[THREADS_PER_BLOCK];
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int cacheIdx = threadIdx.x;

	float temp = 0;

	while (tid < N) 
	{
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

	cache[cacheIdx]=temp;

	__syncthreads();

	int i = blockDim.x/2;
    while (i != 0) 
	{
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        
		__syncthreads();
        i /= 2;
    }

	if (cacheIdx == 0)
		dot[blockIdx.x] = cache[0];
}

int main(int argc, char** argv)
{
	///////////////////////////////////
	//// Initialize arrays
	fa_a = new float[ARRAY_DIMENSION];
	fa_b = new float[ARRAY_DIMENSION];

	for (int i=0; i<ARRAY_DIMENSION; i++)
	{
		fa_a[i] = 10*rand()/RAND_MAX ;
		fa_b[i] = 100*rand()/RAND_MAX ;
	}

	//////////////////////////////////
	//// compute dot product on CPU
	float dot_product=0;
	for (int i=0; i<ARRAY_DIMENSION; i++)
		dot_product += fa_a[i]*fa_b[i];

	///////////////////////////////////
	//// Initialize GPU arrays
	const int ARRAY_SIZE_IN_BYTES = sizeof(float)*ARRAY_DIMENSION;
	cudaMalloc(&gpu_fa_a, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(gpu_fa_a, fa_a, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMalloc(&gpu_fa_b, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(gpu_fa_b, fa_b, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	
	// compute Grid dimension and allocate memory for the partial result
	const int blocksPerGrid = imin( 32, (ARRAY_DIMENSION + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK );

	float*	gpu_fa_partial = NULL;
	cudaMalloc(&gpu_fa_partial, blocksPerGrid*sizeof(float) );
	cudaMemset(gpu_fa_partial, 0, blocksPerGrid*sizeof(float) );
	cudaError err = cudaGetLastError();
	if (err) printf("%s\n", cudaGetErrorString(err));
	
	
	//////////////////////////////////////////////////
	///// Launch kernel, get result and clean GPU mem
	gpuDot<<< blocksPerGrid, THREADS_PER_BLOCK >>>(gpu_fa_partial, gpu_fa_a, gpu_fa_b, ARRAY_DIMENSION );

	float*	fa_partial = new float[blocksPerGrid];
	memset(fa_partial,0, sizeof(float)*blocksPerGrid);
	cudaMemcpy(fa_partial, gpu_fa_partial, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost );
	
	cudaFree(gpu_fa_a);
	cudaFree(gpu_fa_b);
	cudaFree(gpu_fa_partial);

	///////////////////////////////////
	//// Process GPU Partial result
	float gpu_dot_product = 0;
	for (int i=0; i<blocksPerGrid; i++)
		gpu_dot_product += fa_partial[i];

	if (fabs(dot_product - gpu_dot_product) > 1e-1 )
		printf("NOT PASSED.\n");
	else
		printf("PASSED.\n");

	delete[] fa_a;
	delete[] fa_b;
	delete[] fa_partial;

	return 0;
}


