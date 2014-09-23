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

#include <cassert>
#include <cmath>
#include <stdio.h>

// simple function to compute grid dimension for arrays of arbitrary size
inline int getGridSize( int n_of_threads, int threads_per_block)
{
	// Chose what you prefer....pretty same result
	
	// 1.
	return ceil( (float) n_of_threads/ threads_per_block );
	
	// 2.
	//return ( (n_of_threads + threads_per_block - 1)/threads_per_block );

	// 3.
	//return ((n_of_threads % threads_per_block) != 0) 
	//				? (n_of_threads / threads_per_block + 1) 
	//				: (n_of_threads / threads_per_block );
}

__global__ void gpu_simple_kernel(float* a, float* b, float* c, int N)
{
	//int thread_idx = threadIdx.x;

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if ( idx > N)
		return;

#define PRINT_IDS
#if !defined( __CUDA_ARCH__) || (__CUDA_ARCH__ >= 200 ) &&  defined(PRINT_IDS)
	// Check nvcc compiler gencode 
	// at least -gencode=arch=compute_20,code=\"sm_20,compute_20\" should be set
	printf("thread: %3d - block: %3d - threadIdx: %3d, warp: %3d\n", idx, blockIdx.x, threadIdx.x, threadIdx.x/warpSize );
#endif 
	
	c[idx] = a[idx] * b[idx];
}

void launch_gpu_simple_kernel(float* a, float* b, float* c, int N)
{
	const int threadsPerBlock = 64;
	const int numberOfBlocks = getGridSize(N, threadsPerBlock);
	gpu_simple_kernel<<< numberOfBlocks, threadsPerBlock>>>(a, b, c, N);
	cudaThreadSynchronize();
}