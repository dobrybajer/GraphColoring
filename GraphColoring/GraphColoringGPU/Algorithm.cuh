#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "cuda_runtime.h"

namespace version_gpu
{
	extern "C"
	{
		__declspec(dllexport) __host__ __device__ unsigned long Pow(int, int);
		__declspec(dllexport) __host__ __device__ int sgnPow(int);
		__declspec(dllexport) __host__ __device__ int BitCount(int);
		__declspec(dllexport) __host__ __device__ int Combination_n_of_k(int, int);
		__declspec(dllexport) cudaError_t FindChromaticNumberMain(int*, int*, int*, int, int);
	};
}

#endif 