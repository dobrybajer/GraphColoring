#ifndef ALGORITHM_CUH
#define ALGORITHM_CUH

#include "cuda_runtime.h"

/// <summary>
///	Przestrze� nazwy dla algorytmu kolorowania grafu w wersji GPU napisanej w j�zyku CUDA C++.
/// </summary>
namespace version_gpu
{
	__declspec(dllexport) __host__ __device__ unsigned long Pow(int, int);
	__declspec(dllexport) __host__ __device__ int sgnPow(int);
	__declspec(dllexport) __host__ __device__ int BitCount(int);
	__declspec(dllexport) __host__ __device__ unsigned int Combination_n_of_k(int, int);
	__declspec(dllexport) __global__ void BuildIndependentSetGPU(int*, int, int*, int*, int, int*, int*, int*);
	__declspec(dllexport) __global__ void Init(int*, int*, int);
	__declspec(dllexport) __global__ void CreateActualVertices(int*, int*, int);
	__declspec(dllexport) __global__ void PrepareToNewVertices(int*, int*, int, int, int);
	__declspec(dllexport) __global__ void FindChromaticNumber(int, int*, int*);

	extern "C" __declspec(dllexport) cudaError_t FindChromaticNumberGPU(int*, int*, int*, int, int);
}

#endif 