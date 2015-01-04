#ifdef UNITTEST
#define DLLIMPEXP __declspec(dllexport)
#else
#define DLLIMPEXP __declspec(dllimport)
#endif

#ifndef ALGORITHM_CUH
#define ALGORITHM_CUH

#include "cuda_runtime.h"

/// <summary>
///	Przestrzeñ nazwy dla algorytmu kolorowania grafu w wersji GPU napisanej w jêzyku CUDA C++.
/// </summary>
namespace version_gpu
{
	DLLIMPEXP __host__ __device__ unsigned long Pow(int, int);
	DLLIMPEXP __host__ __device__ int sgnPow(int);
	DLLIMPEXP __host__ __device__ int BitCount(int);
	DLLIMPEXP __host__ __device__ unsigned int Combination_n_of_k(int, int);
	DLLIMPEXP __global__ void BuildIndependentSetGPU(int*, int, int*, int*, int, int*, int*, int*);
	DLLIMPEXP __global__ void Init(int*, int*, int);
	DLLIMPEXP __global__ void CreateActualVertices(int*, int*, int);
	DLLIMPEXP __global__ void PrepareToNewVertices(int*, int*, int, int, int);
	DLLIMPEXP __global__ void FindChromaticNumber(int, int*, int*);

	extern "C" DLLIMPEXP cudaError_t FindChromaticNumberGPU(int*, int*, int*, int, int);
}

#endif 