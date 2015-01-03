#include "cuda_runtime.h"
#include "Algorithm.cuh"
#include <stdio.h>
#include <Windows.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace version_gpu
{
	__host__ __device__ unsigned long Pow(int a, int n)
	{
		unsigned long result = 1;

		while (n)
		{
			if (n & 1)
				result *= a;
			
			n >>= 1;
			a *= a;
		}

		return result;
	}

	__host__ __device__ int sgnPow(int n)
	{
		return (n & 1) == 0 ? 1 : -1;
	}

	__host__ __device__ int BitCount(int u)
	{
		int uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
		return ((uCount + (uCount >> 3)) & 030707070707) % 63;
	}

	__host__ __device__ int Combination_n_of_k(int n, int k)
	{
		if (k > n) return 0;

		int r = 1;
		for (int d = 1; d <= k; ++d)
		{
			r *= n--;
			r /= d;
		}
		return r;
	} 

	__global__ void BuildIndependentSetGPU(int* l_set, int n, int* vertices, int* offset, int actCol, int newCol, int* actualVertices, int* newVertices, int* independentSets)
	{
		int i = threadIdx.x;
		int l = l_set[i];

		if (l==-1) return;

		int lastIndex = 0;

		for (int index = 0; index < actCol; ++index)
			lastIndex += (1 << actualVertices[i * actCol + index]);

		for (int j = actualVertices[i * actCol + actCol - 1] + 1; j < n; ++j)
		{
			int lastIndex2 = lastIndex;

			for (int ns = offset[j - 1]; ns < offset[j]; ++ns)
			{
				for (int q = 0; q < actCol; ++q)
				{
					if (actualVertices[i * actCol + q] == vertices[ns])
					{
						lastIndex2 -= (1 << vertices[ns]);
						break;
					}
				}		
			}

			int nextIndex = lastIndex + (1 << j);

			independentSets[nextIndex] = independentSets[lastIndex] + independentSets[lastIndex2] + 1;

			for (int k = 0; k < newCol - 1; ++k)
				newVertices[l * newCol + k] = actualVertices[i * actCol + k];

			newVertices[l * newCol + newCol - 1] = j;
				
			l++;
		}
	}

	__global__ void Init(int* independentSet, int* actualVertices, int verticesCount)
	{
		int PowerNumber = 1 << verticesCount;

		for(int i = 0; i < PowerNumber; ++i)
			independentSet[i] = 0;

		for (int i = 0; i < verticesCount; ++i)
		{
			independentSet[1 << i] = 1;
			actualVertices[i] = i;
		}
	}

	__global__ void CreateActualVertices(int* actualVertices, int* newVertices, int size)
	{
		for(int i = 0; i < size; ++i)
			actualVertices[i] = newVertices[i];
	}

	__global__ void PrepareToNewVertices(int* actualVertices, int* l_set, int n, int actualVerticesRowCount, int actualVerticesColCount)
	{
		int last_el = 0;
		int last_index = 0;
		l_set[0] = 0;

		for(int i = 1; i < actualVerticesRowCount; ++i)
		{
			int previousPossibleCombination = n - actualVertices[last_index * actualVerticesColCount + actualVerticesColCount - 1] - 1;
			int actualPossibleCombination = n - actualVertices[i * actualVerticesColCount + actualVerticesColCount - 1] - 1;
	
			if(actualPossibleCombination <= 0)
				l_set[i] = -1;
			else
			{
				l_set[i] = last_el + previousPossibleCombination;
				last_el = l_set[i];
				last_index = i;
			}
		}
	}

	__global__ void FindChromaticNumber(int N, int* independentSets, int* wynik)
	{
		int n = N;
		int index = threadIdx.x;

		unsigned long s = 0;
		int PowerNumber = 1 << n;

		for (int i = 0; i < PowerNumber; ++i) s += (sgnPow(BitCount(i)) * Pow(independentSets[i], index + 1));
			
		wynik[index] = s > 0 ? index : s; // KAMIL: punkt krytyczny, czy dobrze jest liczone "s"? dla unsigned long long liczy Ÿle...
	}
	
	cudaError_t FindChromaticNumberMain(int* wynik, int* vertices, int* offset, int verticesCount, int allVerticesCount)
	{
		int* dev_vertices = 0;
		int* dev_offset = 0;
		int* dev_wynik = 0;

		int* dev_independentSet = 0;
		int* dev_actualVertices = 0;
		int* dev_newVertices = 0;
		int* dev_l_set = 0;
		int actualVerticesRowCount = verticesCount;
		int actualVerticesColCount = 1;
		int PowerNumber = 1 << verticesCount;

		cudaError_t cudaStatus = cudaSuccess;

		gpuErrchk(cudaSetDevice(0));

		gpuErrchk(cudaMalloc((void**)&dev_vertices, allVerticesCount * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_offset, verticesCount * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_wynik, verticesCount * sizeof(int)));

		gpuErrchk(cudaMalloc((void**)&dev_independentSet, PowerNumber * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_actualVertices, verticesCount * sizeof(int)));

		gpuErrchk(cudaMemcpy(dev_vertices, vertices, allVerticesCount * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(dev_offset, offset, verticesCount * sizeof(int), cudaMemcpyHostToDevice));
    
		Init<<<1,1>>> (dev_independentSet, dev_actualVertices, verticesCount); // czy warto odpaliæ na wiêkszej iloœci w¹tków? (wpisywanie du¿ej iloœci zer)

		for (int el = 1; el < verticesCount; el++) // przy tej konstrukcji alg nie damy rady odpaliæ tej pêtli równolegle
		{	
			int col = el + 1;
			int row = Combination_n_of_k(verticesCount, col);

			gpuErrchk(cudaMalloc((void**)&dev_newVertices, (row * col) * sizeof(int)));
			gpuErrchk(cudaMalloc((void**)&dev_l_set, actualVerticesRowCount * sizeof(int)));
		
			PrepareToNewVertices<<<1,1>>> (dev_actualVertices, dev_l_set, verticesCount, actualVerticesRowCount, actualVerticesColCount); // przy tej konstrukcji funkcji nie damy rady odpaliæ tego na wielu w¹tkach

			BuildIndependentSetGPU<<<1,actualVerticesRowCount>>> (dev_l_set, verticesCount, dev_vertices, dev_offset, actualVerticesColCount, col, dev_actualVertices, dev_newVertices, dev_independentSet); // Koniecznie trzeba odpalaæ tak¿e u¿ywaj¹c bloków. Max w¹tków per blok to np. 1024, a s¹ sytuacje gdzie podawane jest ponad 180k (dla n=20)	

			cudaFree(dev_actualVertices); // czy aby na pewno dobrze jest pamiec zwalniana? nie marnujemy zasobow karty?
			gpuErrchk(cudaMalloc((void**)&dev_actualVertices, (row * col) * sizeof(int))); // czy ponowne mallocowanie jest ok jeœli wczeœniej u¿yto cudaFree?

			CreateActualVertices<<<1,1>>> (dev_actualVertices, dev_newVertices, row * col);

			actualVerticesRowCount = row;
			actualVerticesColCount = col;
		}
	
		FindChromaticNumber<<<1,verticesCount>>> (verticesCount, dev_independentSet, dev_wynik); // Mo¿liwe odpalenie bloków, czyli zrobienie Reduce dla pewnych kawa³ków ca³ej sumy. Ponadto komunikacja- przerwanie obliczeñ natychmiast, gdy jakiœ w¹tek/blok da³ pozytywn¹ odpowiedŸ
	
		gpuErrchk(cudaGetLastError());
    
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpy(wynik, dev_wynik, verticesCount * sizeof(int), cudaMemcpyDeviceToHost));

		return cudaStatus;
	}
}