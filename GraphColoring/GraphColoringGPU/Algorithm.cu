#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Algorithm.cuh"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <Windows.h>

#define BLOCKSIZE 512
#define BLOCKSIZE_LOOP 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/// <summary>
/// Przestrzeń nazwy dla algorytmu kolorowania grafów w wersji GPU napisanej w języku CUDA C++.
/// </summary>
namespace version_gpu
{
	/// <summary>
	/// Szybkie podnoszenie danej liczby do podanej potęgi. Pozwala na potęgowanie liczby, której
	/// wynik jest nie większy od rozmiaru typu INT.
	/// </summary>
	/// <param name="a">Podstawa potęgi.</param>
	/// <param name="n">Wykładnik potęgi.</param>
	/// <returns>Wynik potęgowania.</returns>
	__device__ unsigned int Pow(int a, int n)
	{
		if (n <= 0) return 1;
		if (n == 1) return a;
		if (a <= 0) return 0;
		
		unsigned int result = 1;

		while (n)
		{
			if (n & 1)
				result *= a;
		
			n >>= 1;
			a *= a;
		}

		return result;
	}

	/// <summary>
	/// Szybkie i efektywne podnoszenie do potęgi liczby -1. Polega na sprawdzaniu parzystości
	/// wykładnika potęgi.
	/// </summary>
	/// <param name="n">Wykładnik potęgi.</param>
	/// <returns>Wynik potęgowania.</returns>
	__device__ int sgnPow(int n)
	{
		return (n & 1) == 0 ? 1 : -1;
	}

	/// <summary>
	/// Funkcja zliczająca liczbę ustawionych bitów w reprezentacji bitowej wejściowej liczby.
	/// W przypadku algorytmu, służy do wyznaczania ilości elementów w aktualnie rozpatrywanym podzbiorze.
	/// </summary>
	/// <param name="n">Liczba wejściowa.</param>
	/// <returns>Liczba ustawionych bitów w danej liczbie wejściowej.</returns>
	__device__ int BitCount(int n)
	{
		if (n <= 0) return 0;

		int uCount = n - ((n >> 1) & 033333333333) - ((n >> 2) & 011111111111);
		return ((uCount + (uCount >> 3)) & 030707070707) % 63;
	}

	/// <summary>
	/// Wyznaczanie kombinacji k - elementowych zbioru n - elementowego (kombinacje bez powtórzeń).
	/// Ograniczone możliwości ze względu na możliwie zbyt dużą wielkość wyniku.
	/// </summary>
	/// <param name="n">Liczba elementów w zbiorze.</param>
	/// <param name="k">Liczba elementów w kombinacji.</param>
	/// <returns>Liczba oznaczająca kombinację n po k.</returns>
	__device__ unsigned int Combination_n_of_k(int n, int k)
	{
		if (k > n) return 0;
		if (k == 0 || k == n) return 1;

		if (k * 2 > n) k = n - k;

		unsigned int r = 1;
		for (int d = 1; d <= k; ++d) 
		{
			r *= n--;
			r /= d;
		}
		return r;
	} 

	/// <summary>
	/// Przypisanie początkowych wartości do tablicy zbiorów niezależnych.
	/// Funckja wywoływana na procesorze graficznym.
	/// </summary>
	/// <param name="independentSets">Tablica zbiorów niezależnych.</param>
	/// <param name="actualVertices">Tablica zawierająca aktualnie rozpatrywane pozbiory.</param>
	/// <param name="verticesCount">Liczba wierzchołków w grafie.</param>
	__global__ void Init(int* independentSet, int* actualVertices)
	{
		independentSet[1 << threadIdx.x] = 1;
		actualVertices[threadIdx.x] = threadIdx.x;
	}

	/// <summary>
	/// Tworzenie tablicy zawierającej indeksy początkowe, w których każdy wątek 
	/// powinien zacząć wpisywać dane do tablicy zbiorów niezależnych.
	/// Funckja wywoływana na procesorze graficznym.
	/// </summary>
	/// <param name="actualVertices">Tablica zawierająca aktualnie rozpatrywane pozbiory.</param>
	/// <param name="l_set">
	/// Tablica zawierająca indeksy początkowe, w których każdy wątek 
	/// powinien zacząć wpisywać dane do tablicy zbiorów niezależnych.
	/// </param>
	/// <param name="n">Liczba wierzchołków w grafie.</param>
	/// <param name="actualVerticesRowCount">Liczba aktualnie rozpatrywanych podzbiorów.</param>
	/// <param name="actualVerticesColCount">Liczba elementów w każdym z aktualnie rozpatrywanych podzbiorów.</param>
	__global__ void PrepareToNewVertices(int* actualVertices, int* l_set, int n, int actualVerticesRowCount, int actualVerticesColCount)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < actualVerticesRowCount; i += blockDim.x * gridDim.x) 
		{
			int index = i == 1 ? i - 1: i;
			int value = n - actualVertices[index * actualVerticesColCount + actualVerticesColCount - 1] - 1;
		
			if(i == 0 || i == actualVerticesRowCount - 1 || i > 1 && value == 0) 
				l_set[i] = 0;
			else if(i == 1)
				l_set[i] = value;
			else 
			{
				int q;
				do
				{
					index--;
					q = n - actualVertices[index * actualVerticesColCount + actualVerticesColCount - 1] - 1;
				}
				while(q == 0);
	
				l_set[i] = index == 0 ? 0 : n - actualVertices[index * actualVerticesColCount + actualVerticesColCount - 1] - 1;
			}
		}
	}

	__global__ void Sum(int * input, int *Sums, int numElements) 
	{
		extern __shared__ int scan_array[];
	
		unsigned int thid = threadIdx.x; 
		unsigned int start = 2 * blockIdx.x * blockDim.x;
	    
		scan_array[thid] = start + thid < numElements ? input[start + thid] : 0;
		scan_array[blockDim.x + thid] = start + blockDim.x + thid < numElements ? input[start + blockDim.x + thid] : 0;
	
		__syncthreads();

		// Reduction
		int offset;
		for (offset = 1; offset <= blockDim.x; offset <<= 1) 
		{
		   int index = (thid + 1) * offset * 2 - 1;
	   
		   if (index < 2 * blockDim.x)
			  scan_array[index] += scan_array[index - offset];
	  
		   __syncthreads();
		}
 
		// Post reduction
		for (offset = blockDim.x >> 1; offset; offset >>= 1) 
		{
		   int index = (thid + 1) * offset * 2 - 1;
	   
		   if (index + offset < 2 * blockDim.x)
			  scan_array[index + offset] += scan_array[index];
	  
		   __syncthreads();
		}

		if (start + thid < numElements && input[start + thid] != 0) 
			input[start + thid] = scan_array[thid];
		else if(start + thid < numElements) 
			input[start + thid] = 0;

		if (start + blockDim.x + thid < numElements && input[start+blockDim.x + thid] != 0)
		   input[start + blockDim.x + thid] = scan_array[blockDim.x + thid];
		else if(start + blockDim.x + thid < numElements)
			input[start+blockDim.x + thid] = 0;		

		if (Sums && thid == 0)
		   Sums[blockIdx.x] = scan_array[2 * blockDim.x - 1];
	}

	__global__ void SumsPar(int *Sums, int counter)
	{
		for(int i = 0; i < counter - 1; i++)
			Sums[i + 1] += Sums[i];
	}

	__global__ void Resum(int *g_odata, int *Sums, int n)
	{
		int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
		int index = 2 * (blockDim.x + blockDim.x * blockIdx.x) - 1;

		if (blockIdx.x != 0)
		{
			if (g_odata[2 * workIndex] != 0 && workIndex < n)
				g_odata[2 * workIndex] += Sums[blockIdx.x - 1];

			if (g_odata[2 * workIndex + 1] != 0 && index != 2 * workIndex + 1 && workIndex + 1 < n)
				g_odata[2 * workIndex + 1] += Sums[blockIdx.x - 1];
		}
	
		if (2 * (blockDim.x + blockDim.x * blockIdx.x) - 1 < n && g_odata[2 * (blockDim.x + (blockDim.x) * blockIdx.x) - 1] != 0)
			g_odata[2 * (blockDim.x + blockDim.x * blockIdx.x) - 1] = Sums[blockIdx.x];
	}

	/// <summary>
	/// Obliczanie i uzupełnianie pewnej części tablicy zbiorów niezależnych dla danego grafu. 
	/// Funckja wywoływana na procesorze graficznym.
	/// </summary>
	/// <param name="l_set">
	/// Tabela zawierająca indeksy początkowe, w których każdy wątek 
	/// powinien zacząć wpisywać dane do tablicy zbiorów niezależnych.
	/// </param>
	/// <param name="n">Liczba wierzchołków w grafie.</param>
	/// <param name="vertices">Tablica wszystkich sąsiadów każdego z wierzchołków.</param>
	/// <param name="offset">Tablica pozycji początkowych sąsiadów dla danego wierzchołka.</param>
	/// <param name="actCol">Moc aktualnie rozpatrywanego podzbioru potęgowego.</param>
	/// <param name="newCol">Moc kolejnego rozpatrywanego podzbioru potęgowego.</param>
	/// <param name="actualVertices">Tablica zawierająca aktualnie rozpatrywane pozbiory.</param>
	/// <param name="newVertices">Tablica zawierająca rozpatrywane pozbiory dla następnej iteracji (tworzona w tej funkcji).</param>
	/// <param name="independentSets">Tablica zbiorów niezależnych.</param>
	__global__ void BuildIndependentSetGPU(int* l_set, int n, int* vertices, int* offset, int actRow, int actCol, int* actualVertices, int* newVertices, int* independentSets)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < actRow; i += blockDim.x * gridDim.x) 
	    {
			int l = l_set[i];

			if (i != 0 && l == 0)
				continue;

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

				for (int k = 0; k < actCol; ++k)
					newVertices[l * (actCol + 1) + k] = actualVertices[i * actCol  + k];

				newVertices[l * (actCol + 1) + actCol] = j;
					
				l++;
			}
		}
	}

	/// <summary>
	/// Uzupełnianie aktualnie rozpatrywanego podzbioru wartościami wyliczonymi z bieżącej iteracji.
	/// Funckja wywoływana na procesorze graficznym.
	/// </summary>
	/// <param name="actualVertices">Tablica zawierająca aktualnie rozpatrywane pozbiory.</param>
	/// <param name="newVertices">Tablica zawierająca rozpatrywane pozbiory dla następnej iteracji.</param>
	/// <param name="size">Liczba elementów w rozpatrywanym zbiorze dla kolejnej iteracji.</param>
	__global__ void CreateActualVertices(int* actualVertices, int* newVertices, int size)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) 
			actualVertices[i] = newVertices[i];
	}

	/// <summary>
	/// Główna funkcja obliczająca wynik dla odpowiedniego indeksu, a następnie wpisująca do wyjściowej tablicy wynik.
	/// Funckja wywoływana na procesorze graficznym.
	/// </summary>
	/// <param name="n">Liczba wierzchołków w grafie.</param>
	/// <param name="independentSets">Tablica zbiorów niezależnych.</param>
	/// <param name="wynik">Tablica zawierająca wynik dla każdego z k kolorów (tworzona w tej funkcji).</param>
	__global__ void FindChromaticNumber(int n, int* independentSets, int* wynik)
	{
		int index = blockIdx.x;
        int startIndex = threadIdx.x;

        extern __shared__ unsigned int sd[];

        if(startIndex == BLOCKSIZE - 1)
                for(int i = 0; i < BLOCKSIZE; i++)
                        sd[i] = 0;

        __syncthreads();

        int PowerNumber = 1 << n;

        for(int i = startIndex; i < PowerNumber; i += BLOCKSIZE)
	       sd[threadIdx.x] += (sgnPow(BitCount(i)) * Pow(independentSets[i], index + 1));
   			
        __syncthreads();

        for(int offset = 1; offset < blockDim.x; offset *= 2)
		{
			if(startIndex == 0 || startIndex % (offset * 2) == 0)
				sd[startIndex] += sd[startIndex + offset];
			
			__syncthreads();
		}
	
		if(startIndex == 0)
			wynik[index] = sd[startIndex] > 0 ? index : -1;
	}

	bool PredictSpace(int n, const double ToMb, double* info)
	{
		size_t mem_free = 0;

		cudaMemGetInfo(&mem_free, NULL);
		info[0] = mem_free / ToMb;

		unsigned int vertices = Combination_n_of_k(n, n / 2);
        int maxColumnCount = (n + 1) / 2;
        info[1] = ((1 << n) + 2 * vertices * maxColumnCount + vertices) / ToMb * sizeof(int);

		return info[1] > info[0];
	}

	// Do sprawdzenia szczególnie kwestia alokowanej i zwalnianej pamięci
	/// <summary>
	/// Funkcja uruchamiająca cały przebieg algorytmu. Wykorzystuje pozostałe funkcje w celu obliczenia tablicy
	/// zbiorów niezależnych, a następnie policzenia wyniku dla każdego z k kolorów.
	/// </summary>
	/// <param name="wynik">Tablica zawierająca wynik dla każdego z k kolorów.</param>
	/// <param name="vertices">Tablica wszystkich sąsiadów każdego z wierzchołków.</param>
	/// <param name="offset">Tablica pozycji początkowych sąsiadów dla danego wierzchołka.</param>
	/// <param name="verticesCount">Liczba wierzchołków w grafie.</param>
	/// <param name="allVerticesCount">Liczba wszystkich sąsiadów każdego z wierzchołków.</param>
	/// <returns>Status wykonania funkcji na procesorze graficznym.</returns>
	cudaError_t FindChromaticNumberGPU(int* wynik, double* pamiec, int* vertices, int* offset, int verticesCount, int allVerticesCount)
	{
		const double ToMb = 1048576;

		double* info = new double[2];
		if(PredictSpace(verticesCount, ToMb, info))
		{
			pamiec[0] = -666;
			pamiec[1] = info[0];
			pamiec[2] = info[1];

			return cudaErrorMemoryAllocation;
		}

		int numSMs;
		cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

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
		
		size_t mem_tot = 0;
		size_t mem_free = 0;
		
		cudaError_t cudaStatus = cudaSuccess;

		gpuErrchk(cudaMalloc((void**)&dev_vertices, allVerticesCount * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_offset, verticesCount * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_wynik, verticesCount * sizeof( int)));

		cudaMemGetInfo(&mem_free, &mem_tot);
		pamiec[0] = (mem_tot - mem_free) / ToMb;

		gpuErrchk(cudaMalloc((void**)&dev_independentSet, PowerNumber * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_actualVertices, verticesCount * sizeof(int)));

		gpuErrchk(cudaMemcpy(dev_vertices, vertices, allVerticesCount * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(dev_offset, offset, verticesCount * sizeof(int), cudaMemcpyHostToDevice));

    	cudaMemset(dev_independentSet, 0, PowerNumber * sizeof(int));
		Init<<<1,verticesCount>>> (dev_independentSet, dev_actualVertices);

		for (int el = 1; el < verticesCount; el++)
		{	
			int col = el + 1;
			int row = Combination_n_of_k(verticesCount, col);
			int size = row * col;

			gpuErrchk(cudaMalloc((void**)&dev_newVertices, size * sizeof(int)));
			gpuErrchk(cudaMalloc((void**)&dev_l_set, actualVerticesRowCount * sizeof(int)));

			int gridSize = 1;

			if(actualVerticesRowCount > BLOCKSIZE_LOOP)
				gridSize = 32 * numSMs;
		
			dim3 dimBlockBuild(BLOCKSIZE_LOOP);
			dim3 dimGridBuild(gridSize);
		
			PrepareToNewVertices<<<dimGridBuild,dimBlockBuild>>> (dev_actualVertices, dev_l_set, verticesCount, actualVerticesRowCount, actualVerticesColCount); 

			int counter = 0;
			int* sums;
	
			counter = actualVerticesRowCount % BLOCKSIZE == 0 ? actualVerticesRowCount / BLOCKSIZE : actualVerticesRowCount / BLOCKSIZE + 1;
		
			cudaMalloc((void**)&sums, counter * sizeof(int));
 
			Sum<<<counter,BLOCKSIZE/2,BLOCKSIZE*2*sizeof(int)>>> (dev_l_set, sums, actualVerticesRowCount);
			cudaDeviceSynchronize();
			SumsPar<<<1,1>>>(sums,counter);
			cudaDeviceSynchronize();
			Resum<<<counter,BLOCKSIZE/2>>> (dev_l_set, sums, actualVerticesRowCount);

			BuildIndependentSetGPU<<<dimGridBuild,dimBlockBuild>>> (dev_l_set, verticesCount, dev_vertices, dev_offset, actualVerticesRowCount, actualVerticesColCount, dev_actualVertices, dev_newVertices, dev_independentSet);

			cudaMemGetInfo(&mem_free, &mem_tot);
			pamiec[el] = (mem_tot - mem_free) / ToMb;

			gpuErrchk(cudaFree(dev_l_set)); 
			gpuErrchk(cudaFree(dev_actualVertices)); 

			if(el != verticesCount - 1)
			{
				gpuErrchk(cudaMalloc((void**)&dev_actualVertices, size * sizeof(int))); 
				gpuErrchk(cudaMemcpy(dev_actualVertices, dev_newVertices, size * sizeof(int), cudaMemcpyDeviceToDevice));

				actualVerticesRowCount = row;
				actualVerticesColCount = col;
			}

			gpuErrchk(cudaFree(dev_newVertices)); 
		}

		gpuErrchk(cudaFree(dev_vertices));
		gpuErrchk(cudaFree(dev_offset));

		cudaMemGetInfo(&mem_free, &mem_tot);
		pamiec[verticesCount] = (mem_tot - mem_free) / ToMb;
		
		dim3 dimBlockChro(BLOCKSIZE);
		dim3 dimGridChro(verticesCount);

		FindChromaticNumber<<<dimGridChro,dimBlockChro,BLOCKSIZE*sizeof(unsigned int)>>> (verticesCount, dev_independentSet, dev_wynik);

		cudaMemGetInfo(&mem_free, &mem_tot);
		pamiec[verticesCount + 1] = (mem_tot - mem_free) / ToMb;

		gpuErrchk(cudaFree(dev_independentSet));
    
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpy(wynik, dev_wynik, verticesCount * sizeof(int), cudaMemcpyDeviceToHost))
		gpuErrchk(cudaFree(dev_wynik));

		cudaMemGetInfo(&mem_free, &mem_tot);
		pamiec[verticesCount + 2] = (mem_tot - mem_free) / ToMb;

		return cudaStatus;
	}
}