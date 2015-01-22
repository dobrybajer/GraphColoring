#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Algorithm.cuh"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <Windows.h>

#define BLOCKSIZE 256
#define BLOCKSIZE2 1024 
#define BLOCKSIZE_LOOP 1024
#define PROCESSORCOUNT_FACTOR 32
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
	__host__ __device__ unsigned int Pow(int a, int n)
	{
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
	__host__ __device__ int sgnPow(int n)
	{
		return (n & 1) == 0 ? 1 : -1;
	}

	/// <summary>
	/// Funkcja zliczająca liczbę ustawionych bitów w reprezentacji bitowej wejściowej liczby.
	/// W przypadku algorytmu, służy do wyznaczania ilości elementów w aktualnie rozpatrywanym podzbiorze.
	/// </summary>
	/// <param name="n">Liczba wejściowa.</param>
	/// <returns>Liczba ustawionych bitów w danej liczbie wejściowej.</returns>
	__host__ __device__ int BitCount(int n)
	{
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
	__host__ __device__ unsigned int Combination_n_of_k(int n, int k)
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

	// Konieczne zmiany po wprowadzeniu bloków, a możliwe że i siatek (grid)
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

	// Final rozbicie na petli
	/// <summary>
	/// Przypisanie początkowych wartości do tablicy zbiorów niezależnych.
	/// Funckja wywoływana na procesorze graficznym.
	/// </summary>
	/// <param name="independentSets">Tablica zbiorów niezależnych.</param>
	/// <param name="actualVertices">Tablica zawierająca aktualnie rozpatrywane pozbiory.</param>
	/// <param name="verticesCount">Liczba wierzchołków w grafie.</param>
	__global__ void Init(int* independentSet, int* actualVertices, int verticesCount, int PowerNumber)
	{
		for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < PowerNumber; j += blockDim.x * gridDim.x)
			independentSet[j] = 0;

		int i = threadIdx.x;

		if(i < verticesCount)
		{
			independentSet[1 << i] = 1;
			actualVertices[i] = i;
		}
	}

	// Final po co na gpu
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

	// Możliwe zmiany, jeli będzie lepszy pomysł
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

	//zmiana Łukasza
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
        int startIndex=threadIdx.x;

        extern __shared__ unsigned int sd[];

        if(startIndex == BLOCKSIZE - 1)
                for(int i = 0; i < BLOCKSIZE; i++)
                        sd[i] = 0;

        syncthreads();

        unsigned  int s = 0;
        int PowerNumber = 1 << n;

        for(int i = startIndex; i < PowerNumber; )
        { 
	       sd[blockIdx.x * blockDim.x + threadIdx.x] += (unsigned int)(sgnPow(BitCount(i)) * Pow(independentSets[i], index + 1));
   			i+=BLOCKSIZE;
		}
        syncthreads();

        if(startIndex == 0)
        {
            for(int i = 0; i < BLOCKSIZE; i++)
                    s += sd[blockIdx.x * blockDim.x + i];

            wynik[index] = s > 0 ? index : -1;
        }

	}
	
	int podziel(int number)
	{
		for(int i = BLOCKSIZE2; i > 0; i--)
			if(number%i==0)
				return i;
		return 1;
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
	cudaError_t FindChromaticNumberGPU(int* wynik, int* vertices, int* offset, int verticesCount, int allVerticesCount)
	{
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

		cudaError_t cudaStatus = cudaSuccess;

		gpuErrchk(cudaMalloc((void**)&dev_vertices, allVerticesCount * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_offset, verticesCount * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_wynik, verticesCount * sizeof( int)));

		gpuErrchk(cudaMalloc((void**)&dev_independentSet, PowerNumber * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_actualVertices, verticesCount * sizeof(int)));

		gpuErrchk(cudaMemcpy(dev_vertices, vertices, allVerticesCount * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(dev_offset, offset, verticesCount * sizeof(int), cudaMemcpyHostToDevice));

    	Init<<<PROCESSORCOUNT_FACTOR*numSMs,BLOCKSIZE_LOOP>>> (dev_independentSet, dev_actualVertices, verticesCount, PowerNumber);

		for (int el = 1; el < verticesCount; el++)
		{	
			int col = el + 1;
			int row = Combination_n_of_k(verticesCount, col);
			int size = row * col;

			gpuErrchk(cudaMalloc((void**)&dev_newVertices, size * sizeof(int)));
			gpuErrchk(cudaMalloc((void**)&dev_l_set, actualVerticesRowCount * sizeof(int)));

			PrepareToNewVertices<<<1,1>>> (dev_actualVertices, dev_l_set, verticesCount, actualVerticesRowCount, actualVerticesColCount); 

			int gridSize = 1;

			if(actualVerticesRowCount > BLOCKSIZE_LOOP)
				gridSize = 32 * numSMs;

			//fprintf(stderr,"el: %d, before: %s %s %d\n", el, cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
			BuildIndependentSetGPU<<<gridSize,BLOCKSIZE_LOOP>>> (dev_l_set, verticesCount, dev_vertices, dev_offset, actualVerticesRowCount, actualVerticesColCount, dev_actualVertices, dev_newVertices, dev_independentSet);

			gpuErrchk(cudaFree(dev_l_set)); 
			gpuErrchk(cudaFree(dev_actualVertices)); 

			if(el != verticesCount - 1)
			{
				gpuErrchk(cudaMalloc((void**)&dev_actualVertices, size * sizeof(int))); 
		
				int gridSizeVer = 1;
				if (size > BLOCKSIZE_LOOP)
					gridSizeVer = 32 * numSMs;

				CreateActualVertices<<<gridSizeVer,BLOCKSIZE_LOOP>>> (dev_actualVertices, dev_newVertices, size);

				actualVerticesRowCount = row;
				actualVerticesColCount = col;
			}

			gpuErrchk(cudaFree(dev_newVertices)); 
		}

		gpuErrchk(cudaFree(dev_vertices));
		gpuErrchk(cudaFree(dev_offset));
		
		FindChromaticNumber<<<verticesCount,BLOCKSIZE,verticesCount*BLOCKSIZE*sizeof(unsigned int)>>> (verticesCount, dev_independentSet, dev_wynik);

		gpuErrchk(cudaFree(dev_independentSet));
    
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpy(wynik, dev_wynik, verticesCount * sizeof( int), cudaMemcpyDeviceToHost))
		gpuErrchk(cudaFree(dev_wynik));

		return cudaStatus;
	}
}