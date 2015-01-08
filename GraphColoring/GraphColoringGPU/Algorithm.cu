#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Algorithm.cuh"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <Windows.h>

#define BLOCKSIZE 256
#define BLOCKSIZE2 1024 
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
	__global__ void BuildIndependentSetGPU(int* l_set, int n, int* vertices, int* offset, int actCol, int* actualVertices, int* newVertices, int* independentSets)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
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

			for (int k = 0; k < actCol; ++k)
				newVertices[l * (actCol + 1) + k] = actualVertices[i * actCol + k];

			newVertices[l * (actCol + 1) + actCol] = j;
				
			l++;
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
		//for(int i = threadIdx.x; i < PowerNumber; i += BLOCKSIZE)
		//	independentSet[i] = 0;
		for(int i = 0; i < PowerNumber; i++)
			independentSet[i] = 0;
		//__syncthreads();
		//int i = threadIdx.x;
		//if(threadIdx.x == 0)
		for(int i = 0; i < verticesCount; i++)
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
		int i = blockIdx.x * blockDim.x + threadIdx.x;
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
		//poprawic kod, mozna lepiej
		int index = blockIdx.x;
		int startIndex = threadIdx.x;
		 __shared__ unsigned int sd[BLOCKSIZE];		
	
		if(startIndex == BLOCKSIZE - 1)		
 			for(int i = 0; i < BLOCKSIZE; i++)
				sd[i] = 0;

		__syncthreads();

		unsigned int s = 0;
		int PowerNumber = 1 << n;
	
		for(int i = startIndex; i < PowerNumber; i += BLOCKSIZE) 
		{
			sd[startIndex] += (sgnPow(BitCount(i)) * Pow(independentSets[i], index + 1));
			__syncthreads();
		}

		__syncthreads();

		if(startIndex == BLOCKSIZE - 1)
		{
			for(int i = 0; i < BLOCKSIZE; i++)
				s+=sd[i];

			wynik[index] = s > 0 ? index : s;
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

		int blockSize = 2;
		dim3 dimBlock(blockSize);
		dim3 dimBlockVer(BLOCKSIZE);	
		dim3 dimGridInit(verticesCount);

		gpuErrchk(cudaSetDevice(0));

		gpuErrchk(cudaMalloc((void**)&dev_vertices, allVerticesCount * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_offset, verticesCount * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_wynik, verticesCount * sizeof(int)));

		gpuErrchk(cudaMalloc((void**)&dev_independentSet, PowerNumber * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_actualVertices, verticesCount * sizeof(int)));

		gpuErrchk(cudaMemcpy(dev_vertices, vertices, allVerticesCount * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(dev_offset, offset, verticesCount * sizeof(int), cudaMemcpyHostToDevice));

    	Init<<<1,1>>> (dev_independentSet, dev_actualVertices, verticesCount, 1 << verticesCount); // czy warto odpalić na większej ilości wątków? (wpisywanie dużej ilości zer)

		//Init<<<dimGridInit,dimBlockVer>>> (dev_independentSet, dev_actualVertices, verticesCount, 1 << verticesCount); // czy warto odpalić na większej ilości wątków? (wpisywanie dużej ilości zer)
		//int* tab = new int[(1<<verticesCount)];
		//gpuErrchk(cudaMemcpy(tab, dev_independentSet, (1<<verticesCount) * sizeof(int), cudaMemcpyDeviceToHost));
		//for(int i = 0; i < (1 << verticesCount);i++)
		//	std::cout<<tab[i]<<",";
		//std::cout<<std::endl;
		//cudaThreadSynchronize();

		for (int el = 1; el < verticesCount; el++) // przy tej konstrukcji alg nie damy rady odpalić tej pętli równolegle
		{	
			int col = el + 1;
			int row = Combination_n_of_k(verticesCount, col);

			gpuErrchk(cudaMalloc((void**)&dev_newVertices, (row * col) * sizeof(int)));
			gpuErrchk(cudaMalloc((void**)&dev_l_set, actualVerticesRowCount * sizeof(int)));
		
			PrepareToNewVertices<<<1,1>>> (dev_actualVertices, dev_l_set, verticesCount, actualVerticesRowCount, actualVerticesColCount); // przy tej konstrukcji funkcji nie damy rady odpalić tego na wielu wątkach

			dim3 dimGrid(actualVerticesRowCount/2);

			if(actualVerticesRowCount < BLOCKSIZE2)
			{
				dim3 dimBlockBrute(actualVerticesRowCount);
				dim3 dimGridBrute(1);

				BuildIndependentSetGPU<<<dimGridBrute,dimBlockBrute>>> (dev_l_set, verticesCount, dev_vertices, dev_offset, actualVerticesColCount, dev_actualVertices, dev_newVertices, dev_independentSet); // Koniecznie trzeba odpalać także używając bloków. Max wątków per blok to np. 1024, a są sytuacje gdzie podawane jest ponad 180k (dla n=20)	
			}
			else
			{
				int threads = podziel(actualVerticesRowCount);
				dim3 dimBlockBrute(threads);
				dim3 dimGridBrute(actualVerticesRowCount/threads);

				BuildIndependentSetGPU<<<dimGridBrute,dimBlockBrute>>> (dev_l_set, verticesCount, dev_vertices, dev_offset, actualVerticesColCount, dev_actualVertices, dev_newVertices, dev_independentSet); // Koniecznie trzeba odpalać także używając bloków. Max wątków per blok to np. 1024, a są sytuacje gdzie podawane jest ponad 180k (dla n=20)
			}

			cudaFree(dev_actualVertices); // czy aby na pewno dobrze jest pamiec zwalniana? nie marnujemy zasobow karty?
			gpuErrchk(cudaMalloc((void**)&dev_actualVertices, (row * col) * sizeof(int))); // czy ponowne mallocowanie jest ok jeśli wcześniej użyto cudaFree?

			dim3 dimGridVer(ceil((double)((double)row * col) / (double)BLOCKSIZE));
			CreateActualVertices<<<dimGridVer,dimBlockVer>>> (dev_actualVertices, dev_newVertices, row * col);

			actualVerticesRowCount = row;
			actualVerticesColCount = col;
		}

		dim3 dimBlockChro(BLOCKSIZE);
		dim3 dimGridChro(verticesCount);
	
		FindChromaticNumber<<<dimGridChro,dimBlockChro>>> (verticesCount, dev_independentSet, dev_wynik); // Możliwe odpalenie bloków, czyli zrobienie Reduce dla pewnych kawałków całej sumy. Ponadto komunikacja- przerwanie obliczeń natychmiast, gdy jakiś wątek/blok dał pozytywną odpowiedź
	
		gpuErrchk(cudaGetLastError());
    
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpy(wynik, dev_wynik, verticesCount * sizeof(int), cudaMemcpyDeviceToHost));

		return cudaStatus;
	}
}