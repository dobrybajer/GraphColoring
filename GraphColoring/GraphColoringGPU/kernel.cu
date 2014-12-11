
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Graph.h"

#include <stdio.h>
#include <iostream>

using namespace version_cpu;
using namespace std;

cudaError_t runCuda(int *c, int *a,   int sizec, int sizea);

#pragma region Algorithm

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

	// Final
	 __host__ __device__ int sgnPow(int n)
	{
		return (n & 1) == 0 ? 1 : -1;
	}

	// Sprawdziæ, dlaczego to dzia³a
	 __host__ __device__ int BitCount(int u)
	{
		int uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
		return ((uCount + (uCount >> 3)) & 030707070707) % 63;
	}

	// Sprawdziæ, czy mo¿na lepiej
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

	  int* BuildingIndependentSets(int N ,int* Vertices,int* Offest )
	{
		int n = N;
		int* vertices = Vertices;
		int* offset = Offest;

		int* independentSets;
		int** actualVertices;
		int actualVerticesRowCount;
		int actualVerticesColCount;

		// Inicjalizacja macierzy o rozmiarze 2^N (wartoœci pocz¹tkowe 0)
		independentSets = new int[1 << n] ();

		// Krok 1 algorytmu: przypisanie wartoœci 1 (iloœæ niezale¿nych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
		//CreateActualVertices(n, 1);
		actualVertices = new int*[n];

		for (int i = 0; i < n; ++i)
			actualVertices[i] = new int[1] ();

		actualVerticesRowCount = n;
		actualVerticesColCount = 1;
		
		for (int i = 0; i < n; ++i)
		{
			independentSets[1 << i] = 1;//gubienie dla samych zer
			actualVertices[i][0] = i;
		}

		// G³ówna funkcja tworz¹ca tablicê licznoœci zbiorów niezale¿nych dla wszystkich podzbiorów zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy¿ej.
		for (int el = 1; el < n; el++)
		{
			
			int col = el + 1;
			int row = Combination_n_of_k(n, col);
			
			//int** newVertices = CreateNewVertices(row, col);
			int** newVertices = new int*[row];
			for (int i = 0; i < row; ++i)
				newVertices[i] = new int[col]();

			int l = 0;

			for (int i = 0; i < actualVerticesRowCount; ++i)
			{
				int lastIndex = 0;

				// Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
				for (int index = 0; index < actualVerticesColCount; ++index)
					lastIndex += (1 << actualVertices[i][index]);

				for (int j = actualVertices[i][actualVerticesColCount - 1] + 1; j < n; ++j)
				{
					int lastIndex2 = lastIndex;

					// Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
					for (int ns = offset[j - 1]; ns < offset[j]; ++ns)
					{
						for (int q = 0; q < actualVerticesColCount; ++q)
						{
							if (actualVertices[i][q] == vertices[ns])
							{
								lastIndex2 -= (1 << vertices[ns]);
								break;
							}
						}		
					}

					int nextIndex = lastIndex + (1 << j);

					// Liczba zbiorów niezale¿nych w aktualnie przetwarzanym podzbiorze
					independentSets[nextIndex] = independentSets[lastIndex] + independentSets[lastIndex2] + 1;

					for (int k = 0; k < el; ++k)
						newVertices[l][k] = actualVertices[i][k];

					newVertices[l][el] = j;

					l++;
				}
			}
			//UpdateActualVertices(newVertices, row, col);
			for (int i = 0; i < actualVerticesRowCount; ++i)
			{
				delete[] actualVertices[i];
			}
			delete[] actualVertices;

			actualVertices = newVertices;

			actualVerticesRowCount = row;
			actualVerticesColCount = col;
		}
		return independentSets;
	}

	__global__ void FindChromaticNumber(int N ,int* independentSets,int *wynik)
	{
		int n = N;
		int index= threadIdx.x;

			unsigned long s = 0;
			int PowerNumber = Pow(2, n);
			// Czy mo¿na omin¹æ u¿ycie funkcji BitCount ?
			for (int i = 0; i < PowerNumber; ++i) s += (sgnPow(BitCount(i)) * Pow(independentSets[i], index+1));
			
			if (s > 0)
				wynik[index]=index;
			else
				wynik[index]=s;
		
	}
#pragma endregion Algorithm

int main()
{
	Graph graph = Graph();
	graph= Graph::ReadGraph("test.txt");

	int roz=0;

	roz=1<<graph.GetVerticesCount();

	int* independentSet = BuildingIndependentSets(graph.GetVerticesCount(),graph.GetVertices(),graph.GetNeighborsCount());

	/*for (int i = 0; i <roz; i++)
	{
		cout<<independentSet[i]<<" ";
	}
	cout<<endl;*/
	int* tabWyn=new int[graph.GetVerticesCount()];

	cudaError_t cudaStatus = runCuda(tabWyn, independentSet, graph.GetVerticesCount(),roz);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	int wynik =0;
	for(int i =0 ; i<graph.GetVerticesCount();i++)
		if(tabWyn[i]!=-1 && tabWyn[i]!=0)
		{
			wynik = tabWyn[i]+1;
			break;
		}

		for(int i=0;i<graph.GetVerticesCount();i++)
			cout<<" "<<tabWyn[i];

	cout<<endl<<"Potrzeba "<<wynik<<" kolorow"<<endl;
	

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t runCuda(int *wynik, int *independentSet,   int sizeWynik, int sizeIndep)
{
    int *dev_independentSet = 0;
    int *dev_wynik = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_wynik, sizeWynik * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_independentSet, sizeIndep * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_independentSet, independentSet, sizeIndep * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	FindChromaticNumber<<<1,sizeWynik>>>(sizeWynik,dev_independentSet,dev_wynik);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(wynik, dev_wynik, sizeWynik * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_wynik);
    cudaFree(dev_independentSet);
    
    return cudaStatus;
}
