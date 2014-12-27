
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Graph.h"

#include <stdio.h>
#include <iostream>

using namespace version_cpu;
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

cudaError_t runCuda(int *c, int *a,   int sizec, int sizea);
int* BuildingIndependentSetsGPU(int N ,int* Vertices,int* Offest, int verticesLength);

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
		int* actualVertices;//zmiana na tablice jedno wymiarow¹ 
		int actualVerticesRowCount;
		int actualVerticesColCount;

		// Inicjalizacja macierzy o rozmiarze 2^N (wartoœci pocz¹tkowe 0)
		independentSets = new int[1 << n] ();

		// Krok 1 algorytmu: przypisanie wartoœci 1 (iloœæ niezale¿nych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
		actualVertices = new int[n];

		actualVerticesRowCount = n;//oldRow	
		actualVerticesColCount = 1;//oldCol
		
		for (int i = 0; i < n; ++i)
		{
			independentSets[1 << i] = 1;
			actualVertices[i] = i;
		}

		// G³ówna funkcja tworz¹ca tablicê licznoœci zbiorów niezale¿nych dla wszystkich podzbiorów zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy¿ej.
		for (int el = 1; el < n; el++)
		{
						
			int col = el + 1;
			int row = Combination_n_of_k(n, col);
			int* newVertices = new int[row*col];//zmiana na tablice jedno wymiarow¹ 
			int l = 0;

			for (int i = 0; i < actualVerticesRowCount; ++i)
			{
				int lastIndex = 0;
				// Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
				for (int index = 0; index < actualVerticesColCount; ++index)
					lastIndex += (1 << actualVertices[i*actualVerticesColCount + index]);
				for (int j = actualVertices[i*actualVerticesColCount + actualVerticesColCount - 1] + 1; j < n; ++j)
				{
					int lastIndex2 = lastIndex;
					// Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
					for (int ns = offset[j - 1]; ns < offset[j]; ++ns)
					{
						for (int q = 0; q < actualVerticesColCount; ++q)
						{
							if (actualVertices[i * actualVerticesColCount + q] == vertices[ns])
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
						newVertices[l*col + k] = actualVertices[i * actualVerticesColCount + k];
					newVertices[l * col + el] = j;

					l++;
				}
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

	__global__ void IndependentSetGPU(int N ,int* Vertices,int* Offset ,int actualVerticesRowCount,
		int actualVerticesColCount,int* actualVertices,int* newVertices,int* independentSets,int col, int el )
	{
		int n = N;

		int i=threadIdx.x;
		int l = 0;

			int lastIndex = 0;
			// Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
			for (int index = 0; index < actualVerticesColCount; ++index)
				lastIndex += (1 << actualVertices[i*actualVerticesColCount + index]);
			for (int j = actualVertices[i*actualVerticesColCount + actualVerticesColCount - 1] + 1; j < n; ++j)
			{
				int lastIndex2 = lastIndex;
				// Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
				for (int ns = Offset[j - 1]; ns < Offset[j]; ++ns)
				{
					for (int q = 0; q < actualVerticesColCount; ++q)
					{
						if (actualVertices[i * actualVerticesColCount + q] == Vertices[ns])
						{
							lastIndex2 -= (1 << Vertices[ns]);
							break;
						}
					}		
				}
				int nextIndex = lastIndex + (1 << j);
				// Liczba zbiorów niezale¿nych w aktualnie przetwarzanym podzbiorze
				independentSets[nextIndex] = independentSets[lastIndex] + independentSets[lastIndex2] + 1;
				for (int k = 0; k < el; ++k)
					newVertices[l*col + k] = actualVertices[i * actualVerticesColCount + k];
				newVertices[l * col + el] = j;
				
				l++;
			}
	
	}

#pragma endregion Algorithm

int main()
{
	Graph graph = Graph();
	graph= Graph::ReadGraph("test.txt");

	int roz=0;
	cout<<graph.GetVerticesLength()<<endl;

	roz=1<<graph.GetVerticesCount();

	int* independentSet = BuildingIndependentSets(graph.GetVerticesCount(),graph.GetVertices(),graph.GetNeighborsCount());
	/*cout<<endl;
	for (int i = 0; i <roz; i++)
	{
		cout<<independentSet2[i]<<" ";
	}
	cout<<endl;*/
	//int* independentSet=BuildingIndependentSetsGPU(graph.GetVerticesCount(),graph.GetVertices(),graph.GetNeighborsCount(),graph.GetVerticesLength());
	//cout<<endl;
	//for (int i = 0; i <roz; i++)
	//{
	//	cout<<independentSet[i]<<" ";
	//}
	//cout<<endl;
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

		//for(int i=0;i<graph.GetVerticesCount();i++)
		//	cout<<" "<<tabWyn[i];

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

cudaError_t runCuda(int *wynik, int *independentSet,   int sizeWynik, int sizeIndep)
{
    int *dev_independentSet = 0;
    int *dev_wynik = 0;
    cudaError_t cudaStatus=cudaSuccess;

    // Choose which GPU to run on, change this on a multi-GPU system.
    gpuErrchk(cudaSetDevice(0));
    
    // Allocate GPU buffers for three vectors (two input, one output)    .
    gpuErrchk(cudaMalloc((void**)&dev_wynik, sizeWynik * sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&dev_independentSet, sizeIndep * sizeof(int)));
    
    // Copy input vectors from host memory to GPU buffers.
    gpuErrchk(cudaMemcpy(dev_independentSet, independentSet, sizeIndep * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch a kernel on the GPU with one thread for each element.
	FindChromaticNumber<<<1,sizeWynik>>>(sizeWynik,dev_independentSet,dev_wynik);

    // Check for any errors launching the kernel
    gpuErrchk(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    gpuErrchk(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    gpuErrchk(cudaMemcpy(wynik, dev_wynik, sizeWynik * sizeof(int), cudaMemcpyDeviceToHost));

    return cudaStatus;
}

cudaError_t initIndepSet(int N ,int* Vertices,int verticeslength, int* Offset ,int actualVerticesRowCount,
		int actualVerticesColCount,int* actualVertices,int* newVertices,int* independentSets, int row, int col,int el)
{
	int *dev_Vertices = 0;
    int *dev_Offset = 0;
	int *dev_independentSets=0;
	int *dev_actualVertices=0;
	int *dev_newVertices=0;	
    cudaError_t cudaStatus=cudaSuccess;
	int roz= 1<<N;
	//cout<<roz<<endl;
	gpuErrchk(cudaSetDevice(0));

	cout<<"length"<<verticeslength<<endl;
	gpuErrchk(cudaMalloc((void**)&dev_Vertices, verticeslength * sizeof(int)));
   
	gpuErrchk(cudaMalloc((void**)&dev_Offset, N * sizeof(int)));
    
    gpuErrchk(cudaMalloc((void**)&dev_independentSets, roz * sizeof(int)));
 
	gpuErrchk(cudaMalloc((void**)&dev_actualVertices, (actualVerticesColCount*actualVerticesRowCount) * sizeof(int)));
 
	gpuErrchk(cudaMalloc((void**)&dev_newVertices, (row*col) * sizeof(int)));

	gpuErrchk(cudaMemcpy(dev_independentSets, independentSets, roz * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(dev_Vertices, Vertices, verticeslength * sizeof(int), cudaMemcpyHostToDevice));
 
	gpuErrchk(cudaMemcpy(dev_Offset, Offset, N * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(dev_actualVertices, actualVertices, actualVerticesColCount*actualVerticesRowCount * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(dev_newVertices, newVertices, row*col * sizeof(int), cudaMemcpyHostToDevice));
  
	
	IndependentSetGPU<<<1,actualVerticesRowCount>>>( N , dev_Vertices, dev_Offset , actualVerticesRowCount,
		 actualVerticesColCount,dev_actualVertices, dev_newVertices,dev_independentSets, col, el);
	
	//cudaThreadSynchronize();

	  gpuErrchk(cudaGetLastError());
      
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    gpuErrchk(cudaDeviceSynchronize());
   
    // Copy output vector from GPU buffer to host memory.
    gpuErrchk(cudaMemcpy(independentSets, dev_independentSets, roz * sizeof(int), cudaMemcpyDeviceToHost));
    	
	gpuErrchk(cudaMemcpy(actualVertices, dev_actualVertices, actualVerticesColCount*actualVerticesRowCount * sizeof(int), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaMemcpy(newVertices, dev_newVertices, row*col * sizeof(int), cudaMemcpyDeviceToHost));
   
	/*cudaFree(dev_actualVertices);
	cudaFree(dev_independentSets);
    cudaFree(dev_newVertices);
	cudaFree(dev_Offset);
	cudaFree(dev_Vertices);*/

    return cudaStatus;
}

int* BuildingIndependentSetsGPU(int N ,int* Vertices,int* Offest, int verticesLength)
{
		int n = N;
		int* vertices = Vertices;
		int* offset = Offest;

		int* independentSets;
		int* actualVertices;//zmiana na tablice jedno wymiarow¹ 
		int actualVerticesRowCount;
		int actualVerticesColCount;

		// Inicjalizacja macierzy o rozmiarze 2^N (wartoœci pocz¹tkowe 0)
		independentSets = new int[1 << n] ();

		// Krok 1 algorytmu: przypisanie wartoœci 1 (iloœæ niezale¿nych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
		//CreateActualVertices(n, 1);
		actualVertices = new int[n];

		actualVerticesRowCount = n;//oldRow	
		actualVerticesColCount = 1;//oldCol
		
		for (int i = 0; i < n; ++i)
		{
			independentSets[1 << i] = 1;
			actualVertices[i] = i;
		}

		// G³ówna funkcja tworz¹ca tablicê licznoœci zbiorów niezale¿nych dla wszystkich podzbiorów zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy¿ej.
		for (int el = 1; el < n; el++)
		{	
			cout<<"row "<<actualVerticesRowCount<<endl;
			int col = el + 1;
			int row = Combination_n_of_k(n, col);
			int* newVertices = new int[row*col];//zmiana na tablice jedno wymiarow¹ 
		
			int l = 0;
			int roz=1<<N;
			
			initIndepSet(N,Vertices,verticesLength,Offest,actualVerticesRowCount,actualVerticesColCount,
				actualVertices,newVertices,independentSets,row,col,el);
		
			delete[] actualVertices;

			actualVertices = newVertices;
		
			actualVerticesRowCount = row;
			actualVerticesColCount = col;
			cout<<"nr "<<el<<endl;
    
		}
		return independentSets;
}