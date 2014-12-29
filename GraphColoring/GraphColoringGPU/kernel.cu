#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

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

#pragma region Headers

cudaError_t runCuda(int*, int*, int, int);
cudaError_t runCuda2(int*, int*, int*, int, int);
cudaError_t initIndepSet(int, int*, int, int*, int, int, int*, int*, int*, int, int, int);
int* BuildingIndependentSetsGPU(int N, int* Vertices, int* Offest, int verticesLength);

#pragma endregion Headers

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

	int* BuildingIndependentSets(int N, int* Vertices, int* Offset)
	{
		int n = N;
		int* vertices = Vertices;
		int* offset = Offset;

		int* independentSets;
		int* actualVertices;
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

	int* BuildingIndependentSetsGPU(int N, int* Vertices, int* Offest, int verticesLength)
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

	__global__ void FindChromaticNumber(int N, int* independentSets, int* wynik)
	{
		int n = N;
		int index = threadIdx.x;

		unsigned long s = 0;
		int PowerNumber = 1 << n;

		for (int i = 0; i < PowerNumber; ++i) s += (sgnPow(BitCount(i)) * Pow(independentSets[i], index + 1));
			
		wynik[index] = s > 0 ? index : s; // KAMIL: punkt krytyczny, czy dobrze jest liczone "s"? dla unsigned long long liczy Ÿle...
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

#pragma region Structure

	struct Graph
{
	int* vertices;
	int* neighbors;
	int n;
	int allVerticesCount;
};

	Graph ReadGraph(string path)
{
	fstream plik;
	plik.open(path, ios::in | ios::out);

	if (plik.good())
	{
		string line;
		getline(plik, line);

		int size = stoi(line);
		int i = 0, k = 0;
		int* nNeighborsCount = new int[size];
		vector<string> el;

		while (!plik.eof())
		{
			getline(plik, line);

			stringstream ss(line);
			string item;

			while (getline(ss, item, ','))
				el.push_back(item);

			nNeighborsCount[i] = el.size();

			k = el.size();
			i++;
		}
		plik.close();

		int* nVertices = new int[k];

		for (int i = 0; i < k; i++)
			nVertices[i] = stoi(el[i]);

		Graph g = { nVertices, nNeighborsCount, size, k };

		return g;
	}
	else throw new logic_error("Podczas otwierania pliku wyst¹pi³ b³¹d");
}

#pragma endregion Structure

int main()
{
	Graph graph = ReadGraph("test.txt");

	//int roz = 1 << graph.n;

	//int* independentSet = BuildingIndependentSets(graph.n, graph.vertices, graph.neighbors);
	//int* independentSet = BuildingIndependentSetsGPU(graph.n, graph.vertices, graph.neighbors, graph.allVerticesCount);
	//cout << endl;
	//for (int i = 0; i <roz; i++)
	//{
	//	cout << independentSet[i] << " ";
	//}
	//cout << endl;
	int* tabWyn = new int[graph.n];

	//cudaError_t cudaStatus = runCuda(tabWyn, independentSet, graph.n, roz);
	cudaError_t cudaStatus = runCuda2(tabWyn, graph.vertices, graph.neighbors, graph.n, graph.allVerticesCount);
   
	if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	int wynik;

	for(int i = 0; i < graph.n; i++)
	{
		if(tabWyn[i]!=-1 && tabWyn[i]!=0)
		{
			wynik = tabWyn[i] + 1;
			break;
		}
	}

	//for(int i=0;i<graph.GetVerticesCount();i++)
	//	cout << " " << tabWyn[i];

	cout << endl << "Potrzeba " << wynik << " kolorow." << endl;
	

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

#pragma region CudaFunctions

cudaError_t runCuda(int *wynik, int *independentSet, int sizeWynik, int sizeIndep)
{
    int *dev_independentSet = 0;
    int *dev_wynik = 0;
    cudaError_t cudaStatus = cudaSuccess;

    // Choose which GPU to run on, change this on a multi-GPU system.
    gpuErrchk(cudaSetDevice(0));
    
    // Allocate GPU buffers for three vectors (two input, one output)    .
    gpuErrchk(cudaMalloc((void**)&dev_wynik, sizeWynik * sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&dev_independentSet, sizeIndep * sizeof(int)));
    
    // Copy input vectors from host memory to GPU buffers.
    gpuErrchk(cudaMemcpy(dev_independentSet, independentSet, sizeIndep * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch a kernel on the GPU with one thread for each element.
	FindChromaticNumber<<<1,sizeWynik>>>(sizeWynik, dev_independentSet, dev_wynik);

    // Check for any errors launching the kernel
    gpuErrchk(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    gpuErrchk(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    gpuErrchk(cudaMemcpy(wynik, dev_wynik, sizeWynik * sizeof(int), cudaMemcpyDeviceToHost));

    return cudaStatus;
}

cudaError_t initIndepSet(int N, int* Vertices, int verticeslength, int* Offset, int actualVerticesRowCount,
		int actualVerticesColCount, int* actualVertices, int* newVertices, int* independentSets, int row, int col, int el)
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
		 actualVerticesColCount,dev_actualVertices, dev_newVertices, dev_independentSets, col, el);
	
	cudaThreadSynchronize();

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

#pragma endregion CudaFunctions

#pragma region CudaFunctions - version 2

__global__ void Init1(int* independentSet, int* actualVertices, int verticesCount)
{
	for (int i = 0; i < verticesCount; ++i)
	{
		independentSet[1 << i] = 1;
		actualVertices[i] = i;
	}
}

__global__ void Init2(int* actualVertices, int* newVertices)
{
	actualVertices = newVertices;
}

__global__ void Init3(int* actualVertices, int* l_set, int n, int actualVerticesRowCount, int actualVerticesColCount)
{
	int last_el = 0;
	l_set[0] = 0;
	for(int i = 1; i < actualVerticesRowCount; ++i)
	{
		int j = n - actualVertices[(i - 1) * actualVerticesColCount + actualVerticesColCount - 1] - 1;
	
		if(j <= 0)
			l_set[i] = -1;
		else
		{
			l_set[i] = last_el + j;
			last_el = j;
		}
	}
}

__global__ void IndependentSetGPU2(int* l_set, int n, int* Vertices, int* Offset, int actualVerticesRowCount, int actualVerticesColCount, int* actualVertices, int* newVertices, int* independentSets, int col, int el)
{
	int i = threadIdx.x;
	int l = l_set[i];

	if (l==-1) return;

	int lastIndex = 0;
	// Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
	for (int index = 0; index < actualVerticesColCount; ++index)
		lastIndex += (1 << actualVertices[i * actualVerticesColCount + index]);

	for (int j = actualVertices[i * actualVerticesColCount + actualVerticesColCount - 1] + 1; j < n; ++j)
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

cudaError_t runCuda2(int* wynik, int* vertices, int* offset, int verticesCount, int allVerticesCount)
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
    
	Init1<<<1,1>>> (dev_independentSet, dev_actualVertices, verticesCount);

	for (int el = 1; el < verticesCount; el++)
	{	
		int col = el + 1;
		int row = Combination_n_of_k(verticesCount, col);

		gpuErrchk(cudaMalloc((void**)&dev_newVertices, (row * col) * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&dev_l_set, actualVerticesRowCount * sizeof(int)));
		
		Init3<<<1,1>>> (dev_actualVertices, dev_l_set, verticesCount, actualVerticesRowCount, actualVerticesColCount);

		int* l_set = new int[actualVerticesRowCount];
		gpuErrchk(cudaMemcpy(l_set, dev_l_set, actualVerticesRowCount * sizeof(int), cudaMemcpyDeviceToHost));
		cout << "l_set"<<endl;
		for(int r = 0; r < actualVerticesRowCount; ++r)
			cout << l_set[r] << " ";
		cout << endl;



		IndependentSetGPU2<<<1,actualVerticesRowCount>>> (dev_l_set, verticesCount, dev_vertices, dev_offset, actualVerticesRowCount, actualVerticesColCount, dev_actualVertices, dev_newVertices, dev_independentSet, col, el);
	
		cudaThreadSynchronize();

		cudaFree(dev_actualVertices);
		Init2<<<1,1>>> (dev_actualVertices, dev_newVertices);
		
		actualVerticesRowCount = row;
		actualVerticesColCount = col;
		cout<<"nr "<<el<<endl;
	}

	FindChromaticNumber<<<1,verticesCount>>>(verticesCount, dev_independentSet, dev_wynik);

    gpuErrchk(cudaGetLastError());
    
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(wynik, dev_wynik, verticesCount * sizeof(int), cudaMemcpyDeviceToHost));

    return cudaStatus;
}

#pragma endregion CudaFunctions - version 2