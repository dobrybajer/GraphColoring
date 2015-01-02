#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Algorithm.cuh"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <Windows.h>

using namespace std;
using namespace version_gpu;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

#pragma region Time Measuring

double get_wall_time()
{
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)) { return 0; }
	if (!QueryPerformanceCounter(&time)) { return 0; }
	return (double)time.QuadPart / freq.QuadPart;
}

double get_cpu_time()
{
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0)
		return (double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	else
		return 0;
}

#pragma endregion Time Measuring

int main()
{
	Graph graph = ReadGraph("../../TestFiles/GraphExample10.txt");

	int* tabWyn = new int[graph.n];

	double wall0 = get_wall_time();
	double cpu0 = get_cpu_time();

	cudaError_t cudaStatus = FindChromaticNumberMain(tabWyn, graph.vertices, graph.neighbors, graph.n, graph.allVerticesCount);
	if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "FindChromaticNumberMain failed!");
        return 1;
    }

	int wynik = -2;

	for(int i = 0; i < graph.n; i++)
	{
		if(tabWyn[i]!=-1 && tabWyn[i]!=0)
		{
			wynik = tabWyn[i] + 1;
			break;
		}
	}

	double wall1 = get_wall_time();
	double cpu1 = get_cpu_time();

	cout << "Wall Time = " << wall1 - wall0 << " seconds" << endl;
	cout << "CPU Time  = " << cpu1 - cpu0 << " seconds" << endl;

	cout << "Potrzeba " << wynik << " kolorow." << endl;
	
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
