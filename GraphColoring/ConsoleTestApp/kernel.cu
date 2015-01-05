#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "..\GraphColoringCPU\Algorithm.h"
#include "..\GraphColoringGPU\Algorithm.cuh"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <Windows.h>

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

Graph ReadGraphBitVersion(string path)
{
	fstream plik;
	plik.open(path, ios::in | ios::out);

	if (plik.good())
	{
		string line;
		getline(plik, line);

		int size = stoi(line);
		int i = 0;

		int* nVertices = new int[size]();

		while (!plik.eof())
		{
			getline(plik, line);

			stringstream ss(line);
			string item;

			while (getline(ss, item, ','))
				nVertices[i] |= (1 << stoi(item));

			i++;
		}
		plik.close();

		Graph g = { nVertices, NULL, size };

		return g;
	}
	else throw new logic_error("Podczas otwierania pliku wyst¹pi³ b³¹d");

	return Graph();
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

#pragma region Versions

int GPU(Graph g)
{
	int* tabWyn = new int[g.n];

	cudaError_t cudaStatus = version_gpu::FindChromaticNumberGPU(tabWyn, g.vertices, g.neighbors, g.n, g.allVerticesCount);
	if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "FindChromaticNumberMain failed!");
        return -1;
    }

	int wynik = -2;

	for(int i = 0; i < g.n; i++)
	{
		if(tabWyn[i]!=-1 && tabWyn[i]!=0)
		{
			wynik = tabWyn[i] + 1;
			break;
		}
	}

	return wynik;
}

int CPU(Graph g, int flag)
{
	if(flag == 1)
		return  version_cpu::FindChromaticNumber(g.vertices, g.neighbors, g.n);
	else
		return  version_cpu::FindChromaticNumber(g.vertices, g.neighbors, g.n, 1);
}

#pragma endregion Versions

int main()
{
	// https://sites.google.com/site/graphcoloring/vertex-coloring - przyk³ady grafów z ich liczb¹ chromatyczn¹
	// http://mat.gsia.cmu.edu/COLOR/instances.html - grafy w postaci krawêdziowej
	// dla Myciel3 poprawny wynik to 4, a dla Myciel4 wynik to 5

	int deviceReset = 0;

	//string path = "../../TestFiles/GraphExampleMyciel4.txt";
	string path = "../../TestFiles/GraphExample3_2.txt";

	int wynik;
	int what = -1;
	Graph graph;

	while(what != 3)
	{
		cout << "Podaj wersje aplikacji:" << endl;
		cout << "0 - wersja GPU" << endl;
		cout << "1 - tablicowa wersja CPU" << endl;
		cout << "2 - bitowa wersja CPU" << endl;
		cout << "3 - wyjscie z aplikacji" << endl;
		cout << endl;

		cin >> what;
		cout << endl;
		
		if(what == 3)
			return 1;

		try
		{
			if(what == 2)
				graph = ReadGraphBitVersion(path);
			else
				graph = ReadGraph(path);
		}
		catch (logic_error le)
		{
			cout << "Fatal error while reading graph from file: " << le.what() << endl;
			cin.get();
			return 1;
		}

		double wall0 = get_wall_time();
		double cpu0 = get_cpu_time();

		if(what == 0)
		{
			wynik = GPU(graph);
			deviceReset = 1;
		}
		else if(what == 1)
			wynik = CPU(graph, 1);
		else
			wynik = CPU(graph, 2);

		double wall1 = get_wall_time();
		double cpu1 = get_cpu_time();

		cout << "Wall Time = " << wall1 - wall0 << " seconds" << endl;
		cout << "CPU Time  = " << cpu1 - cpu0 << " seconds" << endl;
		cout << endl;

		cout << "Potrzeba " << wynik << " kolorow." << endl;
		cout << endl;
	}

	if(deviceReset == 1)
	{
		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) 
		{
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}
	
    return 0;
}
