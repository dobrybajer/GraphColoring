#include <iostream>
#include <Windows.h>

#include "Algorithm.h" 
#include "Graph.h" 
using namespace version_cpu;
using namespace std;

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

int main()
{
	// https://sites.google.com/site/graphcoloring/vertex-coloring - przyk³ady grafów z ich liczb¹ chromatyczn¹
	// http://mat.gsia.cmu.edu/COLOR/instances.html - grafy w postaci krawêdziowej
	// dla Myciel3 poprawny wynik to 4, a dla Myciel4 wynik to 5

	//string path = "../TestFiles/GraphExampleMyciel4.txt";
	string path = "../TestFiles/GraphExample20.txt";
	Graph g = Graph::ReadGraph(path);


	double wall0 = get_wall_time();
	double cpu0 = get_cpu_time();

	{
		int wynik = FindChromaticNumber(g.GetVertices(), g.GetNeighborsCount(), g.GetVerticesCount());
		cout << "wynik to: " << wynik << endl << endl;
	}

	double wall1 = get_wall_time();
	double cpu1 = get_cpu_time();

	cout << "Wall Time = " << wall1 - wall0 << " seconds" << endl;
	cout << "CPU Time  = " << cpu1 - cpu0 << " seconds" << endl;

	cin.get();

	return 0;
}