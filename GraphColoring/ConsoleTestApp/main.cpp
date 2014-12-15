#include <Windows.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include "Algorithm.h"

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

struct Graph
{
	int* vertices;
	int* neighbors;
	int n;
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

		Graph g = { nVertices, nNeighborsCount, size };

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

int BitCountaa(int u)
{
	int uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
	return ((uCount + (uCount >> 3)) & 030707070707) % 63;
}

int main(int argc, char *argv[])
{
	// https://sites.google.com/site/graphcoloring/vertex-coloring - przyk³ady grafów z ich liczb¹ chromatyczn¹
	// http://mat.gsia.cmu.edu/COLOR/instances.html - grafy w postaci krawêdziowej
	// dla Myciel3 poprawny wynik to 4, a dla Myciel4 wynik to 5

	//string path = "../../TestFiles/GraphExampleMyciel4.txt";
	string path = "../../TestFiles/GraphExample21.txt";
	Graph g;
	try
	{
		g = ReadGraph(path);
		//g = ReadGraphBitVersion(path);
	}
	catch (logic_error le)
	{
		cout << "Fatal error while reading graph from file: " << le.what() << endl;
		cin.get();
		return 1;
	}

	cout << "Press any key to start computing ...";
	cin.get();

	double wall0 = get_wall_time();
	double cpu0 = get_cpu_time();

	{
		//int wynik = FindChromaticNumber(g.vertices, g.neighbors, g.n);
		//cout << "wynik to: " << wynik << endl << endl;
		cout << "BitCount: " << BitCountaa(536870912/16) << endl;
		for (int i = Pow(2, 12); i < Pow(2, 31); i *= 2)
		{
			cout << "BitCount " << i <<  ": " << BitCountaa(i) << endl;
		}
	}

	double wall1 = get_wall_time();
	double cpu1 = get_cpu_time();

	cout << "Wall Time = " << wall1 - wall0 << " seconds" << endl;
	cout << "CPU Time  = " << cpu1 - cpu0 << " seconds" << endl;

	cin.get();

	return 0;
}