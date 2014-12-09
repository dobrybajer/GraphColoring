#include "Graph.h"
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

using namespace std;

namespace version_cpu
{
	Graph::Graph()
	{

	}

	Graph::Graph(int* _vertices, int _size)
	{
		vertices = _vertices;
		verticesCount = _size;
	}

	int* Graph::GetVertices()
	{
		return vertices;
	}

	int Graph::GetVerticesCount()
	{
		return verticesCount;
	}

	Graph Graph::ReadGraph(string path)
	{
		fstream plik;
		plik.open(path, ios::in | ios::out);

		if (plik.good())
		{
			string line;
			getline(plik, line);

			int size = stoi(line);
			int i = 0;
			
			int* nVertices = new int[size] ();

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

			return Graph(nVertices, size);
		}
		else throw new logic_error("Podczas otwierania pliku wyst�pi� b��d");

		return Graph();
	}
}

