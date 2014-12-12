#include "Graph.h"
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace version_cpu
{
	Graph::Graph()
	{

	}

	Graph::Graph(int* _vertices, int* _neighborsCount, int _size)
	{
		vertices = _vertices;
		neighborsCount = _neighborsCount;
		verticesCount = _size;
	}

	int* Graph::GetVertices()
	{
		return vertices;
	}

	int* Graph::GetNeighborsCount()
	{
		return neighborsCount;
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

			return Graph(nVertices, nNeighborsCount, size);
		}
		else throw new logic_error("Podczas otwierania pliku wyst¹pi³ b³¹d");

		return Graph();
	}
}
