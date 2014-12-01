#ifndef GRAPH_H
#define GRAPH_H

#include <string>

namespace version_cpu
{
	class Graph
	{
	private:
		int* vertices;
		int* neighborsCount;
		int verticesCount;

	public:
		__declspec(dllexport) Graph();
		__declspec(dllexport) Graph(int*, int*, int);

		__declspec(dllexport) int* GetVertices();
		__declspec(dllexport) int* GetNeighborsCount();
		__declspec(dllexport) int GetVerticesCount();

		__declspec(dllexport) static Graph ReadGraph(std::string path);
	};
}
#endif