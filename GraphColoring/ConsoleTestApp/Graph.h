#ifndef GRAPH_H
#define GRAPH_H

#include <string>
namespace version_cpu
{
	class __declspec(dllexport) Graph
	{
	private:
		int* vertices;
		int* neighborsCount;
		int verticesCount;

	public:

		Graph();
		Graph(int*, int*, int);

		int* __cdecl GetVertices();
		int* __cdecl GetNeighborsCount();
		int __cdecl GetVerticesCount();

		static Graph ReadGraph(std::string path);

	};
}

#endif