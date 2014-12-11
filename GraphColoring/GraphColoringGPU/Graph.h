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
		 Graph();
		 Graph(int*, int*, int);

		 int* GetVertices();
		 int* GetNeighborsCount();
		 int GetVerticesCount();

		 static Graph ReadGraph(std::string path);
	};
}
#endif