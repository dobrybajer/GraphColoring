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
		int verticesLength;

	public:
		 Graph();
		 Graph(int*, int*, int, int);

		 int* GetVertices();
		 int* GetNeighborsCount();
		 int GetVerticesCount();
		 int GetVerticesLength();

		 static Graph ReadGraph(std::string path);
	};
}
#endif