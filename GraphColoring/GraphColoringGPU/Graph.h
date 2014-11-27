#include <list>
#include <iostream>
#include <fstream>
#include <conio.h>

using namespace std;

class Graph
{
public:
	int ** vertex;
	int *neighbourCount;
	int _vertexCount;
	Graph();
	Graph(int vertexCount);
	Graph(int ** _vertex);
	static Graph ReadGraph(string path);
	static void WriteGraph(string path, Graph graph);
	int getVertexCount();
	~Graph(void);
	//static int * creatNeighbours(string neighbours,int& neigh);
};
