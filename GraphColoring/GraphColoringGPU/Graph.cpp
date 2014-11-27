#include "Graph.h"
#include <iostream>
#include <fstream>
#include <conio.h>
#include <string>
#include <vector>
#include <sstream>
using namespace std;

Graph::Graph(){}

Graph::Graph(int vertexCount )
{
	vertex = new int * [vertexCount];
	_vertexCount = vertexCount;
}

Graph::Graph(int ** _vertex)
{
	vertex =_vertex;
}

Graph Graph::ReadGraph(string path)
{
	Graph graph;
	int ** nGraph;
	int * count;
    string line;
	fstream plik;
	int size=0;
	int i=0;
		plik.open( path, ios::in | ios::out );
		if( plik.good() )
		{				
			getline( plik, line );		
			nGraph= new int*[stoi(line)];
			count = new int[stoi(line)];
			size=stoi(line);
			while( !plik.eof() )
			{
				getline( plik, line );
				int *tmp;
				vector <string> el;
				stringstream ss(line);
				string item;
				int val;
				while (getline(ss, item, ',')) 
					el.push_back(item);
				nGraph[i] = new int[el.size()];
				count[i]= el.size();
				for(int j=0;j<el.size();j++)
					nGraph[i][j]=stoi(el[j]);
	//			cout << line << endl;
				i++;
			}
			plik.close();
		} else cout << "Error! Nie udalo otworzyc sie pliku!" << endl;		
		graph=Graph(nGraph);
		graph._vertexCount=size;
		graph.neighbourCount=count;
		return graph;
}
	
static void WriteGraph(string path, Graph graph)
{

}

int Graph::getVertexCount()
{
	return _vertexCount;
}

////int * Graph::creatNeighbours(string neighbours,int& neigh )
//{
//	int *tmp;
//	vector <string> el;
//    stringstream ss(neighbours);
//    string item;
//	int val;
//    while (getline(ss, item, ',')) 
//        el.push_back(item);
//
//	tmp = new int[el.size()+1];
//
//	for(int i=0;i<el.size();i++)
//		tmp[i]=stoi(el[i]);
//	
//	return tmp;	
//}


Graph::~Graph(void)
{

}
