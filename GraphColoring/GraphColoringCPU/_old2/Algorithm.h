#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "Graph.h"

using namespace std;

namespace version_cpu
{
	class ChromaticNumber
	{
	private:
		int** actualVertices;
		int actualVerticesRowCount;
		int actualVerticesColCount;

		int verticesCount;
		int** independentSets;

		__declspec(dllexport) unsigned long Pow(int, int);
		__declspec(dllexport) int sgnPow(int);
		__declspec(dllexport) void BuildingIndependentSets(Graph);
		__declspec(dllexport) void CreateActualVertices(int,int);
		__declspec(dllexport) int** CreateNewVertices(int, int);

	public:
		__declspec(dllexport) ChromaticNumber();

		__declspec(dllexport) int FindChromaticNumber(Graph);
	};
}
#endif