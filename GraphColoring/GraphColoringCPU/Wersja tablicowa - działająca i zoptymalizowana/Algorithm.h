#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "Graph.h"

using namespace std;

namespace version_cpu
{
	class ChromaticNumber
	{
	private:
		int* independentSets;
		int** actualVertices;
		int actualVerticesRowCount;
		int actualVerticesColCount;

		__declspec(dllexport) void BuildingIndependentSets(Graph);
		__declspec(dllexport) void CreateActualVertices(int,int);
		__declspec(dllexport) void UpdateActualVertices(int**,int,int);

	public:
		__declspec(dllexport) ChromaticNumber();
		__declspec(dllexport) int FindChromaticNumber(Graph);
	};
}
#endif