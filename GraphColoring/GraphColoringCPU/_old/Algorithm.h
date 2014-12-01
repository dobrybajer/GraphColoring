#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "Graph.h"
#include <vector>

using namespace std;

namespace version_cpu
{
	class ChromaticNumber
	{
	private:
		vector< vector<int> > actualVertices;
		int verticesCount;
		int** independentSets;

		__declspec(dllexport) long Pow(int, int);
		__declspec(dllexport) int sgnPow(int);
		__declspec(dllexport) void BuildingIndependentSets(Graph);

	public:
		__declspec(dllexport) ChromaticNumber();

		__declspec(dllexport) int FindChromaticNumber(Graph);
	};
}
#endif