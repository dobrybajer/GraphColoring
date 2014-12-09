#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "Graph.h"

using namespace std;

namespace version_cpu
{
	template <typename T>
	class ChromaticNumber
	{
	private:
		T* actualVertices;
		int verticesCount;
		T* independentSets;

		__declspec(dllexport) unsigned long long Pow(int, int);
		__declspec(dllexport) T BitCount(T);
		__declspec(dllexport) int sgnPow(int);
		__declspec(dllexport) T MaxCombinationCount(int, int);
		__declspec(dllexport) void BuildingIndependentSets(Graph);	

	public:
		__declspec(dllexport) ChromaticNumber<T>();
		__declspec(dllexport) int FindChromaticNumber(Graph);
	};
}
#endif