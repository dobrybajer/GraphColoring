#ifndef ALGORITHM_H
#define ALGORITHM_H

namespace version_cpu
{
	extern "C"
	{
		__declspec(dllexport) unsigned long Pow(int, int);
		__declspec(dllexport) int sgnPow(int);
		__declspec(dllexport) int BitCount(int);
		__declspec(dllexport) int Combination_n_of_k(int, int);
		__declspec(dllexport) int** CreateVertices(int, int);
		__declspec(dllexport) int* BuildingIndependentSets(int*, int*, int);
		__declspec(dllexport) int FindChromaticNumber(int*, int*, int);
	};
}

#endif
