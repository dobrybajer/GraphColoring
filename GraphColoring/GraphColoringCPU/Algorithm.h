#ifndef ALGORITHM_H
#define ALGORITHM_H

/// <summary>
///	Przestrze� nazwy dla algorytmu kolorowania grafu w wersji CPU napisanej w j�zyku C++.
/// </summary>
namespace version_cpu
{
	__declspec(dllexport) unsigned int Pow(int, int);
	__declspec(dllexport) int sgnPow(int);
	__declspec(dllexport) int BitCount(int);
	__declspec(dllexport) unsigned int Combination_n_of_k(int, int);
	__declspec(dllexport) unsigned int GetFirstBitPosition(int);
	__declspec(dllexport) unsigned int GetBitPosition(int, int);
	__declspec(dllexport) int** CreateVertices(int, int);
	__declspec(dllexport) int* BuildingIndependentSets_BitVersion(int*,int*, int);
	__declspec(dllexport) int* BuildingIndependentSets_TableVersion(int*,int*, int*, int);

	extern "C" __declspec(dllexport) int FindChromaticNumber(int*,int*, int*, int, int=0);
}

#endif
