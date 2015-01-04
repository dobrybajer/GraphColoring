#ifdef UNITTEST
#define DLLIMPEXP __declspec(dllexport)
#else
#define DLLIMPEXP __declspec(dllimport)
#endif

#ifndef ALGORITHM_H
#define ALGORITHM_H

/// <summary>
///	Przestrzeñ nazwy dla algorytmu kolorowania grafu w wersji CPU napisanej w jêzyku C++.
/// </summary>
namespace version_cpu
{
	DLLIMPEXP unsigned int Pow(int, int);
	DLLIMPEXP int sgnPow(int);
	DLLIMPEXP int BitCount(int);
	DLLIMPEXP unsigned int Combination_n_of_k(int, int);
	DLLIMPEXP int GetFirstBitPosition(int);
	DLLIMPEXP unsigned int GetBitPosition(int, int);
	DLLIMPEXP int** CreateVertices(int, int);
	DLLIMPEXP int* BuildingIndependentSets_BitVersion(int*, int);
	DLLIMPEXP int* BuildingIndependentSets_TableVersion(int*, int*, int);

	extern "C" DLLIMPEXP int FindChromaticNumber(int*, int*, int, int=0);
}

#endif
