#include "Algorithm.h"
#include "Graph.h"
#include <vector>

using namespace std;

namespace version_cpu
{
	ChromaticNumber::ChromaticNumber()
	{

	}
	
	long ChromaticNumber::Pow(int a, int n)
	{
		long result = 1;

		while (n)
		{
			if (n & 1)
				result *= a;
			
			n >>= 1;
			a *= a;
		}

		return result;
	}

	int ChromaticNumber::sgnPow(int n)
	{
		return (n & 1) == 0 ? 1 : -1;
	}

	void ChromaticNumber::BuildingIndependentSets(Graph g)
	{
		int n = g.GetVerticesCount();
		int* vertices = g.GetVertices();
		int* offset = g.GetNeighborsCount();

		// Inicjalizacja macierzy o rozmiarze 2^N (wartoœci pocz¹tkowe 0)
		independentSets = new int*[1 << n];
		for (int i = 0; i < (1 << n); ++i)
			independentSets[i] = new int[2] {0};

		// Krok 1 algorytmu: przypisanie wartoœci 1 (iloœæ niezale¿nych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
		for (int i = 0; i < n; ++i)
		{
			independentSets[1 << i][0] = 1;
			independentSets[1 << i][1] = 1;

			actualVertices.push_back(vector<int>());
			actualVertices[i].push_back(i);
		}

		// G³ówna funkcja tworz¹ca tablicê licznoœci zbiorów niezale¿nych dla wszystkich podzbiorów zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy¿ej.
		for (int el = 1; el < n; el++)
		{
			vector< vector<int> > newVertices;

			for (int i = 0; i < (int)actualVertices.size(); ++i)
			{
				int lastIndex = 0;
				vector<int> actualSubset = actualVertices[i];
				int count = actualSubset.size();

				// Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
				for (int index = 0; index < count; ++index)
					lastIndex += (1 << actualSubset[index]);

				for (int j = actualSubset[count - 1] + 1; j < n; ++j)
				{
					int lastIndex2 = lastIndex;

					// Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
					for (int ns = offset[j - 1]; ns < offset[j]; ++ns)
					{
						for (int q = 0; q < count; ++q)
						{
							if (actualSubset[q] == vertices[ns])
							{
								lastIndex2 -= (1 << vertices[ns]);
								break;
							}
						}		
					}

					int nextIndex = lastIndex + (1 << j);

					// Liczba zbiorów niezale¿nych w aktualnie przetwarzanym podzbiorze

					independentSets[nextIndex][0] = independentSets[lastIndex][0] + independentSets[lastIndex2][0] + 1;

					vector<int> list(actualSubset.begin(), actualSubset.end());
					list.push_back(j);

					newVertices.push_back(list);
					independentSets[nextIndex][1] = list.size();
				}
			}
			
			actualVertices = newVertices;
		}
	}

	int ChromaticNumber::FindChromaticNumber(Graph g)
	{
		BuildingIndependentSets(g);
		int n = g.GetVerticesCount();
		
		for (int k = 1; k <= n; ++k)
		{
			long s = 0;
			for (int i = 0; i < Pow(2, n); ++i) s += sgnPow(n - independentSets[i][1]) * Pow(independentSets[i][0], k);

			if (s <= 0) continue;

			return k;
		}
		
		return -1;
	}
}