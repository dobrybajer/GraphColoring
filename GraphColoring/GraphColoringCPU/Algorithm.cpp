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

	void ChromaticNumber::CreateActualVertices(int row, int col)
	{
		if (actualVertices != 0)
		{
			for (int i = 0; i < actualVerticesRowCount; ++i)
			{
				delete[] actualVertices[i];
			}
			delete[] actualVertices;
		}

		actualVertices = new int*[row];
		for (int i = 0; i < row; ++i)
			actualVertices[i] = new int[col] {0};

		actualVerticesRowCount = row;
		actualVerticesColCount = col;
	}

	int** ChromaticNumber::CreateNewVertices(int row, int col)
	{
		int** nVertices = new int*[row];
		for (int i = 0; i < row; ++i)
			nVertices[i] = new int[col] {0};

		return nVertices;
	}

	int choose(int n, int k) 
	{
		if (k > n) {
			return 0;
		}
		int r = 1;
		for (int d = 1; d <= k; ++d) {
			r *= n--;
			r /= d;
		}
		return r;
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
		CreateActualVertices(n, 1);
		
		for (int i = 0; i < n; ++i)
		{
			independentSets[1 << i][0] = 1;
			independentSets[1 << i][1] = 1;

			actualVertices[i][0] = i;
		}

		// G³ówna funkcja tworz¹ca tablicê licznoœci zbiorów niezale¿nych dla wszystkich podzbiorów zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy¿ej.
		for (int el = 1; el < n; el++)
		{
			int row = choose(n, el + 1);
			int col = el + 1;
			int** newVertices = CreateNewVertices(row, col);
			int l = 0;

			for (int i = 0; i < actualVerticesRowCount; ++i)
			{
				int lastIndex = 0;

				// Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
				for (int index = 0; index < actualVerticesColCount; ++index)
					lastIndex += (1 << actualVertices[i][index]);

				for (int j = actualVertices[i][actualVerticesColCount - 1] + 1; j < n; ++j)
				{
					int lastIndex2 = lastIndex;

					// Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
					for (int ns = offset[j - 1]; ns < offset[j]; ++ns)
					{
						for (int q = 0; q < actualVerticesColCount; ++q)
						{
							if (actualVertices[i][q] == vertices[ns])
							{
								lastIndex2 -= (1 << vertices[ns]);
								break;
							}
						}		
					}

					int nextIndex = lastIndex + (1 << j);

					// Liczba zbiorów niezale¿nych w aktualnie przetwarzanym podzbiorze
					independentSets[nextIndex][0] = independentSets[lastIndex][0] + independentSets[lastIndex2][0] + 1;
					independentSets[nextIndex][1] = el + 1;

					for (int k = 0; k < el; ++k)
						newVertices[l][k] = actualVertices[i][k];

					newVertices[l][el] = j;

					l++;
				}
			}
			
			actualVertices = newVertices;
			actualVerticesColCount = col;
			actualVerticesRowCount = row;
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