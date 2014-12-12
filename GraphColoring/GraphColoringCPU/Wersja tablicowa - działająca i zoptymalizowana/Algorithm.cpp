#include "Algorithm.h"
#include "Graph.h"

namespace version_cpu
{
	ChromaticNumber::ChromaticNumber()
	{

	}
	
	// Sprawdziæ czy mo¿na lepiej
	unsigned long Pow(int a, int n)
	{
		unsigned long result = 1;

		while (n)
		{
			if (n & 1)
				result *= a;
			
			n >>= 1;
			a *= a;
		}

		return result;
	}

	// Final
	int sgnPow(int n)
	{
		return (n & 1) == 0 ? 1 : -1;
	}

	// Sprawdziæ, dlaczego to dzia³a
	int BitCount(int u)
	{
		int uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
		return ((uCount + (uCount >> 3)) & 030707070707) % 63;
	}

	// Sprawdziæ, czy mo¿na lepiej
	int Combination_n_of_k(int n, int k)
	{
		if (k > n) return 0;

		int r = 1;
		for (int d = 1; d <= k; ++d)
		{
			r *= n--;
			r /= d;
		}
		return r;
	} 

	int** CreateNewVertices(int row, int col)
	{
		int** nVertices = new int*[row];
		for (int i = 0; i < row; ++i)
			nVertices[i] = new int[col]();

		return nVertices;
	}

	void ChromaticNumber::CreateActualVertices(int row, int col)
	{
		actualVertices = new int*[row];

		for (int i = 0; i < row; ++i)
			actualVertices[i] = new int[col] ();

		actualVerticesRowCount = row;
		actualVerticesColCount = col;
	}

	void ChromaticNumber::UpdateActualVertices(int** newVertices, int row, int col)
	{
		for (int i = 0; i < actualVerticesRowCount; ++i)
		{
			delete[] actualVertices[i];
		}
		delete[] actualVertices;

		actualVertices = newVertices;

		actualVerticesRowCount = row;
		actualVerticesColCount = col;
	}

	void ChromaticNumber::BuildingIndependentSets(Graph g)
	{
		int n = g.GetVerticesCount();
		int* vertices = g.GetVertices();
		int* offset = g.GetNeighborsCount();

		// Inicjalizacja macierzy o rozmiarze 2^N (wartoœci pocz¹tkowe 0)
		independentSets = new int[1 << n] ();

		// Krok 1 algorytmu: przypisanie wartoœci 1 (iloœæ niezale¿nych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
		CreateActualVertices(n, 1);
		
		for (int i = 0; i < n; ++i)
		{
			independentSets[1 << i] = 1;
			actualVertices[i][0] = i;
		}

		// G³ówna funkcja tworz¹ca tablicê licznoœci zbiorów niezale¿nych dla wszystkich podzbiorów zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy¿ej.
		for (int el = 1; el < n; el++)
		{
			int col = el + 1;
			int row = Combination_n_of_k(n, col);
			
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
					independentSets[nextIndex] = independentSets[lastIndex] + independentSets[lastIndex2] + 1;

					for (int k = 0; k < el; ++k)
						newVertices[l][k] = actualVertices[i][k];

					newVertices[l][el] = j;

					l++;
				}
			}
			UpdateActualVertices(newVertices, row, col);
		}
	}

	int ChromaticNumber::FindChromaticNumber(Graph g)
	{
		BuildingIndependentSets(g);
		int n = g.GetVerticesCount();
	
		for (int k = 1; k <= n; ++k)
		{
			unsigned long s = 0;
			int PowerNumber = Pow(2, n);
			// Czy mo¿na omin¹æ u¿ycie funkcji BitCount ?
			for (int i = 0; i < PowerNumber; ++i) s += (sgnPow(BitCount(i)) * Pow(independentSets[i], k));
			
			if (s <= 0) continue;
		
			return k;
		}
		
		return -1;
	}
}