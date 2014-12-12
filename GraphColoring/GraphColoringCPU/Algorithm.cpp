#include "Algorithm.h"

namespace version_cpu
{
	// Sprawdzi� czy mo�na lepiej
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

	// Sprawdzi�, dlaczego to dzia�a
	int BitCount(int u)
	{
		int uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
		return ((uCount + (uCount >> 3)) & 030707070707) % 63;
	}

	// Sprawdzi�, czy mo�na lepiej
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

	int** CreateVertices(int row, int col)
	{
		int** nVertices = new int*[row];
		for (int i = 0; i < row; ++i)
			nVertices[i] = new int[col]();

		return nVertices;
	}

	int* BuildingIndependentSets(int* vertices, int* offset, int n)
	{
		int* independentSets;
		int** actualVertices;
		int** newVertices;

		int actualVerticesRowCount;
		int actualVerticesColCount;

		// Inicjalizacja macierzy o rozmiarze 2^N (warto�ci pocz�tkowe 0)
		independentSets = new int[1 << n] ();

		// Krok 1 algorytmu: przypisanie warto�ci 1 (ilo�� niezale�nych zbior�w) dla podzbior�w 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych element�w (1 poziom tworzenia wszystkich podzbior�w)
		actualVertices = CreateVertices(n, 1);

		actualVerticesRowCount = n;
		actualVerticesColCount = 1;
		
		for (int i = 0; i < n; ++i)
		{
			independentSets[1 << i] = 1;
			actualVertices[i][0] = i;
		}

		// G��wna funkcja tworz�ca tablic� liczno�ci zbior�w niezale�nych dla wszystkich podzbior�w zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy�ej.
		for (int el = 1; el < n; el++)
		{
			int col = el + 1;
			int row = Combination_n_of_k(n, col);
			
			newVertices = CreateVertices(row, col);
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

					// Liczba zbior�w niezale�nych w aktualnie przetwarzanym podzbiorze
					independentSets[nextIndex] = independentSets[lastIndex] + independentSets[lastIndex2] + 1;

					for (int k = 0; k < el; ++k)
						newVertices[l][k] = actualVertices[i][k];

					newVertices[l][el] = j;

					l++;
				}
			}
	
			for (int i = 0; i < actualVerticesRowCount; ++i)
			{
				delete[] actualVertices[i];
			}
			delete[] actualVertices;

			actualVertices = newVertices;

			actualVerticesRowCount = row;
			actualVerticesColCount = col;
		}

		for (int i = 0; i < actualVerticesRowCount; ++i)
		{
			delete[] actualVertices[i];
		}
		delete[] actualVertices;

		return independentSets;
	}

	int FindChromaticNumber(int* vertices, int* offset, int n)
	{
		int* independentSets = BuildingIndependentSets(vertices, offset, n);

		for (int k = 1; k <= n; ++k)
		{
			unsigned long s = 0;
			int PowerNumber = Pow(2, n);
			// Czy mo�na omin�� u�ycie funkcji BitCount ?
			for (int i = 0; i < PowerNumber; ++i) s += (sgnPow(BitCount(i)) * Pow(independentSets[i], k));
			
			if (s <= 0) continue;

			delete[] independentSets;

			return k;
		}

		delete[] independentSets;
		return -1;
	}
}