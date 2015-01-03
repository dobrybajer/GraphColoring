#include "Algorithm.h"

namespace version_cpu
{
	// Sprawdziæ czy mo¿na lepiej
	unsigned int Pow(int a, int n)
	{
		unsigned int result = 1;

		while (n)
		{
			if (n & 1)
				result *= a;
			
			n >>= 1;
			a *= a;
		}

		return result;
	}

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
	unsigned int Combination_n_of_k(int n, int k)
	{
		if (k > n) return 0;
		if (k == 0 || k == n) return 1;

		if (k * 2 > n) k = n - k;

		unsigned int r = 1;
		for (int d = 1; d <= k; ++d) {
			r *= n--;
			r /= d;
		}
		return r;
	} 

	int GetFirstBitPosition(int n)
	{
		int cnt = BitCount(n);
		int c = 0, i = 0;
		while (c != cnt)
		{
			if (n & (1 << i))
				c++;
			i++;
		}
		return i - 1 > 0 ? i - 1 : 0;
	}

	unsigned int GetBitPosition(int n, int col)
	{
		unsigned int f = 0, c = 0, i = 0;
		while (f != col)
		{
			if (n & (1 << i))
			{
				c += (1 << i);
				f++;
			}

			i++;
		}
		return c;
	}

	int** CreateVertices(int row, int col)
	{
		int** nVertices = new int*[row];
		for (int i = 0; i < row; ++i)
			nVertices[i] = new int[col]();

		return nVertices;
	}

	int* BuildingIndependentSets_BitVersion(int* vertices, int n)
	{
		int* independentSets;
		int* actualVertices;
		int* newVertices;

		int actualRow = n;

		// Inicjalizacja macierzy o rozmiarze 2^N (wartoœci pocz¹tkowe 0)
		independentSets = new int[1 << n]();

		// Krok 1 algorytmu: przypisanie wartoœci 1 (iloœæ niezale¿nych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
		actualVertices = new int[n]();

		for (int i = 0; i < n; ++i)
		{
			independentSets[(1 << i)] = 1;
			actualVertices[i] |= (1 << i);
		}

		// G³ówna funkcja tworz¹ca tablicê licznoœci zbiorów niezale¿nych dla wszystkich podzbiorów zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy¿ej.
		for (int el = 1; el < n; el++)
		{
			int row = Combination_n_of_k(n, el + 1);

			newVertices = new int[row]();
			int l = 0;

			for (int i = 0; i < actualRow; ++i)
			{
				// Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
				int lastIndex = GetBitPosition(actualVertices[i], el);

				for (int j = GetFirstBitPosition(actualVertices[i]) + 1; j < n; ++j)
				{
					int lastIndex2 = lastIndex;

					// Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
					for (int ns = 0, no = 0; ns < BitCount(vertices[j]); no++)
					{
						if ((vertices[j] & (1 << no)))
						{
							if (actualVertices[i] & (1 << no) && j != no)
								lastIndex2 -= (1 << no);

							ns++;
						}
					}

					int nextIndex = lastIndex + (1 << j);

					// Liczba zbiorów niezale¿nych w aktualnie przetwarzanym podzbiorze
					independentSets[nextIndex] = independentSets[lastIndex] + independentSets[lastIndex2] + 1;

					newVertices[l] = actualVertices[i];
					newVertices[l] |= (1 << j);

					l++;
				}
			}
			actualRow = row;
			delete[] actualVertices;
			actualVertices = new int[row]();
			for (int i = 0; i < row; ++i)
				actualVertices[i] = newVertices[i];
			delete[] newVertices;
		}
		delete[] actualVertices;

		return independentSets;
	}

	int* BuildingIndependentSets_TableVersion(int* vertices, int* offset, int n)
	{
		int* independentSets;
		int** actualVertices;
		int** newVertices;

		int actualVerticesRowCount;
		int actualVerticesColCount;

		// Inicjalizacja macierzy o rozmiarze 2^N (wartoœci pocz¹tkowe 0)
		independentSets = new int[1 << n] ();

		// Krok 1 algorytmu: przypisanie wartoœci 1 (iloœæ niezale¿nych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
		actualVertices = CreateVertices(n, 1);

		actualVerticesRowCount = n;
		actualVerticesColCount = 1;
		
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

					// Liczba zbiorów niezale¿nych w aktualnie przetwarzanym podzbiorze
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

	int FindChromaticNumber(int* vertices, int* offset, int n, int flag)
	{
		int* independentSets = flag == 1 ? 
			BuildingIndependentSets_BitVersion(vertices, n) : 
			BuildingIndependentSets_TableVersion(vertices, offset, n);

		int PowerNumber = Pow(2, n);

		for (int k = 1; k <= n; ++k)
		{
			unsigned int s = 0;
			
			for (int i = 0; i < PowerNumber; ++i) s += (sgnPow(BitCount(i)) * Pow(independentSets[i], k));
			
			if (s <= 0) continue;

			delete[] independentSets;

			return k;
		}

		delete[] independentSets;
		return -1;
	}
}