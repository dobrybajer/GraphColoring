#include "Algorithm.h"
#include "Graph.h"
#include <vector>
#include <iostream>

using namespace std;

namespace version_cpu
{
	ChromaticNumber::ChromaticNumber()
	{

	}
	
	unsigned long long ChromaticNumber::Pow(int a, int n)
	{
		unsigned long long result = 1;

		while (n)
		{
			if (n & 1)
				result *= a;
			
			n >>= 1;
			a *= a;
		}

		return result;
	}

	int ChromaticNumber::BitCount(int u)
	{
		int uCount;

		uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
		return ((uCount + (uCount >> 3)) & 030707070707) % 63;
	}

	int ChromaticNumber::sgnPow(int n)
	{
		return (n & 1) == 0 ? 1 : -1;
	}

	int ChromaticNumber::MaxCombinationCount(int n, int k)
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

	int ChromaticNumber::GetFirstBitPosition(int n)
	{
		int cnt = BitCount(n);
		int c = 0, i = 0;
		while (c != cnt)
		{
			if (n & (1 << i))
				c++;
			i++;
		}
		return i-1;
	}

	unsigned long long GetBitPosition(int n, int col)
	{
		unsigned long long f = 0, c = 0, i = 0;
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

	void ChromaticNumber::BuildingIndependentSets(Graph g)
	{
		int n = g.GetVerticesCount();
		int* vertices = g.GetVertices();
		int cnt = (int)1 << n;
		int actualRow = n;
		// Inicjalizacja macierzy o rozmiarze 2^N (wartoœci pocz¹tkowe 0)
		independentSets = new int[cnt] ();

		// Krok 1 algorytmu: przypisanie wartoœci 1 (iloœæ niezale¿nych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
		actualVertices = new int[n] ();
		
		for (int i = 0; i < n; ++i)
		{
			independentSets[(1 << i)] = 1;
			actualVertices[i] |= (1 << i);
		}

		// G³ówna funkcja tworz¹ca tablicê licznoœci zbiorów niezale¿nych dla wszystkich podzbiorów zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy¿ej.
		for (int el = 1; el < n; el++)
		{
			int row = MaxCombinationCount(n, el + 1);
	
			int* newVertices = new int[row] ();
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
						if ((vertices[j] & (1 << no)) )
						{
							if (actualVertices[i] & (1<<no) && j!=no)
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
			actualVertices = new int[row] ();
			for (int i = 0; i < row;++i)
				actualVertices[i] = newVertices[i];
			delete[] newVertices;
		}
	}

	int ChromaticNumber::FindChromaticNumber(Graph g)
	{
		BuildingIndependentSets(g);
		int n = g.GetVerticesCount();

		for (int k = 1; k <= n; ++k)
		{
			unsigned long long s = 0;
			
			for (int i = 0; i < Pow(2, n); ++i)	s += sgnPow(n - BitCount(i)) * Pow(independentSets[i], k);
			
			if (s <= 0) continue;

			return k;
		}
		
		return -1;
	}
}