#include "Algorithm.h"
#include "Graph.h"
#include <vector>

using namespace std;

namespace version_cpu
{
	template <typename T>
	ChromaticNumber<T>::ChromaticNumber<T>()
	{

	}
	
	template <typename T>
	unsigned long long ChromaticNumber<T>::Pow(int a, int n)
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

	template <typename T>
	T ChromaticNumber<T>::BitCount(T u)
	{
		T uCount;

		uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
		return ((uCount + (uCount >> 3)) & 030707070707) % 63;
	}

	template <typename T>
	int ChromaticNumber<T>::sgnPow(int n)
	{
		return (n & 1) == 0 ? 1 : -1;
	}

	template <typename T>
	T ChromaticNumber<T>::MaxCombinationCount(int n, int k)
	{
		if (k > n) return 0;
		
		T r = 1;
		for (int d = 1; d <= k; ++d) 
		{
			r *= n--;
			r /= d;
		}

		return r;
	}

	template <typename T>
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
		return i;
	}

	int GetBitPosition(int n, int col)
	{
		int f = 0, c = 0, i = 0;
		while (f != col)
		{
			if (n & (1 << i))
			{
				c += i;
				f++;
			}

			i++;
		}
		return c;
	}

	template <typename T>
	void ChromaticNumber<T>::BuildingIndependentSets(Graph g)
	{
		int n = g.GetVerticesCount();
		int* vertices = g.GetVertices();
		T cnt = (T)1 << n;

		// Inicjalizacja macierzy o rozmiarze 2^N (wartoœci pocz¹tkowe 0)
		independentSets = new int[cnt] {0};

		// Krok 1 algorytmu: przypisanie wartoœci 1 (iloœæ niezale¿nych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
		actualVertices = new int[n] {0};
		
		for (int i = 0; i < n; ++i)
		{
			independentSets[cnt] = 1;
			actualVertices[i] |= 1 << i;
		}

		// G³ówna funkcja tworz¹ca tablicê licznoœci zbiorów niezale¿nych dla wszystkich podzbiorów zbioru N-elementowego.
		// Zaczynamy od 1, bo krok pierwszy wykonany wy¿ej.
		for (int el = 1; el < n; el++)
		{
			T row = MaxCombinationCount(n, el + 1);

			int* newVertices = new int[row] {0};
			int l = 0;

			for (int i = 0; i < row; ++i)
			{
				// Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
				int lastIndex = GetBitPosition(actualVertices[i], el);

				for (int j = GetFirstBitPosition(actualVertices[i]) + 1; j < n; ++j)
				{
					int lastIndex2 = lastIndex;

					// Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
					for (int ns = 0, int no = 0; ns < BitCount(vertices[j]); no++)
					{
						if (vertives[j] & (1 << no))
						{
							if (actualVertices[i] & vertices[no])
								lastIndex2 -= (1 << no);

							ns++;
						}
					}

					int nextIndex = lastIndex + (1 << j);

					// Liczba zbiorów niezale¿nych w aktualnie przetwarzanym podzbiorze
					independentSets[nextIndex] = independentSets[lastIndex] + independentSets[lastIndex2] + 1;

					newVertices[l] = actualVertices[i];
					newVertices[l] |= 1 << j;

					l++;
				}
			}
			
			actualVertices = newVertices;
			delete[] newVertices;
		}
	}

	template <typename T>
	int ChromaticNumber<T>::FindChromaticNumber(Graph g)
	{
		BuildingIndependentSets(g);
		int n = g.GetVerticesCount();
		
		for (int k = 1; k <= n; ++k)
		{
			unsigned long long s = 0;
			for (T i = 0; i < Pow(2, n); ++i) s += sgnPow(n-BitCount(i)) * Pow(independentSets[i], k);

			if (s <= 0) continue;

			return k;
		}
		
		return -1;
	}
}