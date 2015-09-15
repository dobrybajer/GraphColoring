﻿#include "Algorithm.h"
#include "Windows.h"
#include "psapi.h"

/// <summary>
/// Przestrzeń nazwy dla algorytmu kolorowania grafów w wersji CPU napisanej w języku C++.
/// </summary>
namespace version_cpu
{
	const double ToMb = 1048576;

	/// <summary>
	/// Metoda zwraca aktualne zużycie pamięci RAM przez aplikację.
	/// </summary>
	/// <returns>Rozmiar pamięci.</returns>
	double getUsedMemory()
	{
		PROCESS_MEMORY_COUNTERS pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));

		return pmc.WorkingSetSize / ToMb;
	}

	/// <summary>
	/// Szybkie podnoszenie danej liczby do podanej potęgi. Pozwala na potęgowanie liczby, której
	/// wynik jest nie większy od rozmiaru typu INT.
	/// </summary>
	/// <param name="a">Podstawa potęgi.</param>
	/// <param name="n">Wykładnik potęgi.</param>
	/// <returns>Wynik potęgowania.</returns>
	unsigned int Pow(int a, int n)
	{
		if (n <= 0) return 1;
		if (n == 1) return a;
		if (a <= 0) return 0;
		
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

	/// <summary>
	/// Szybkie i efektywne podnoszenie do potęgi liczby -1. Polega na sprawdzaniu parzystości
	/// wykładnika potęgi.
	/// </summary>
	/// <param name="n">Wykładnik potęgi.</param>
	/// <returns>Wynik potęgowania.</returns>
	int sgnPow(int n)
	{
		return (n & 1) == 0 ? 1 : -1;
	}

	/// <summary>
	/// Funkcja zliczająca liczbę ustawionych bitów w reprezentacji bitowej wejściowej liczby.
	/// W przypadku algorytmu, służy do wyznaczania ilości elementów w aktualnie rozpatrywanym podzbiorze.
	/// </summary>
	/// <param name="n">Liczba wejściowa.</param>
	/// <returns>Liczba ustawionych bitów w danej liczbie wejściowej.</returns>
	int BitCount(int n)
	{
		if (n <= 0) return 0;

		int uCount = n - ((n >> 1) & 033333333333) - ((n >> 2) & 011111111111);
		return ((uCount + (uCount >> 3)) & 030707070707) % 63;
	}
	
	/// <summary>
	/// Wyznaczanie kombinacji k - elementowych zbioru n - elementowego (kombinacje bez powtórzeń).
	/// Ograniczone możliwości ze względu na możliwie zbyt dużą wielkość wyniku.
	/// </summary>
	/// <param name="n">Liczba elementów w zbiorze.</param>
	/// <param name="k">Liczba elementów w kombinacji.</param>
	/// <returns>Liczba oznaczająca kombinację n po k.</returns>
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

	/// <summary>
	/// Wyznacza pozycję najbardziej znaczącego bitu w reprezentacji bitowej liczby wejściowej.
	/// W przypadku algorytmu używane jest do wyznaczania największego elementu w rozpatrywanym podzbiorze.
	/// Używane tylko w wersji bitowej algorytmu.
	/// </summary>
	/// <param name="n">Liczba wejściowa.</param>
	/// <returns>Pozycja najbardziej znaczącego bitu.</returns>
	unsigned int GetFirstBitPosition(int n)
	{
		if(n <= 0) return 0;

		int ret = 0;

		while (n >>= 1)
    		ret++;
		
		return ret;
	}

	/// <summary>
	/// Wyznaczanie sumy wszystkich bitów aż do k-tego bitu włącznie.
	/// Używane tylko w wersji bitowej algorytmu.
	/// </summary>
	/// <param name="n">Liczba wejściowa.</param>
	/// <param name="k">Liczba bitów równych 1 brana do całkowitej sumy.</param>
	/// <returns>Obliczona suma.</returns>
	unsigned int GetBitPosition(int n, int k)
	{
		if (k <= 0 || n <= 0) return 0;

		unsigned int f = 0, c = 0, i = 0;
		while (f != k && (1 << i) <= n)
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

	/// <summary>
	/// Pomocnicza funkcja alokująca pamięć dla tablicy dwuwymiarowej o podanych w parametrach
	/// wyjściowych wymiarach.
	/// Używane tylko w wersji tablicowej algorytmu.
	/// </summary>
	/// <param name="row">Liczba wierszy.</param>
	/// <param name="col">Liczba kolumn.</param>
	/// <returns>Wskaźnik na utworzoną dwuwymiarową tablicę.</returns>
	int** CreateVertices(int row, int col)
	{
		int** nVertices = new int*[row];
		for (int i = 0; i < row; ++i)
			nVertices[i] = new int[col]();

		return nVertices;
	}

	void DeleteVertices(int** tab, int rowCount)
	{
		for (int i = 0; i < rowCount; ++i)
			delete[] tab[i];
		
		delete[] tab;
	}

	/// <summary>
	/// Funckja tworząca tablicę zbiorów niezależnych dla podanego grafu wejściowego. Wersja bitowa,
	/// tj. informacja o sąsiadach danego wierzchołka przechowywana jest w samej liczbie, poprzez
	/// ustawienie odpowiedniego bitu. Ogranicza to liczbę wierzchołków do 32 (w przypadku użycia
	/// typu INT) lub do 64 (w przypadku użycia typu UNSIGNED LONG LONG). 
	/// Zaletą jest około dwukrotne przyspieszenie obliczeń w stosunku do wersji tablicowej.
	/// </summary>
	/// <param name="vertices">
	/// Graf wejściowy reprezentowany jako tablica liczb, zawierających informację o sąsiadach
	/// każdego wierzchołka.
	/// </param>
	/// <param name="n">Liczba wierzchołków w grafie.</param>
	/// <returns>Tablica zbiorów niezależnych.</returns>
	int* BuildingIndependentSets_BitVersion(double* memory, int* vertices, int n)
	{
		int* independentSets;
		int* actualVertices;
		int* newVertices;

		int actualRow = n;

		independentSets = new int[1 << n]();

		actualVertices = new int[n]();
		newVertices = new int[1]();

		for (int i = 0; i < n; ++i)
		{
			independentSets[(1 << i)] = 1;
			actualVertices[i] |= (1 << i);
		}

		for (int el = 1; el < n; el++)
		{
			int row = Combination_n_of_k(n, el + 1);

			newVertices = new int[row]();
			int l = 0;

			for (int i = 0; i < actualRow; ++i)
			{
				unsigned int lastIndex = GetBitPosition(actualVertices[i], el);

				for (int j = GetFirstBitPosition(actualVertices[i]) + 1; j < n; ++j)
				{
					unsigned int lastIndex2 = lastIndex;

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

					independentSets[nextIndex] = independentSets[lastIndex] + independentSets[lastIndex2] + 1;

					newVertices[l] = actualVertices[i];
					newVertices[l] |= (1 << j);

					l++;
				}
			}

			memory[el] = getUsedMemory();

			actualRow = row;
			delete[] actualVertices;

			if (el != n - 1)
				actualVertices = newVertices;
		}
		
		delete[] newVertices;

		memory[n] = getUsedMemory();

		return independentSets;
	}

	/// <summary>
	/// Funckja tworząca tablicę zbiorów niezależnych dla podanego grafu wejściowego. 
	/// Wersja tablicowa, tj. informacja o sąsiadach danego wierzchołka przechowywana jest w drugim
	/// wymiarze tablicy grafu wejściowego. Zwiększa to zużycie pamięci RAM oraz około dwukrotnie
	/// wydłuża czas obliczeń w stosunku do wersji bitowej, ale nie ogranicza* problemu do 32 lub 64 wierzchołków.
	/// </summary>
	/// <param name="vertices">Lista wszystkich sąsiadów każdego z wierzchołków.</param>
	/// <param name="vertices">Lista pozycji początkowych sąsiadów dla danego wierzchołka.</param>
	/// <param name="n">Liczba wierzchołków w grafie.</param>
	/// <returns>Tablica zbiorów niezależnych.</returns>
	int* BuildingIndependentSets_TableVersion(double* memory, int* vertices, int* offset, int n)
	{
		int* independentSets;
		int** actualVertices;
		int** newVertices;

		int actualVerticesRowCount;
		int actualVerticesColCount;

		independentSets = new int[1 << n] ();
		actualVertices = CreateVertices(n, 1);
		newVertices = CreateVertices(1, 1);

		actualVerticesRowCount = n;
		actualVerticesColCount = 1;
		
		for (int i = 0; i < n; ++i)
		{
			independentSets[1 << i] = 1;
			actualVertices[i][0] = i;
		}

		for (int el = 1; el < n; el++)
		{
			int col = el + 1;
			int row = Combination_n_of_k(n, col);
			
			newVertices = CreateVertices(row, col);
			int l = 0;

			for (int i = 0; i < actualVerticesRowCount; ++i)
			{
				int lastIndex = 0;

				for (int index = 0; index < actualVerticesColCount; ++index)
					lastIndex += (1 << actualVertices[i][index]);

				for (int j = actualVertices[i][actualVerticesColCount - 1] + 1; j < n; ++j)
				{
					int lastIndex2 = lastIndex;

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

					independentSets[nextIndex] = independentSets[lastIndex] + independentSets[lastIndex2] + 1;

					for (int k = 0; k < el; ++k)
						newVertices[l][k] = actualVertices[i][k];

					newVertices[l][el] = j;

					l++;
				}
			}

			memory[el] = getUsedMemory();

			DeleteVertices(actualVertices, actualVerticesRowCount);

			if (el != n - 1)
				actualVertices = newVertices;

			actualVerticesRowCount = row;
			actualVerticesColCount = col;
		}

		DeleteVertices(newVertices, actualVerticesRowCount);

		memory[n] = getUsedMemory();

		return independentSets;
	}

	/// <summary>
	/// Główna funkcja algorytmu. Uruchamia wyznaczanie zbioru niezależnego 
	/// (odpowiednia metoda w zależnoci od ostatniego parametru), a następnie wylicza k - kolorowalność grafu.
	/// </summary>
	/// <param name="vertices">Graf wejściowy reprezentowany zależnie od wyboru metody.</param>
	/// <param name="offset">Lista pozycji początkowych sąsiadów dla danego wierzchołka (używane tylko gdy flaga ustawiona na 0).</param>
	/// <param name="n">Liczba wierzchołków w grafie.</param>
	/// <param name="flag">
	/// Flaga informująca o wyborze metody wyznaczania zbioru niezależnego:
	/// - 0 (wartość domyślna) - metoda tablicowa
	/// - 1 - metoda bitowa
	/// Parametr opcjonalny.
	/// </param>
	/// <returns>Licza k oznaczająca k - kolorowalność grafu, bądź wartość -1 w przypadku błędu.</returns>
	int FindChromaticNumber(double* memoryUsage, int* vertices, int* offset, int n, int flag)
	{
		memoryUsage[0] = getUsedMemory();

		int* independentSets = flag == 1 ? 
			BuildingIndependentSets_BitVersion(memoryUsage, vertices, n) :
			BuildingIndependentSets_TableVersion(memoryUsage, vertices, offset, n);

		int PowerNumber = (1 << n);

		for (int k = 1; k <= n; ++k)
		{
			unsigned int s = 0;
			
			for (int i = 0; i < PowerNumber; ++i) s += (sgnPow(BitCount(i)) * Pow(independentSets[i], k));
			
			if (s <= 0) continue;

			memoryUsage[n + 1] = getUsedMemory();

			delete[] independentSets;

			memoryUsage[n + 2] = getUsedMemory();

			return k;
		}

		memoryUsage[n + 1] = getUsedMemory();

		delete[] independentSets;

		memoryUsage[n + 2] = getUsedMemory();

		return -1;
	}
}