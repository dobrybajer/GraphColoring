using System.Collections.Generic;
using System.Linq;

namespace GraphColoring.Algorithm
{
    /// <summary>
    /// Statyczna klasa obliczająca problem kolorowania grafu metodą włączeń-wyłączeń w niezoptymalizowanej wersji napisanej w języku C#.
    /// </summary>
    static class ChromaticNumber
    {
        /// <summary>
        /// Aktualnie przetwarzany podgraf (podzbiór zbioru wszystkich wierzchołków).
        /// </summary>
        private static List<List<int>> _actualVertices;

        /// <summary>
        /// Tablica o rozmiarze 2^N zawierająca ilość niezależnych zbiorów dla każdego podzbioru zbioru N-elementowego (1. kolumna) oraz
        /// liczbę elementów w danym podzbiorze (2. kolumna).
        /// </summary>
        private static int[,] _independentSets;

        /// <summary>
        /// Funkcja podnosząca liczbę do podanej potęgi. Ze względu na operacje na typie long, ma większy zakres potęgowania niż Math.Pow.
        /// </summary>
        /// <param name="a">Podstawa potęgi.</param>
        /// <param name="n">Wykładnik potęgi.</param>
        /// <returns>Wynik potęgowania.</returns>
        private static long Pow(int a, int n)
        {
            long s = 1;

            while (n != 0)
            {
                s = s * a;
                n--;
            }

            return s;
        }
     
        /// <summary>
        /// Zapisanie w globalnej tablicy, zbiorów niezależnych dla 2^N podzbiorów wierzchołków grafu wejściowego.
        /// </summary>
        /// <param name="vertices">Lista sąsiadów każdego wierzchołka.</param>
        /// <param name="neighborsCount">Lista indeksów ostatniego sąsiada dla każdego wierzchołka.</param>
        /// <param name="verticesCount">Liczba wierzchołków danego grafu.</param>
        private static void BuildingIndependentSets(int[] vertices, int[] neighborsCount, int verticesCount)
        {
            _actualVertices = new List<List<int>>();
            var n = verticesCount;
            var neighbors = vertices;
            var offset = neighborsCount;

            // Inicjalizacja macierzy o rozmiarze 2^N (wartości początkowe 0)
            _independentSets = new int[1 << n, 2];

            // Krok 1 algorytmu: przypisanie wartości 1 (ilość niezależnych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
            for (var i = 0; i < n; ++i)
            {
                _independentSets[1 << i, 0] = 1;
                _independentSets[1 << i, 1] = 1;

                _actualVertices.Add(new List<int> { i });
            }

            // Główna funkcja tworząca tablicę liczności zbiorów niezależnych dla wszystkich podzbiorów zbioru N-elementowego.
            // Zaczynamy od 1, bo krok pierwszy wykonany wyżej.
            for (var j = 1; j < n; j++)
            {
                var newVertices = new List<List<int>>();

                foreach (var l in _actualVertices)
                {
                    var lastIndex = 0;

                    // Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
                    for (var index = 0; index < l.Count; ++index)
                        lastIndex += 1 << l[index];
                    
                    for (var i = l[l.Count - 1] + 1; i < n; ++i)
                    {
                        var lastIndex2 = lastIndex;

                        // Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
                        for (var ns = offset[i-1]; ns < offset[i]; ns++)
                        {
                            var e = neighbors[ns];

                            if (l.Contains(e))
                                lastIndex2 -= (1 << e);
                        }

                        var nextIndex = lastIndex + (1 << i);
                        
                        // Liczba zbiorów niezależnych w aktualnie przetwarzanym podzbiorze
                        _independentSets[nextIndex, 0] = _independentSets[lastIndex, 0] + _independentSets[lastIndex2, 0] + 1;

                        var list = new List<int>(l) {i};

                        newVertices.Add(list);
                        _independentSets[nextIndex, 1] = list.Count();
                    }
                }
                _actualVertices = newVertices;
            } 
        }

        /// <summary>
        /// Funkcja informuje na ile minimalnie kolorów może być pokolorowany dany graf.
        /// </summary>
        /// <param name="vertices">Lista sąsiadów każdego wierzchołka.</param>
        /// <param name="neighborsCount">Lista indeksów ostatniego sąsiada dla każdego wierzchołka.</param>
        /// <param name="verticesCount">Liczba wierzchołków danego grafu.</param>
        /// <returns>Wartość minimalnego k-kolorowania bądź 0, jeśli takiego kolorowania nie ma (sic!)</returns>
        public static int FindChromaticNumber(int[] vertices, int[] neighborsCount, int verticesCount)
        {
            BuildingIndependentSets(vertices, neighborsCount, verticesCount);

            for (var k = 1; k <= verticesCount; ++k)
            {
                long s = 0;
                for (var i = 0; i < Pow(2, verticesCount); ++i) s += Pow(-1, verticesCount - _independentSets[i, 1]) * Pow(_independentSets[i, 0], k);

                if (s <= 0) continue;

                return k;
            }
            return -1;
        }

    }
}
