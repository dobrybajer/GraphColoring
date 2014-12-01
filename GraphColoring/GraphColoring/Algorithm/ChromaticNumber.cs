using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Windows;
using GraphColoring.Structures;

namespace GraphColoring.Algorithm
{
    static class ChromaticNumber
    {
        /// <summary>
        /// Aktualnie przetwarzany podgraf (podzbiór zbioru wszystkich wierzchołków).
        /// </summary>
        private static List<List<int>> _actualVertices;

        /// <summary>
        /// Ilość wierzchołków w grafie.
        /// </summary>
        private static int _n;

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
        /// <param name="g">Graf na podstawie którego jest budowana tablica 2^N.</param>
        /// </summary>
        private static void BuildingIndependentSets(Graph g)
        {
            _actualVertices = new List<List<int>>();
            _n = g.VertexCount; 

            // Inicjalizacja macierzy o rozmiarze 2^N (wartości początkowe 0)
            _independentSets = new int[1 << _n, 2];

            // Krok 1 algorytmu: przypisanie wartości 1 (ilość niezależnych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
            for (var i = 0; i < _n; ++i)
            {
                _independentSets[1 << i, 0] = 1;
                _independentSets[1 << i, 1] = 1;

                _actualVertices.Add(new List<int> { i });
            }

            // Główna funkcja tworząca tablicę liczności zbiorów niezależnych dla wszystkich podzbiorów zbioru N-elementowego.
            // Zaczynamy od 1, bo krok pierwszy wykonany wyżej.
            for (var j = 1; j < _n; j++)
            {
                var newVertices = new List<List<int>>();

                foreach (var l in _actualVertices)
                {
                    var lastIndex = 0;

                    // Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
                    for (var index = 0; index < l.Count; ++index)
                        lastIndex += 1 << l[index];
                    
                    for (var i = l[l.Count - 1] + 1; i < _n; ++i)
                    {
                        var lastIndex2 = lastIndex;

                        // Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
                        for (var ns = 0; ns < g.GetNeighboursOfVertex(i).Count; ns++)
                        {
                            var e = g.GetNeighboursOfVertex(i)[ns];

                            if (l.Contains(e))
                                lastIndex2 -= 1 << e;
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
        /// <param name="g">Wejściowy graf, dla którego liczymy liczbę chromatyczną.</param>
        /// <returns>Wartość minimalnego k-kolorowania bądź 0, jeśli takiego kolorowania nie ma (sic!)</returns>
        public static int FindChromaticNumber(Graph g)
        {
            var watch = Stopwatch.StartNew();

            BuildingIndependentSets(g);

            for (var k = 1; k <= _n; ++k)
            {
                long s = 0;
                for (var i = 0; i < Pow(2, _n); ++i) s += Pow(-1, _n - _independentSets[i, 1]) * Pow(_independentSets[i, 0], k);

                if (s <= 0) continue;

                watch.Stop();
                MessageBox.Show(string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}ms", k, watch.ElapsedMilliseconds));
 
                return k;
            }
            return -1;
        }

        /// <summary>
        /// Funkcja sprawdza czy dany graf da się pokolorować na k kolorów.
        /// </summary>
        /// <param name="g">Wejściowy graf, dla którego liczymy liczbę chromatyczną.</param>
        /// <param name="k">Liczba kolorów dla danego kolorowania grafu.</param>
        /// <returns>True - graf jest k-kolorowalny, false - w p.p.</returns>
        public static bool IsChromaticNumber(Graph g, int k)
        {
            var watch = Stopwatch.StartNew();

            BuildingIndependentSets(g);

            long s = 0;
            for (var i = 0; i < Pow(2, _n); ++i) s += Pow(-1, _n - _independentSets[i, 1]) * Pow(_independentSets[i, 0], k);

            watch.Stop();

            MessageBox.Show(s > 0
                ? string.Format("Graf jest {0}-kolorowalny\nCzas obliczeń: {1}ms", k, watch.ElapsedMilliseconds)
                : string.Format("Graf nie jest {0}-kolorowalny.\nCzas obliczeń: {1}ms", k, watch.ElapsedMilliseconds));

            return s > 0;
        }
    }
}
