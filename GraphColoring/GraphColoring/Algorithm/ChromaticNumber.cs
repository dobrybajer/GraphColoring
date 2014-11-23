using System.Collections.Generic;
using System.Linq;

namespace GraphColoring.Algorithm
{
    class ChromaticNumber
    {
        /// <summary>
        /// Aktualnie przetwarzany podgraf (podzbiór zbioru wszystkich wierzchołków).
        /// </summary>
        public List<List<int>> ActualVertices = new List<List<int>>();

        /// <summary>
        /// Ilość wierzchołków w grafie.
        /// </summary>
        public int N;

        /// <summary>
        /// Tablica o rozmiarze 2^N zawierająca ilość niezależnych zbiorów dla każdego podzbioru zbioru N-elementowego (1. kolumna) oraz
        /// liczbę elementów w danym podzbiorze (2. kolumna).
        /// </summary>
        public int[,] IndependentSets = null;

        /// <summary>
        /// Funkcja podnosząca liczbę do podanej potęgi. Ze względu na operacje na typie long, ma większy zakres potęgowania niż Math.Pow.
        /// </summary>
        /// <param name="a">Podstawa potęgi.</param>
        /// <param name="n">Wykładnik potęgi.</param>
        /// <returns>Wynik potęgowania.</returns>
        public long Pow(int a, int n)
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
        /// <param name="g">Wejściowy graf.</param>
        public void BuildingIndependentSets(Graph g)
        {
            N = g.VertexCount;

            // Inicjalizacja macierzy o rozmiarze 2^N (wartości początkowe 0)
            IndependentSets = new int[1 << N, 2];

            // Krok 1 algorytmu: przypisanie wartości 1 (ilość niezależnych zbiorów) dla podzbiorów 1-elementowych, oraz dodanie ich do aktualnie przetwarzanych elementów (1 poziom tworzenia wszystkich podzbiorów)
            for (var i = 0; i < N; ++i)
            {
                IndependentSets[1 << i, 0] = 1;
                IndependentSets[1 << i, 1] = 1;

                ActualVertices.Add(new List<int> { i });
            }

            // Główna funkcja tworząca tablicę liczności zbiorów niezależnych dla wszystkich podzbiorów zbioru N-elementowego.
            // Zaczynamy od 1, bo krok pierwszy wykonany wyżej.
            for (var j = 1; j < N; j++)
            {
                var newVertices = new List<List<int>>();

                foreach (var l in ActualVertices)
                {
                    var lastIndex = 0;

                    // Sprawdzenie indeksu poporzedniego zbioru dla rozpatrywanego podzbioru
                    for (var index = 0; index < l.Count; ++index)
                        lastIndex += 1 << l[index];
                    
                    for (var i = l[l.Count - 1] + 1; i < N; ++i)
                    {
                        var lastIndex2 = lastIndex;

                        // Sprawdzenie indeksu poprzedniego zbioru dla rozpatrywanego podzbioru \ {i}
                        for (var ns = 0; ns < g.GetNeighboursOfVertex(i).Count; ns++)
                        {
                            var e = g.GetNeighboursOfVertex(i)[ns];

                            if (l.Contains(e))
                                lastIndex2 -= 1 << e;
                        }

                        var nextIndex = lastIndex + 1 << i;
                        
                        // Liczba zbiorów niezależnych w aktualnie przetwarzanym podzbiorze
                        IndependentSets[nextIndex, 0] = IndependentSets[lastIndex, 0] + IndependentSets[lastIndex2, 0] + 1;

                        var list = new List<int>(l) {i};

                        newVertices.Add(list);
                        IndependentSets[nextIndex, 1] = list.Count();
                    }
                }
                ActualVertices = newVertices;
            } 
        }

        /// <summary>
        /// Funkcja informuje na ile minimalnie kolorów może być pokolorowany dany graf.
        /// </summary>
        /// <param name="g">Wejściowy graf.</param>
        /// <returns>Wartość minimalnego k-kolorowania bądź 0, jeśli takiego kolorowania nie ma (sic!)</returns>
        public int FindChromaticNumber(Graph g)
        {
            BuildingIndependentSets(g);

            for (var k = 1; k <= N; ++k)
            {
                long s = 0;
                for (var i = 0; i < Pow(2, N); ++i) s += Pow(-1, N - IndependentSets[i, 1]) * Pow(IndependentSets[i, 0], k);

                if (s > 0) return k;
            }
            return 0;
        }

        /// <summary>
        /// Funkcja sprawdza czy dany graf da się pokolorować na k kolorów.
        /// </summary>
        /// <param name="g">Wejściowy graf.</param>
        /// <param name="k">Liczba kolorów dla danego kolorowania grafu.</param>
        /// <returns>True - graf jest k-kolorowalny, false - w p.p.</returns>
        public bool IsChromaticNumber(Graph g, int k)
        {
            BuildingIndependentSets(g);

            long s = 0;
            for (var i = 0; i < Pow(2, N); ++i) s += Pow(-1, N - IndependentSets[i, 1]) * Pow(IndependentSets[i, 0], k);

            return s > 0;
        }
    }
}
