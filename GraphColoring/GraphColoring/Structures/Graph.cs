using GraphColoring.Algorithm;

namespace GraphColoring.Structures
{
    /// <summary>
    /// Publiczna klasa przechowująca informację o grafie w obiekcie.
    /// </summary>
    public class Graph
    {
        /// <summary>
        /// Pole przechowujące listę sąsiadów dla wszystkich wierzchołków po kolei.
        /// </summary>
        private readonly int[] _vertices;

        /// <summary>
        /// Pole przechowujące listę indeksów ostatniego sąsiada danego wierzchołka. Indeksy odnoszą się do tablicy  "_vertices"
        /// </summary>
        private readonly int[] _neighboursCount;

        /// <summary>
        /// Właściwość zwracająca listę sąsiadów dla wszystkich wierzchołków po kolei.
        /// </summary>
        public int[] Vertices
        {
            get { return _vertices; }
        }

        /// <summary>
        /// Właściwość zwracająca listę indeksów ostatniego sąsiada danego wierzchołka.
        /// </summary>
        public int[] NeighboursCount
        {
            get { return _neighboursCount; }
        }

        /// <summary>
        /// Właściwość zwracająca liczbę wierzchołków danego grafu.
        /// </summary>
        public int VerticesCount
        {
            get { return _neighboursCount.Length; }
        }

        /// <summary>
        /// Konstruktor. Na podstawie listy wierzchołków tworzy obiekt grafu.
        /// </summary>
        /// <param name="vertices">Lista sąsiadów każdego z wierzchołków danego grafu.</param>
        /// <param name="neighboursCount">Lista ostatnich numerów sąsiadów wierzchołka z tablicy "vertices".</param>
        public Graph(int[] vertices, int[] neighboursCount)
        {
            _vertices = vertices;
            _neighboursCount = neighboursCount;
        }

        /// <summary>
        /// Metoda wykonująca algorytm znajdywania k-kolorowania grafu.
        /// </summary>
        /// <returns>Liczba K-kolorowania grafu.</returns>
        public int GetChromaticNumber()
        {
            return ChromaticNumber.FindChromaticNumber(Vertices, NeighboursCount, VerticesCount);
        }
    }
}
