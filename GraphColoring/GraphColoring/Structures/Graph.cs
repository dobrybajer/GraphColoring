﻿namespace GraphColoring.Structures
{
    /// <summary>
    /// Publiczna klasa przechowująca informację o grafie w obiekcie.
    /// </summary>
    public class Graph
    {
        /// <summary>
        /// Pole przechowujące wartość gęstości grafu.
        /// </summary>
        private readonly double _density;

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
        /// Właściwość zwracająca liczbę wszystkich sąsiadów każdego z wierchołków
        /// </summary>
        public int AllVerticesCount
        {
            get { return _vertices.Length; }
        }

        /// <summary>
        /// Właściwość zwracająca gęstość grafu jako stosunek sumy liczby wszystkich sąsiadów wszystkich wierzchołków do liczby sąsiadów w grafie pełnym dla danego rozmiaru zadania.
        /// </summary>
        public double Density
        {
            get { return _density; }
        }

        /// <summary>
        /// Konstruktor. Na podstawie listy wierzchołków tworzy obiekt grafu.
        /// </summary>
        /// <param name="vertices">Lista sąsiadów każdego z wierzchołków danego grafu.</param>
        /// <param name="neighboursCount">Lista ostatnich numerów sąsiadów wierzchołka z tablicy "vertices".</param>
        /// <param name="density">Gęstość grafu.</param>
        public Graph(int[] vertices, int[] neighboursCount, double density)
        {
            _vertices = vertices;
            _neighboursCount = neighboursCount;
            _density = density;
        }
    }
}
