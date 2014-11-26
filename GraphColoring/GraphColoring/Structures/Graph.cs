using System.Collections.Generic;
using System.Linq;
using GraphColoring.Algorithm;

namespace GraphColoring.Structures
{
    public class Graph
    {
        /// <summary>
        /// Pole przechowujące listę wierzchołków danego grafu.
        /// </summary>
        private readonly List<Vertex> _vertices;

        /// <summary>
        /// Metoda zwracająca sąsiadów wierzchołka i.
        /// </summary>
        /// <param name="i">Wierzchołek, dla którego szukamy sąsiadów.</param>
        /// <returns>Listą numerów sąsiadów danego wierzchołka.</returns>
        public List<int> GetNeighboursOfVertex(int i)
        {
            return _vertices.ElementAt(i).Neighbours;
        }

        /// <summary>
        /// Właściwość zwracająca listę wierzchołków danego grafu.
        /// </summary>
        public List<Vertex> Vertices
        {
            get { return _vertices; }
        }

        /// <summary>
        /// Właściwość zwracająca liczbę wierzchołków danego grafu.
        /// </summary>
        public int VertexCount
        {
            get { return _vertices.Count; }
        }

        /// <summary>
        /// Konstruktor. Na podstawie listy wierzchołków tworzy obiekt grafu.
        /// </summary>
        /// <param name="vertices">Lista wierzchołków danego grafu.</param>
        public Graph(List<Vertex> vertices)
        {
            _vertices = vertices;
        }

        /// <summary>
        /// Metoda wykonująca algorytm znajdywania k-kolorowania grafu.
        /// </summary>
        /// <returns>Liczba K-kolorowania grafu.</returns>
        public int GetChromaticNumber()
        {
            return ChromaticNumber.FindChromaticNumber(this);
        }

        /// <summary>
        /// Metoda wykonująca algorytm sprawdzania czy dany graf jest k-kolorowalny.
        /// </summary>
        /// <param name="k">Parametr k-kolorowania grafu.</param>
        /// <returns>Prawda jeśli graf jest k-kolorowalny, fałsz w p.p.</returns>
        public bool CheckChromaticNumber(int k)
        {
            return ChromaticNumber.IsChromaticNumber(this, k);
        }
    }
}
