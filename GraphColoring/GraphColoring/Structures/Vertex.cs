using System.Collections.Generic;

namespace GraphColoring.Structures
{
    public class Vertex
    {
        /// <summary>
        /// Sąsiedzi (wierzchołki połączone krawędzią) z danym wierzchołkiem.
        /// </summary>
        public List<int> Neighbours { get; set; }

        /// <summary>
        /// Konstruktor. Na podstawie danej listy liczb, tworzy listę sąsiadów danego wierzchołka.
        /// </summary>
        /// <param name="neighbours">Lista składająca się z numerów wierzchołków (liczby).</param>
        public Vertex(List<int> neighbours)
        {
            Neighbours = neighbours;
        }

    }
}
