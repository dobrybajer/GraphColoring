using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace GraphColoring
{
    public class Graph
    {
        public List<Vertex> vertex;

        public List<int> GetNeighboursOfVertex(int i)
        {
            return vertex.ElementAt(i).Neighbours;
        }

        public List<Vertex> WVertex
        {
            get { return vertex; }
            set { vertex = value; }
        }

        private int _vertexCount;

        public int VertexCount
        {
            get { return _vertexCount; }
            set { _vertexCount = value; }
        }

        public Graph(List<Vertex> vertex)
        {
            this.vertex = vertex;
            this.VertexCount = vertex.Count();
        }

        public static Graph ReadGraph(string path)
        {
            List<Vertex> nGraph = new List<Vertex>();
            String line;

            try
            {
                using (StreamReader sr = new StreamReader(path))
                {
                    while ((line = sr.ReadLine()) != null)
                    {
                        nGraph.Add(new Vertex(creatNeighbours(line)));
                    }
                    sr.Close();
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Nie udało sie odczytać pliku :(");
                Console.WriteLine(e.Message);
            }
            return new Graph(nGraph);
        }

        public static void WriteGraph(string path, Graph graph)
        {
            List<Vertex> nGraph = new List<Vertex>();
            String line;
            try
            {
                using (StreamWriter sw = new StreamWriter(path))
                {
                    foreach (var el in graph.vertex)
                    {
                        line = "Color: " + el.Color.ToString() + "; Neighbour: ";
                        foreach (var elm in el.Neighbours)
                            line += elm.ToString() + ",";
                        sw.WriteLine(line.TrimEnd(','));
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Nie udało sie zapisać pliku :(");
                Console.WriteLine(e.Message);
            }
        }

        private static List<int> creatNeighbours(String neighbours)
        {
            List<int> tmp = new List<int>();
            string[] value = neighbours.Split(';');
            foreach (string el in value)
                tmp.Add(Int32.Parse(el));
            return tmp;
        }
    }
}
