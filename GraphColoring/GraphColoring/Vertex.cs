using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GraphColoring
{
    class Vertex
    {
        int color;
        List<int> neighbours;

        public int Color
        {
            get { return color; }
            set { color = value; }
        }
        public List<int> Neighbours
        {
            get { return neighbours; }
            set { neighbours = value; }
        }

        public Vertex(List<int> neighbours)
        {
            this.color = 0;
            this.neighbours = neighbours;
        }

    }
}
