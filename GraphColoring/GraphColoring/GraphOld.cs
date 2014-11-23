using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using GraphType = System.Collections.Generic.List<System.Collections.Generic.List<int>>;

namespace GraphColoring
{
    class GraphOld
    {
        private GraphType _g;

        public GraphType G
        {
            get { return _g; }
            set
            {
                if(value != null)
                    _g = value;
                else
                {
                    MessageBox.Show("Błędnie załadowany graf, proszę spróbować później.");
                }
            }
        }

        public GraphOld(string _fromFile)
        {
            G = new GraphType();
            LoadFile(_fromFile);
        }

        private async void LoadFile(string path)
        {
            try
            {
                using (var sr = new StreamReader(path))
                {
                    //var line = await sr.ReadToEndAsync();
                    string line;
                    while ((line = sr.ReadLine()) != null)
                    {
                        var result = Regex.Split(line, ";|,");
                        var ltmp = new List<int>();
                        for (var i = 1; i < result.Length; ++i)
                        {
                            ltmp.Add(i);
                        }
                        G.Add(ltmp);
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Błąd podczas ładowania pliku.");
            }
        }

        public GraphOld(int n)
        {
            G = new GraphType();
        }
    }
}
