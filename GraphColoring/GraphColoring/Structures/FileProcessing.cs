using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

namespace GraphColoring.Structures
{
    /// <summary>
    /// Statyczna klasa parsująca plik znajdujący się w podanej ścieżce, a następnie zamieniająca go na obiekt grafu
    /// </summary>
    static class FileProcessing
    {
        /// <summary>
        /// Statyczna metoda, której zadaniem jest odczytanie grafu z podanej ścieżki do pliku tekstowego, sparsowanie tekstu, oraz na jego podstawie stworzenie i zwrócenie gotowej instancji grafu.
        /// </summary>
        /// <param name="path">Ścieżka do pliku z danymi.</param>
        /// <returns>Wynikowy graf utworzony na podstawie danych z pliku tekstowego.</returns>
        public static Graph ReadFile(string path)
        {
            try
            {
                bool pas = false;
                string[] tmp = path.Split(new char[] { '\\' });
                if (tmp[tmp.Length - 1][0] == 'D')
                {
                    string oldpath=tmp[0]+"\\";
                    for(int i=1;i <tmp.Length-1;i++)
                         oldpath+=tmp[i]+"\\";
                    oldpath = oldpath + "TMP" + tmp[tmp.Length - 1];
                    DIMACSParser(path, oldpath);
                    path = oldpath;
                    pas = true;
                }

                using (var sr = new StreamReader(path))
                {
                    var vertices = new List<int>();
                    var neighborsCount = new List<int>();

                    string line;

                    if (!sr.EndOfStream)
                        sr.ReadLine();

                    while ((line = sr.ReadLine()) != null)
                    {
                        vertices.AddRange(line.Split(',').Select(Int32.Parse).ToList());
                        neighborsCount.Add(vertices.Count);
                    }
                    sr.Close();

                    if(pas)
                        File.Delete(path);
                    Thread.Sleep(100);
                    return new Graph(vertices.ToArray(), neighborsCount.ToArray());
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(@"Podczas przetwarzania pliku z danymi wystapił błąd:");
                Console.WriteLine(e.Message);

                return null;
            }
        }

        public static void DIMACSParser(string path, string newFile)
	    {		
          //  fstream plik;
            // plik.open(path.c_str(), ios::in | ios::out);
            StringBuilder sb = new StringBuilder();
            StreamReader sr = new StreamReader(path);

           // if (plik.good())
            //{

            //string newF = @"..\\..\\..\\Debug\\test.txt";
            StreamWriter sw = File.CreateText(newFile);
                List<string> el = new List<string>();
                List<List<int>> elementy = new List<List<int>>();

                string wiersz, line;

                while ((line = sr.ReadLine()) != null)
                {

                    if (line.Substring(0, 1) == "p")
                    {
                        var item = line.Split(new char[] { ' ' });
                        sw.WriteLine(item[2]);
                        for (int i = 0; i < Convert.ToInt32(item[2]); i++)
                        {
                            elementy.Add(new List<int>());
                        }
                    }
                    else if (line.Substring(0, 1) == "e")
                    {
                        var ele = line.Split(new char[] { ' ' });
                        elementy.ElementAt(Convert.ToInt32(ele[1]) - 1).Add(Convert.ToInt32(ele[2]) - 1);
                        elementy.ElementAt(Convert.ToInt32(ele[2]) - 1).Add(Convert.ToInt32(ele[1]) - 1);
                    }

                }
                string wiersz2=null;
                foreach (var it in elementy)
                {
                    //string wiersz;
                    wiersz2 = null;
                    for (int j = 0; j < it.Count; j++)
                    {
                        wiersz2 += it.ElementAt(j).ToString();
                        if (j != it.Count - 1) wiersz2 = wiersz2 + ",";
                    }
                    sw.WriteLine(wiersz2);
                }

                sw.Close();
        //    }
          //  else
           // { }
	    }

        public static Graph ConvertToBitVersion(Graph g)
        {
            var n = g.VerticesCount;
            var v = new int[n];

            for (var i = 0; i < n; ++i)
            {
                for (var j = g.NeighboursCount[i - 1 > 0 ? i - 1 : 0]; j < g.NeighboursCount[i]; ++j)
                {
                    v[i] |= (1 << g.Vertices[j]);
                }
            }

            return new Graph(v, g.NeighboursCount.ToArray());
        }
    }
}
