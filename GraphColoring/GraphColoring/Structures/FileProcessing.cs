using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

namespace GraphColoring.Structures
{
    /// <summary>
    /// Statyczna klasa parsująca plik znajdujący się w podanej ścieżce, a następnie zamieniająca go na obiekt grafu
    /// </summary>
    internal static class FileProcessing
    {
        /// <summary>
        /// Statyczna metoda, której zadaniem jest odczytanie grafu z podanej ścieżki do pliku tekstowego, sparsowanie tekstu,
        /// oraz na jego podstawie stworzenie i zwrócenie gotowej instancji grafu.
        /// </summary>
        /// <param name="path">Ścieżka do pliku z danymi.</param>
        /// <returns>Wynikowy graf utworzony na podstawie danych z pliku tekstowego.</returns>
        public static Graph ReadFile(string path)
        {
            try
            {
                var pas = false;
                var tmp = path.Split('\\');
                if (tmp[tmp.Length - 1][0] == 'D')
                {
                    var oldpath = tmp[0] + "\\";

                    for (var i = 1; i < tmp.Length - 1; i++)
                        oldpath += tmp[i] + "\\";

                    oldpath = oldpath + "TMP" + tmp[tmp.Length - 1];
                    DimacsParser(path, oldpath);
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

                    if (pas)
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

        /// <summary>
        /// Statyczna metoda konwertująca graf w formacie DIMACS, na graf w formacie używanym przez aplikację.
        /// </summary>
        /// <param name="path">Ścieżka do pliku z reprezentacją grafu w formacie DIMACS</param>
        /// <param name="newFile">Tekst zawierający skonwertowany graf w formacie używanym w aplikacji.</param>
        public static void DimacsParser(string path, string newFile)
        {
            var sr = new StreamReader(path);
            var sw = File.CreateText(newFile);

            var elementy = new List<List<int>>();

            string line;

            while ((line = sr.ReadLine()) != null)
            {
                switch (line.Substring(0, 1))
                {
                    case "p":
                        var item = line.Split(' ');
                        sw.WriteLine(item[2]);
                        for (var i = 0; i < Convert.ToInt32(item[2]); i++)
                        {
                            elementy.Add(new List<int>());
                        }
                        break;
                    case "e":
                        var ele = line.Split(' ');
                        elementy.ElementAt(Convert.ToInt32(ele[1]) - 1).Add(Convert.ToInt32(ele[2]) - 1);
                        elementy.ElementAt(Convert.ToInt32(ele[2]) - 1).Add(Convert.ToInt32(ele[1]) - 1);
                        break;
                }
            }
            foreach (var it in elementy)
            {
                string wiersz = null;
                for (var j = 0; j < it.Count; j++)
                {
                    wiersz += it.ElementAt(j).ToString();
                    if (j != it.Count - 1) wiersz = wiersz + ",";
                }
                sw.WriteLine(wiersz);
            }

            sw.Close();
        }

        /// <summary>
        /// Funkcja konwertująca graf z formatu tablicowego, na format bitowy (oba używane przez różne modyfikacje tego samego algorytmu).
        /// </summary>
        /// <param name="g">Graph w formacie używanym przez aplikację, w wersji tablicowej.</param>
        /// <returns>Graph w formacie używanym przez aplikację, w wersji bitowej.</returns>
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