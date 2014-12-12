using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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
    }
}
