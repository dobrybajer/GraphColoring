using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace GraphColoring.Structures
{
    static class FileProcessing
    {
        /// <summary>
        /// Statyczna metoda, której zadaniem jest odczytanie grafu z podanej ścieżki do pliku tekstowego, sparsowanie tekstu, oraz na jego podstawie stworzenie i zwrócenie gotowej instancji grafu.
        /// </summary>
        /// <param name="path">Ścieżka do pliku z danymi.</param>
        /// <returns>Wynikowy graf utworzony na podstawie danych z pliku tekstowego.</returns>
        public static Graph ReadFile(string path)
        {
            var vertices = new List<Vertex>();

            try
            {
                using (var sr = new StreamReader(path))
                {
                    String line;
                    while ((line = sr.ReadLine()) != null)
                    {
                        var value = line.Split(';');
                        vertices.Add(new Vertex(value.Select(Int32.Parse).ToList()));
                    }
                    sr.Close();

                    return new Graph(vertices);
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
        /// Statyczna metoda służaca do zapisania dowolnego grafu do pliku w domyślnym formacie. Więcej informacji o formacie w dokumentaji do aplikacji.
        /// </summary>
        /// <param name="path">Ścieżka do wynikowego pliku z grafem w postaci tekstowej.</param>
        /// <param name="graph">Graf do zapisania do pliku.</param>
        public static void WriteFile(string path, Graph graph)
        {
            try
            {
                using (var sw = new StreamWriter(path))
                {
                    foreach (var line in from el in graph.Vertices let line = "Neighbours: " select el.Neighbours.Aggregate(line, (current, elm) => current + (elm.ToString(CultureInfo.InvariantCulture) + ",")))
                    {
                        sw.WriteLine(line.TrimEnd(','));
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(@"Podczas zapisu grafu do pliku wystapił błąd:");
                Console.WriteLine(e.Message);
            }
        }
    }
}
