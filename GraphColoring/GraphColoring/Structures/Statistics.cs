using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualBasic.Devices;

namespace GraphColoring.Structures
{
    /// <summary>
    /// Klasa przechowująca dane o statystykach. Umożliwia zapis statystyk do pliku, a także informuje o przewidywanym czasie oraz pamięci potrzebnej do obliczenia algorytmu.
    /// </summary>
    class Statistics
    {
        #region Zmienne

        private const double ToMb = 1048576; 
        private readonly List<GraphStat> _statsList;

        #endregion

        #region Konstruktor

        /// <summary>
        /// Domyślny konstruktor. Tworzy nowy obiket listy zawierającej statystyki dla każdego problemu.
        /// </summary>
        public Statistics()
        {
            _statsList = new List<GraphStat>();
        }

        #endregion

        #region Zarządzanie statystykami
        /// <summary>
        /// Metoda dodająca statystyki dla danego problemu. Identyfikowanie następuje po nazwie pliku tekstowego zawierającego reprezentację grafu.
        /// </summary>
        /// <param name="fileName">Nazwa pliku tekstowego zawierającego reprezentację grafu.</param>
        /// <param name="n">Liczba wierzchołków wejścciowego grafu.</param>
        /// <param name="type">Typ algorytmu wywołującego metodę.</param>
        /// <param name="time">Czas obliczeń danego algorytmu.</param>
        /// <param name="memoryUsage">Tablica zawierająca pamięć używaną przez aplikację podczas działania algorytmu.</param>
        public void Add(string fileName, int n, int type, TimeSpan time, int[] memoryUsage)
        {
            var i = _statsList.FindIndex(x => x.FileName == fileName);
            if (i == -1)
                _statsList.Add(new GraphStat(fileName, n, type, time, memoryUsage));
            else
                _statsList[i].Update(type, time, memoryUsage);
        }

        /// <summary>
        /// Metoda zapisuje bieżące statysyki do pliku
        /// </summary>
        /// <param name="path">Ścieżka do pliku ze statystykami.</param>
        public void SaveToFile(string path)
        {
            var content = _statsList.Aggregate("", (current, g) => current + g.SaveToFile());
            File.WriteAllText(path, content);
        }

        /// <summary>
        /// Funkcja zwraca statystyki w postaci tekstu.
        /// </summary>
        /// <returns>Tekst zawierający statystyki.</returns>
        public string DisplayStats()
        {
            return _statsList.Aggregate("", (current, g) => current + g.SaveToFile());
        }

        /// <summary>
        /// Metoda zwracająca informację czy istnieją w pamięci dostępne statystyki obliczeń.
        /// </summary>
        /// <returns>True - statystyki są dostępne, false - w p.p.</returns>
        public bool IsEmpty()
        {
            return _statsList.Count == 0;
        }

        #endregion

        #region Obliczanie przewidywanego czasu oraz pamięci
        /// <summary>
        /// Estymowanie pamięci potrzebnej do wykonania obliczeń dla konkretnego zadania.
        /// </summary>
        /// <param name="n">Liczba wierzchołków wejściowego grafu.</param>
        /// <param name="type">Typ algorytmu wywołującego metodę.</param>
        /// <returns>Dwuelementowa tablica double zawierająca dane o pamięci potrzebnej oraz całkowitej pamięci wirtualnej dostępnej dla aplikacji (podane w MB). W przypadku gdy obliczenie może być wykonane, zwrócona zostanie wartość NULL.</returns>
        public double[] PredictSpace(int n, int type)
        {
            var pcInfo = new ComputerInfo();

            var vertices = Combination_n_of_k((ulong)n, (ulong)n / 2);
            var maxColumnCount = (ulong)(n + 1) / 2;
            double totalMemoryRequired = 0;

            switch (type)
            {
                case 0: // GPU
                    return new[] { (Math.Pow(2, n) + 2 * vertices * maxColumnCount + vertices) * sizeof(int) / ToMb };
                case 1: // CPU Table
                    totalMemoryRequired = (Math.Pow(2, n) + 2 * vertices * maxColumnCount) * sizeof(int);
                    break;
                case 2: // CPU Bit
                    totalMemoryRequired = (Math.Pow(2, n) + 2 * vertices) * sizeof(int);
                    break;
            }

            return totalMemoryRequired > pcInfo.AvailableVirtualMemory
                ? new[] {totalMemoryRequired / ToMb, pcInfo.AvailableVirtualMemory / ToMb}
                : null;
        }

        /// <summary>
        /// Estymowanie czasu potrzebnego do wykonania obliczeń dla konkretnego zadania.
        /// </summary>
        /// <param name="n">Liczba wierzchołków wejściowego grafu.</param>
        /// <param name="type">Typ algorytmu wywołującego metodę.</param>
        /// <returns>Czas potrzebny do wykonania obliczeń bądź TimeSpan(0) w przypadku braku danych.</returns>
        public TimeSpan PredictTime(int n, int type)
        {
            if (_statsList.Count == 0)
                return new TimeSpan(0);

            List<TimeSpan> times = null;

            var processedVerticesCount = 0;
            GraphStat tmp;

            switch (type)
            {
                    
                case 0: // GPU
                    tmp = _statsList.OrderByDescending(y => y.VerticesCount).FirstOrDefault(x => x.GpuTime.Count != 0);
                    if (tmp != null)
                    {
                        processedVerticesCount = tmp.VerticesCount;
                        times = tmp.GpuTime;
                    }
                    break;
                case 1: // CPU Table
                    tmp = _statsList.OrderByDescending(y => y.VerticesCount).FirstOrDefault(x => x.CpuTableTime.Count != 0);
                    if (tmp != null)
                    {
                        processedVerticesCount = tmp.VerticesCount;
                        times = tmp.CpuTableTime;
                    }
                    break;
                case 2: // CPU Bit
                    tmp = _statsList.OrderByDescending(y => y.VerticesCount).FirstOrDefault(x => x.CpuBitTime.Count!=0);
                    if (tmp != null)
                    {
                        processedVerticesCount = tmp.VerticesCount;
                        times = tmp.CpuBitTime;
                    }
                    break;
            }

            if (times == null || times.Count == 0) return new TimeSpan(0);

            var ticks = times.Sum(t => t.Ticks) / times.Count;
            return new TimeSpan((long)(ticks * Pow(n - processedVerticesCount)));
        }

        /// <summary>
        /// Funckja obliczająca liczbę kombinacji bez powtórzeń liczby n z liczby k.
        /// </summary>
        /// <param name="n">Górna liczba w Dwumianie Newtona.</param>
        /// <param name="k">Dolna liczba w Dwumianie Newtona.</param>
        /// <returns>Liczba kombinacji.</returns>
        static ulong Combination_n_of_k(ulong n, ulong k)
	    {
		    if (k > n) return 0;
		    if (k == 0 || k == n) return 1;

		    if (k * 2 > n) k = n - k;

            ulong r = 1;
            for (ulong d = 1; d <= k; ++d)
            {
			    r *= n--;
			    r /= d;
		    }
		    return r;
	    }

        /// <summary>
        /// Funckja licząca potęgi liczby 2 (zarówno dodatnie jak i ujemne tj. ułamki)
        /// </summary>
        /// <param name="n">Wykładnik potęgi.</param>
        /// <returns>Liczba 2 podniesiona do odpowiedniej potęgi.</returns>
        static double Pow(int n)
        {
            double result = 1;

            if (n > 0)
            {
                while (n > 0)
                {
                    result *= 2;
                    n--;
                }
            }
            else
            {
                while (n < 0)
                {
                    result /= 2;
                    n++;
                }
            }

		    return result;
        }

        #endregion
    }

    /// <summary>
    /// Wewnetrzna klasa przechowująca dane statystyczne dla pojedynczego problemu identyfikowanego na podstawie nazwy pliku wejściowego.
    /// </summary>
    internal class GraphStat
    {
        #region Zmienne

        private long _avgGpuTicks = -1;
        private long _avgCpuTableTicks = -1;
        private long _avgCpuBitTicks = -1;

        private List<int> _memoryGpuUsage;
        private List<int> _memoryCpuTableUsage;
        private List<int> _memoryCpuBitUsage;

        public string FileName { get; set; }
        public int VerticesCount { get; set; }
        public List<TimeSpan> GpuTime { get; set; }
        public List<TimeSpan> CpuTableTime { get; set; }
        public List<TimeSpan> CpuBitTime { get; set; }

        #endregion

        #region Konstruktor
        /// <summary>
        /// Domyślny konstruktor. Inicjalizuje zmienne używane w klasie.
        /// </summary>
        /// <param name="fileName">Nazwa pliku tekstowego zawierającego reprezentację grafu.</param>
        /// <param name="n">Liczba wierzchołków wejścciowego grafu.</param>
        /// <param name="type">Typ algorytmu wywołującego metodę.</param>
        /// <param name="time">Czas obliczeń danego algorytmu.</param>
        /// <param name="memory">Tablica zawierająca pamięć używaną przez aplikację podczas działania algorytmu.</param>
        public GraphStat(string fileName, int n, int type, TimeSpan time, IEnumerable<int> memory)
        {
            FileName = fileName;
            VerticesCount = n;

            GpuTime = new List<TimeSpan>();
            CpuTableTime = new List<TimeSpan>();
            CpuBitTime = new List<TimeSpan>();

            switch (type)
            {
                case 0:
                    GpuTime.Add(time);
                    _memoryGpuUsage = new List<int>(memory);
                    break;
                case 1:
                    CpuTableTime.Add(time);
                    _memoryCpuTableUsage = new List<int>(memory);
                    break;
                case 2:
                    CpuBitTime.Add(time);
                    _memoryCpuBitUsage = new List<int>(memory);
                    break;
            }
        }

        #endregion

        #region Aktualizowanie oraz parsowanie statystyk do tekstu

        /// <summary>
        /// Metoda aktualizująca dane statystyczne konkretnego zadania.
        /// </summary>
        /// <param name="type">Typ algorytmu wywołującego metodę.</param>
        /// <param name="time">Czas obliczeń danego algorytmu.</param>
        /// <param name="memory">Tablica zawierająca pamięć używaną przez aplikację podczas działania algorytmu.</param>
        public void Update(int type, TimeSpan time, IEnumerable<int> memory)
        {
            switch (type)
            {
                case 0:
                    GpuTime.Add(time);
                    _memoryGpuUsage = new List<int>(memory);
                    break;
                case 1:
                    CpuTableTime.Add(time);
                    _memoryCpuTableUsage = new List<int>(memory);
                    break;
                case 2:
                    CpuBitTime.Add(time);
                    _memoryCpuBitUsage = new List<int>(memory);
                    break;
            }
        }

        public string SaveToFile()
        {
            var dnLine = Environment.NewLine + Environment.NewLine;
            var nLine = Environment.NewLine;

            var message = "______________________" + FileName + "______________________" + dnLine;
            message += "Czasy dla obliczeń dla różnych wersji:" + nLine;
            if (GpuTime.Count != 0)
            {
                message += "a) GPU" + nLine + "Czasy pojedyncze:" + nLine;
                var i = 1;
                message = GpuTime.Aggregate(message, (current, t) => current + string.Format("{0}: {1}" + nLine, i++, t));
                message += nLine + "Czas średni: ";
                var ticks = GpuTime.Sum(t => t.Ticks);
                ticks /= GpuTime.Count;
                _avgGpuTicks = ticks;
                message += string.Format("{0}" + nLine, new TimeSpan(ticks));
                message += "Zużycie pamięci RAM na początku algorytmu (włącznie z danymi aplikacji) [MB]: ";
                var first = _memoryGpuUsage.First();
                message += string.Format("{0}" + nLine, (float)first / 1000000);
                message += "Zużycie pamięci RAM w punktach pośrednich algorytmu (tylko algorytm) [MB]: ";
                message = _memoryGpuUsage.Aggregate(message, (current, m) => current + string.Format("{0} ", (float)(m - first) / 1000000));
                message += nLine + "Zużycie pamięci RAM na końcu algorytmu (tylko algorytm) [MB]: ";
                message += string.Format("{0}" + nLine, (float)(_memoryGpuUsage.Last() - first) / 1000000) + nLine;
            }
            if (CpuTableTime.Count != 0)
            {
                message += "b) CPU TABLE" + nLine + "Czasy pojedyncze:" + nLine;
                var i = 1;
                message = CpuTableTime.Aggregate(message, (current, t) => current + string.Format("{0}: {1}" + nLine, i++, t));
                message += nLine + "Czas średni: ";
                var ticks = CpuTableTime.Sum(t => t.Ticks);
                ticks /= CpuTableTime.Count;
                _avgCpuTableTicks = ticks;
                message += string.Format("{0}" + nLine, new TimeSpan(ticks));
                message += "Zużycie pamięci RAM na początku algorytmu (włącznie z danymi aplikacji) [MB]: ";
                var first = _memoryCpuTableUsage.First();
                message += string.Format("{0}" + nLine, (float)first / 1000000);
                message += "Zużycie pamięci RAM w punktach pośrednich algorytmu (tylko algorytm) [MB]: ";
                message = _memoryCpuTableUsage.Aggregate(message, (current, m) => current + string.Format("{0} ", (float)(m - first) / 1000000));
                message += nLine + "Zużycie pamięci RAM na końcu algorytmu (tylko algorytm) [MB]: ";
                message += string.Format("{0}" + nLine, (float)(_memoryCpuTableUsage.Last() - first) / 1000000) + nLine;
            }
            if (CpuBitTime.Count != 0)
            {
                message += "c) CPU BIT" + nLine + "Czasy pojedyncze:" + nLine;
                var i = 1;
                message = CpuBitTime.Aggregate(message, (current, t) => current + string.Format("{0}: {1}" + nLine, i++, t));
                message += nLine + "Czas średni: ";
                var ticks = CpuBitTime.Sum(t => t.Ticks);
                ticks /= CpuBitTime.Count;
                _avgCpuBitTicks = ticks;
                message += string.Format("{0}" + nLine, new TimeSpan(ticks));
                message += "Zużycie pamięci RAM na początku algorytmu (włącznie z danymi aplikacji) [MB]: ";
                var first = _memoryCpuBitUsage.First();
                message += string.Format("{0}" + nLine, (float)first / 1000000);
                message += "Zużycie pamięci RAM w punktach pośrednich algorytmu (tylko algorytm) [MB]: ";
                message = _memoryCpuBitUsage.Aggregate(message, (current, m) => current + string.Format("{0} ", (float)(m - first) / 1000000));
                message += nLine + "Zużycie pamięci RAM na końcu algorytmu (tylko algorytm) [MB]: ";
                message += string.Format("{0}" + nLine, (float)(_memoryCpuBitUsage.Last() - first) / 1000000) + nLine;
            }

            message += "Współczynnik zmiany czasu obliczeń cpu_table/gpu: ";
            if (_avgGpuTicks != -1 && _avgCpuTableTicks != -1)
                message += string.Format("{0}" + nLine, _avgCpuTableTicks / (float)_avgGpuTicks);
            else
                message += "Brak informacji." + nLine;

            message += "Współczynnik zmiany czasu obliczeń cpu_bit/gpu: ";
            if (_avgGpuTicks != -1 && _avgCpuBitTicks != -1)
                message += string.Format("{0}" + nLine, _avgCpuBitTicks / (float)_avgGpuTicks);
            else
                message += "Brak informacji." + nLine;

            message += "Współczynnik zmiany czasu obliczeń cpu_table/cpu_bit: ";
            if (_avgCpuTableTicks != -1 && _avgCpuBitTicks != -1)
                message += string.Format("{0}" + nLine, _avgCpuTableTicks / (float)_avgCpuBitTicks);
            else
                message += "Brak informacji." + nLine;

            message += "######################################################################" + dnLine;

            return message;
        }
        #endregion
    }
}
