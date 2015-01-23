using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GraphColoring.Structures
{
    class Statistics
    {
        private readonly List<GraphStat> _statsList;
 
        public Statistics()
        {
            _statsList = new List<GraphStat>();
        }

        public void Add(string fileName, int type, TimeSpan time, int[] memoryUsage)
        {
            var i = _statsList.FindIndex(x => x.FileName == fileName);
            if (i == -1)
                _statsList.Add(new GraphStat(fileName, type, time, memoryUsage));
            else
                _statsList[i].Update(type, time, memoryUsage);
        }

        public void SaveToFile(string path)
        {
            var content = _statsList.Aggregate("", (current, g) => current + g.SaveToFile());
            File.WriteAllText(path, content);
        }
    }

    internal class GraphStat
    {
        private long _avgGpuTicks = -1;
        private long _avgCpuTableTicks = -1;
        private long _avgCpuBitTicks = -1;
        private List<int> _memoryGpuUsage;
        private List<int> _memoryCpuTableUsage;
        private List<int> _memoryCpuBitUsage;
        public string FileName { get; set; }
        public List<TimeSpan> GpuTime { get; set; }
        public List<TimeSpan> CpuTableTime { get; set; }
        public List<TimeSpan> CpuBitTime { get; set; }

        public GraphStat(string fileName, int type, TimeSpan time, IEnumerable<int> memory)
        {
            FileName = fileName;

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
                message += string.Format("{0}" + nLine, (float)((float)_avgCpuTableTicks / (float)_avgGpuTicks));
            else
                message += "Brak informacji." + nLine;

            message += "Współczynnik zmiany czasu obliczeń cpu_bit/gpu: ";
            if (_avgGpuTicks != -1 && _avgCpuBitTicks != -1)
                message += string.Format("{0}" + nLine, (float)_avgCpuBitTicks / (float)_avgGpuTicks);
            else
                message += "Brak informacji." + nLine;

            message += "Współczynnik zmiany czasu obliczeń cpu_table/cpu_bit: ";
            if (_avgCpuTableTicks != -1 && _avgCpuBitTicks != -1)
                message += string.Format("{0}" + nLine, (float)_avgCpuTableTicks / (float)_avgCpuBitTicks);
            else
                message += "Brak informacji." + nLine;

            message += "######################################################################" + dnLine;

            return message;
        }
    }
}
