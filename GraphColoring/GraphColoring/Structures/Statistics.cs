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

        public void Add(string fileName, int type, TimeSpan time)
        {
            var i = _statsList.FindIndex(x => x.FileName == fileName);
            if (i == -1)
                _statsList.Add(new GraphStat(fileName, type, time));
            else
                _statsList[i].Update(type, time);
        }

        public void SaveToFile(string path)
        {
            var content = _statsList.Aggregate("", (current, g) => current + g.SaveToFile());
            File.WriteAllText(path, content);
        }
    }

    internal class GraphStat
    {
        private long avgGpuTicks = -1;
        private long avgCpuTableTicks = -1;
        private long avgCpuBitTicks = -1;
        public string FileName { get; set; }
        public List<TimeSpan> GpuTime { get; set; }
        public List<TimeSpan> CpuTableTime { get; set; }
        public List<TimeSpan> CpuBitTime { get; set; }

        public GraphStat(string fileName, int type, TimeSpan time)
        {
            FileName = fileName;

            GpuTime = new List<TimeSpan>();
            CpuTableTime = new List<TimeSpan>();
            CpuBitTime = new List<TimeSpan>();

            switch (type)
            {
                case 0:
                    GpuTime.Add(time);
                    break;
                case 1:
                    CpuTableTime.Add(time);
                    break;
                case 2:
                    CpuBitTime.Add(time);
                    break;
            }
        }

        public void Update(int type, TimeSpan time)
        {
            switch (type)
            {
                case 0:
                    GpuTime.Add(time);
                    break;
                case 1:
                    CpuTableTime.Add(time);
                    break;
                case 2:
                    CpuBitTime.Add(time);
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
                avgGpuTicks = ticks;
                message += string.Format("{0}" + nLine, new TimeSpan(ticks));
            }
            if (CpuTableTime.Count != 0)
            {
                message += "b) CPU TABLE" + nLine + "Czasy pojedyncze:" + nLine;
                var i = 1;
                message = CpuTableTime.Aggregate(message, (current, t) => current + string.Format("{0}: {1}" + nLine, i++, t));
                message += nLine + "Czas średni: ";
                var ticks = CpuTableTime.Sum(t => t.Ticks);
                ticks /= CpuTableTime.Count;
                avgCpuTableTicks = ticks;
                message += string.Format("{0}" + nLine, new TimeSpan(ticks));
            }
            if (CpuBitTime.Count != 0)
            {
                message += "c) CPU BIT" + nLine + "Czasy pojedyncze:" + nLine;
                var i = 1;
                message = CpuBitTime.Aggregate(message, (current, t) => current + string.Format("{0}: {1}" + nLine, i++, t));
                message += nLine + "Czas średni: ";
                var ticks = CpuBitTime.Sum(t => t.Ticks);
                ticks /= CpuBitTime.Count;
                avgCpuBitTicks = ticks;
                message += string.Format("{0}" + nLine, new TimeSpan(ticks));
            }

            message += "Współczynnik zmiany czasu obliczeń cpu_table/gpu: ";
            if (avgGpuTicks != -1 && avgCpuTableTicks != -1)
                message += string.Format("{0}" + nLine, (float)((float)avgCpuTableTicks / (float)avgGpuTicks));
            else
                message += "Brak informacji." + nLine;

            message += "Współczynnik zmiany czasu obliczeń cpu_bit/gpu: ";
            if (avgGpuTicks != -1 && avgCpuBitTicks != -1)
                message += string.Format("{0}" + nLine, (float)avgCpuBitTicks / (float)avgGpuTicks);
            else
                message += "Brak informacji." + nLine;

            message += "Współczynnik zmiany czasu obliczeń cpu_table/cpu_bit: ";
            if (avgCpuTableTicks != -1 && avgCpuBitTicks != -1)
                message += string.Format("{0}" + nLine, (float)avgCpuTableTicks / (float)avgCpuBitTicks);
            else
                message += "Brak informacji." + nLine;

            message += "######################################################################" + dnLine;

            return message;
        }
    }
}
