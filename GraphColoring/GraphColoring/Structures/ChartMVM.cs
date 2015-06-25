using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Linq;

namespace GraphColoring.Structures
{
    /// <summary>
    /// Klasa rozszerzająca klasę ObservableCollection. Wykonuje odświeżenie danych po akcjach takich jak Insert oraz Update.
    /// </summary>
    /// <typeparam name="T">Typ obiektów, które zawieta kolekcja</typeparam>
    internal class SmartCollection<T> : ObservableCollection<T>
    {
        /// <summary>
        /// Metoda wstawia do kolekcji w odpowiednim miejscu podany element, a następnie informuje o pewnej zmianie kolekcji.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="item"></param>
        public void SmartInsert(int index, T item)
        {
            if(index < Items.Count)
                Items.Insert(index, item);
            else
                Items.Add(item);

            OnPropertyChanged(new PropertyChangedEventArgs("Count"));
            OnPropertyChanged(new PropertyChangedEventArgs("Item[]"));
            OnCollectionChanged(new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Reset));
        }

        /// <summary>
        /// Metoda uaktualnia element od podanym indeksie podaną wartością, a następnie informuje o pewnej zmianie kolekcji.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="item"></param>
        public void SmartUpdate(int index, T item)
        {
            if(index < Items.Count)
                Items[index] = item;

            OnPropertyChanged(new PropertyChangedEventArgs("Count"));
            OnPropertyChanged(new PropertyChangedEventArgs("Item[]"));
            OnCollectionChanged(new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Reset));
        }
    }

    /// <summary>
    /// Klasa zawierająca model jednego punktu na wykresie zależności wykorzystanej pamięci od iteracji w jakiej znajdują się obliczenia. Liczba iteracji to liczba wierzchołków + 1
    /// </summary>
    internal class SpaceModel
    {
        /// <summary>
        /// Współrzędna składowa osi X wykresu - reprezentuje iterację w jakiej znajduje się obliczenie
        /// </summary>
        public int X { get; set; }

        /// <summary>
        /// Współrzędna składowa osi Y wykresu - reprezentuje wykorzystaną pamięć RAM w danej iteracji obliczeń
        /// </summary>
        public double Y { get; set; }

        /// <summary>
        /// Domyślny konstruktor. Współrzędną X zaokrągla do maksymalnie 3 miejsc po przecinku.
        /// </summary>
        /// <param name="x">Iteracja w jakiej znajduje się obliczenie.</param>
        /// <param name="y">Wartość użytej pamięci na potrzeby obliczeń algorytmu w danej chwili.</param>
        public SpaceModel(int x, double y)
        {
            X = x;
            Y = Math.Round(y, 5);
        }
    }

    /// <summary>
    /// Klasa zawierająca model jednego punktu na wykresie zależności czasu obliczenia od liczby wierzchołków grafu wejściowego.
    /// </summary>
    internal class TimeModel
    {
        /// <summary>
        /// Współrzędna składowa osi X wykresu - reprezentuje rozmiar zadania (liczba wierzchołków)
        /// </summary>
        public int X { get; set; }

        /// <summary>
        /// Współrzędna składowa osi Y wykresu - reprezentuje czas obliczeń danego zadania
        /// </summary>
        public double Y { get; set; }

        /// <summary>
        /// Domyślny konstruktor. 
        /// </summary>
        /// <param name="x">Rozmiar zadania (liczba wierzchołków).</param>
        /// <param name="y">Czas potrzebny na obliczenie danego zadania.</param>
        public TimeModel(int x, double y)
        {
            X = x;
            Y = y;
        }
    }

    /// <summary>
    /// Klasa łącząca model danych z wykresem. Odpowiada za aktualizację wykresu na podstawie przekazanych danych.
    /// </summary>
    internal class ViewModel
    {
        #region Kolekcje reprezentujące konkretne linie na wykresie - wykres dla statystyk pamięci
        /// <summary>
        /// Zmienna reprezentująca zależność wykorzystanej pamięci od iteracji obliczenia dla algorytmu GPU.
        /// </summary>
        public SmartCollection<SpaceModel> CollectionGpu { get; set; }

        /// <summary>
        /// Zmienna reprezentująca zależność wykorzystanej pamięci od iteracji obliczenia dla algorytmu CPU w wersji tablicowej.
        /// </summary>
        public SmartCollection<SpaceModel> CollectionCput { get; set; }

        /// <summary>
        /// Zmienna reprezentująca zależność wykorzystanej pamięci od iteracji obliczenia dla algorytmu CPU w wersji bitowej.
        /// </summary>
        public SmartCollection<SpaceModel> CollectionCpub { get; set; }

        /// <summary>
        /// Zmienna reprezentująca przewidywaną zależność wykorzystanej pamięci od iteracji obliczenia dla algorytmu GPU.
        /// </summary>
        public SmartCollection<SpaceModel> CollectionGpuPredict { get; set; }

        /// <summary>
        /// Zmienna reprezentująca przewidywaną zależność wykorzystanej pamięci od iteracji obliczenia dla algorytmu CPU w wersji tablicowej.
        /// </summary>
        public SmartCollection<SpaceModel> CollectionCputPredict { get; set; }

        /// <summary>
        /// Zmienna reprezentująca przewidywaną zależność wykorzystanej pamięci od iteracji obliczenia dla algorytmu CPU w wersji bitowej.
        /// </summary>
        public SmartCollection<SpaceModel> CollectionCpubPredict { get; set; }

        #endregion

        #region Kolekcje reprezentujące konkretne linie na wykresie - wykres dla statystyk czasu
        /// <summary>
        /// Kolekcja reprezentująca zależność czasu obliczenia od rozmiaru zadania dla algorytmu GPU.
        /// </summary>
        public SmartCollection<TimeModel> TimeGpu { get; set; }

        /// <summary>
        /// Kolekcja reprezentująca zależność średniego czasu obliczenia od rozmiaru zadania dla algorytmu GPU.
        /// </summary>
        public SmartCollection<TimeModel> TimeGpuAvg { get; set; }

        /// <summary>
        /// Kolekcja reprezentująca zależność czasu obliczenia od rozmiaru zadania dla algorytmu CPU w wersji tablicowej.
        /// </summary>
        public SmartCollection<TimeModel> TimeCput { get; set; }

        /// <summary>
        /// Kolekcja reprezentująca zależność średniego czasu obliczenia od rozmiaru zadania dla algorytmu CPU w wersji tablicowej.
        /// </summary>
        public SmartCollection<TimeModel> TimeCputAvg { get; set; }

        /// <summary>
        /// Kolekcja reprezentująca zależność czasu obliczenia od rozmiaru zadania dla algorytmu CPU w wersji bitowej.
        /// </summary>
        public SmartCollection<TimeModel> TimeCpub { get; set; }

        /// <summary>
        /// Kolekcja reprezentująca zależność średniego czasu obliczenia od rozmiaru zadania dla algorytmu CPU w wersji bitowej.
        /// </summary>
        public SmartCollection<TimeModel> TimeCpubAvg { get; set; }
        #endregion

        #region Konstruktor
        /// <summary>
        /// Domyślny konstruktor. Inicjalizuje wszystkie zmienne.
        /// </summary>
        public ViewModel()
        {
            CollectionGpu = new SmartCollection<SpaceModel>();
            CollectionCput = new SmartCollection<SpaceModel>();
            CollectionCpub = new SmartCollection<SpaceModel>();

            CollectionGpuPredict = new SmartCollection<SpaceModel>();
            CollectionCputPredict = new SmartCollection<SpaceModel>();
            CollectionCpubPredict = new SmartCollection<SpaceModel>();

            TimeGpu = new SmartCollection<TimeModel>();
            TimeCput = new SmartCollection<TimeModel>();
            TimeCpub = new SmartCollection<TimeModel>();

            TimeGpuAvg = new SmartCollection<TimeModel>();
            TimeCputAvg = new SmartCollection<TimeModel>();
            TimeCpubAvg = new SmartCollection<TimeModel>();

            TimeGpuAvg.Add(new TimeModel(0, 0));
            TimeCputAvg.Add(new TimeModel(0, 0));
            TimeCpubAvg.Add(new TimeModel(0, 0));
        }
        #endregion

        #region Metody główne zarządzające danymi na wykresie

        /// <summary>
        /// Metoda dodająca dane do zmiennych przechowujących informację liniach wykresu.
        /// </summary>
        /// <param name="pamiec">Tablica przechowująca informację o użytej pamięci w poszczególnych iteracjach algorytmu.</param>
        /// <param name="timeElapsed"></param>
        /// <param name="type">Typ algorytmu wywołującego daną metodę.</param>
        public void AddData(double[] pamiec, long timeElapsed, int type)
        {
            if (pamiec == null || pamiec.Length == 0) return;

            var n = pamiec.Length - 3;
            var predictSpace = PredictSpace((ulong)n, type).ToArray();

            var first = pamiec[0];

            for (var i = 0; i < n + 3; ++i)
            {
                var pamiecYtmp = pamiec[i] - first;
                var pamiecY = pamiecYtmp < 0 ? 0 : pamiecYtmp;

                var stdModel = new SpaceModel(i, pamiecY);
                var prdModel = new SpaceModel(i, predictSpace[i]);

                switch (type)
                {
                    case 0:
                        CollectionGpu.Add(stdModel);
                        CollectionGpuPredict.Add(prdModel);
                        break;
                    case 1:
                        CollectionCput.Add(stdModel);
                        CollectionCputPredict.Add(prdModel);
                        break;
                    case 2:
                        CollectionCpub.Add(stdModel);
                        CollectionCpubPredict.Add(prdModel);
                        break;
                }
            }

            switch (type)
            {
                case 0:
                    AddTime(TimeGpu, TimeGpuAvg, n, timeElapsed);
                    break;
                case 1:
                    AddTime(TimeCput, TimeCputAvg, n, timeElapsed);
                    break;
                case 2:
                    AddTime(TimeCpub, TimeCpubAvg, n, timeElapsed);
                    break;
            }
        }

        /// <summary>
        /// Czyszczenie wykresów.
        /// </summary>
        public void ClearChart()
        {
            CollectionGpu.Clear();
            CollectionCput.Clear();
            CollectionCpub.Clear();

            CollectionGpuPredict.Clear();
            CollectionCputPredict.Clear();
            CollectionCpubPredict.Clear();
        }
        #endregion

        #region Metody pomocnicze (statyczne)

        /// <summary>
        /// Metoda dodająca dane statystyczne do kolekcji przechowujących informacje o czasie wykonania algorytmu.
        /// </summary>
        /// <param name="colTime">Kolekcja z czasami wykonania algorytmu.</param>
        /// <param name="colTimeAvg">Kolekcja z czasem średnim wyliczonym na podstawie kolekcji z czasami poszczegołnymi.</param>
        /// <param name="n">Liczba wierzchołków w grafie.</param>
        /// <param name="time">Czas obliczenia pojedynczego zadania.</param>
        private static void AddTime(ICollection<TimeModel> colTime, SmartCollection<TimeModel> colTimeAvg, int n,
           long time)
        {
            colTime.Add(new TimeModel(n, time));
            var avgTime = colTime.Where(x => x.X == n).Average(y => y.Y);
            avgTime = avgTime < 0 ? 0 : avgTime;
            var index = FindIndex(n, colTimeAvg);

            if (index < 0)
                colTimeAvg.SmartUpdate(-index, new TimeModel(n, avgTime));
            else
                colTimeAvg.SmartInsert(index, new TimeModel(n, avgTime));
        }

        /// <summary>
        /// Metoda znajduje indeks wskazujący na miejsce, w którym ma być wstawiony nowy punkt na wykreesie czasu.
        /// </summary>
        /// <param name="n">Rozmiar zadania (liczba wierzchołków).</param>
        /// <param name="collection">Kolekcja zawierająca średnie czasy dla konkretnej metody algorytmu.</param>
        /// <returns>Indeks miejsca, w którym należy wstawić nowy wpis.</returns>
        private static int FindIndex(int n, IReadOnlyCollection<TimeModel> collection)
        {
            for (var i = 0; i < collection.Count; ++i)
            {
                if (i != collection.Count - 1 && collection.ElementAt(i).X < n && collection.ElementAt(i + 1).X > n)
                    return i + 1;
                if (collection.ElementAt(i).X == n)
                    return -i;
            }

            return collection.Count;
        }

        /// <summary>
        /// Metoda obliczająca przewidywane wykorzystanie pamięci dla każdej iteracji algorytmu w zależności od wybranej metody.
        /// </summary>
        /// <param name="n">Liczba wierzchołków grafu wejściowego.</param>
        /// <param name="type">Typ metody algorytmu wywołującej daną metodę.</param>
        /// <returns>Tablica przechowująca informację o przewidywanym użyciu pamięci w poszczególnych iteracjach algorytmu.</returns>
        private static IEnumerable<double> PredictSpace(ulong n, int type)
        {
            var powerNumber = (1 << (int)n) / Common.ToMb;
            var tmp = new double[n + 3];
            tmp[0] = tmp[n + 2] = 0;
            tmp[n] = tmp[n + 1] = powerNumber;

            for (ulong i = 1; i < n; ++i)
            {
                var actualVertices = Common.Combination_n_of_k(n, i);
                var newVertices = Common.Combination_n_of_k(n, i + 1);

                switch (type)
                {
                    case 0: // GPU
                        tmp[i] = powerNumber + (actualVertices * i + newVertices * (i + 1) + actualVertices) / Common.ToMb;
                        break;
                    case 1: // CPU Table
                        tmp[i] = powerNumber + (actualVertices * i + newVertices * (i + 1)) / Common.ToMb;
                        break;
                    case 2: // CPU Bit
                        tmp[i] = powerNumber + (actualVertices + newVertices) / Common.ToMb;
                        break;
                }
            }

            return tmp;
        }
        #endregion
    }
}
