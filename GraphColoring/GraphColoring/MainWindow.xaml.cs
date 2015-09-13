﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Resources;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Documents;
using System.Windows.Forms;
using System.Windows.Media;
using GraphColoring.Resources.Strings;
using GraphColoring.Structures;
using MahApps.Metro.Controls;
using MahApps.Metro.Controls.Dialogs;
using Microsoft.Win32;
using Application = System.Windows.Application;
using MouseEventArgs = System.Windows.Input.MouseEventArgs;
using OpenFileDialog = System.Windows.Forms.OpenFileDialog;
//using System.Windows.Media.Imaging;

[assembly: NeutralResourcesLanguage("pl-PL")]
namespace GraphColoring
{

    /// <summary>
    /// Klasa obsługująca UI i zdarzenia w nim występujące.
    /// </summary>
    public partial class MainWindow
    {
        // 2 TODO wykres - bug/error

        #region Definicje metod z załączonych plików DDL
        /// <summary>
        /// Zewnętrzna funkcja ustawiająca ścieżkę do folderu z plikami DLL używanymi w aplikacji.
        /// </summary>
        /// <param name="lpPathName">Ścieżka do folderu z plikami DLL.</param>
        /// <returns>Wartość bool informująca o powodzeniu akcji.</returns>
        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        static extern bool SetDllDirectory(string lpPathName);

        /// <summary>
        /// Funkcja wywołująca algorytm kolorowania grafu sekwencyjnie (z użyciem procesora CPU). Implementacja algorytmu zawarta oraz dostępna z pliku DLL.
        /// </summary>
        /// <param name="pamiec">Tablica przechowująca informacje o statystykach wykorzystania pamięci podczas działania algorytmu.</param>
        /// <param name="vertices">Tablica przechowująca reprezentację grafu.</param>
        /// <param name="neighborsCount">Tablica przechowująca informację o sąsiadach każdego wierzchołka.</param>
        /// <param name="n">Liczba wierzchołków grafu wejściowego.</param>
        /// <param name="flag">Flaga informująca, jaką modfikację algorytmu wywołujemy: 0 - tablicowa (domyślnie), 1 - bitowa.</param>
        /// <returns>Liczba k oznaczająca, że dany graf jest nie więcej niż k-kolorowalny.</returns>
        [DllImport("GraphColoringCPU.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int FindChromaticNumber([MarshalAs(UnmanagedType.LPArray)]double[] pamiec,[MarshalAs(UnmanagedType.LPArray)]int[] vertices, [MarshalAs(UnmanagedType.LPArray)]int[] neighborsCount, int n, int flag = 0);

        /// <summary>
        /// Funkcja wywołująca algorytm kolorowania grafu równolegle (z użyciem procesora GPU). Implementacja algorytmu zawarta oraz dostępna z pliku DLL.
        /// </summary>
        /// <param name="wynik">Wynik obliczeń zawarty w tablicy. Z niej wyciąga się końcowy wynik oznaczający kolorowalność grafu.</param>
        /// <param name="pamiec">Tablica przechowująca informacje o statystykach wykorzystania pamięci podczas działania algorytmu.</param>
        /// <param name="vertices">Tablica przechowująca reprezentację grafu.</param>
        /// <param name="neighborsCount">Tablica przechowująca informację o sąsiadach każdego wierzchołka.</param>
        /// <param name="n">Liczba wierzchołków grafu wejściowego.</param>
        /// <param name="allVertices">Liczba wszystkich sąsiadów wszystkich wierzchołków.</param>
        /// <returns>Liczba k oznaczająca, że dany graf jest nie więcej niż k-kolorowalny.</returns>
        [DllImport("GraphColoringGPU.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int FindChromaticNumberGPU([MarshalAs(UnmanagedType.LPArray)]int[] wynik, [MarshalAs(UnmanagedType.LPArray)]double[] pamiec, [MarshalAs(UnmanagedType.LPArray)]int[] vertices, [MarshalAs(UnmanagedType.LPArray)]int[] neighborsCount, int n, int allVertices);
        
        #endregion

        #region Stałe

        private const string Output = "\\Output\\";
        private const string Dll = "\\DLL\\"; //"..\\..\\..\\Debug\\";
        private const string Doc = "\\Doc\\";
        private const string DocFile = "\\Doc\\Help.pdf";
        private const string NavigateBlank = "about:blank";
        private const string Log = "Log_";
        private const string Stats = "Stats_";
        private const string Type = ".txt";
        private const string Markup = "yyyy-MM-dd HH:mm:ss > ";
        private const string EndLine = "\r";
        private const string PatternStar = "*";
        private const string NotPressed = "NotPressed";
        private const string Pressed = "Pressed";
        private const string CheckboxPressed = "#CC119EDA";
        //private const string CulturePl = "pl-PL";
        //private const string CultureEn = "en-US";
        //private const string ImagePl = "pack://siteoforigin:,,,/Resources/Images/pl.jpg";
        //private const string ImageEn = "pack://siteoforigin:,,,/Resources/Images/en.jpg";
        private static readonly Color TileMouseOn = Colors.DimGray;
        private readonly Color _cNormal = Colors.DimGray;
        private readonly Color _cPath = Colors.DodgerBlue;
        private readonly Color _cResult = Colors.SeaGreen;
        private readonly Color _cError = Colors.Red;
        private readonly Color _cWarning = Colors.Orange;

        #endregion

        #region Zmienne

        /// <summary>
        /// Zmienna przechowująca informację, czy wybrana przez użytkownika ścieżka jest ścieżką do pliku, czy do folderu. Wartość true - plik, false - folder.
        /// </summary>
        public bool IsFile { get; private set; }

        /// <summary>
        /// Zmienna przechowująca ścieżkę do pliku lub folderu z plikami zawierającymi reprezentację grafu. 
        /// </summary>
        public string GraphPath { get; private set; }

        /// <summary>
        /// Zmienna przechowująca ścieżkę do pliku z logiem, domyślnie plik z logiem znajduje się w folderze z projektem. Ścieżka względna: "/Output/Log_{DATA}.txt", gdzie parametr {DATA} to aktualna data i czas w domyślnym formacie TIMESTAMP systemu WINDOWS.
        /// </summary>
        public string LogFile { get; private set; }

        /// <summary>
        /// Zmienna przechowująca ścieżkę do pliku ze statystykami, domyślnie plik ze statystykami znajduje się w folderze z projektem. Ścieżka względna: "/Output/Statistics_{DATA}.txt", gdzie parametr {DATA} to aktualna data i czas w domyślnym formacie TIMESTAMP systemu WINDOWS.
        /// </summary>
        public string StatsFile { get; private set; }

        /// <summary>
        /// Zmienna przechowująca element wspólny nazwy plików w przypadku wybrania ścieżki do folderu z plikami. W przypadku podania pustej wartości wybierane są wszystkie pliki z wybranego folderu. 
        /// </summary>
        public string SearchPattern { get; private set; }

        /// <summary>
        /// Zmienna przechowująca ścieżkę do folderu z plikami DLL, w których zawarta jest implementacja algorytmu. 
        /// </summary>
        public string DllFolder { get; private set; }

        /// <summary>
        /// Zmienna przechowująca ścieżkę do pliku z instrukcją użytkownika.
        /// </summary>
        public string HelpDocPath { get; private set; }

        private readonly Bindings _bindings;
        private readonly ViewModel _viewModel;
        private Action _cancelWork;
        private readonly Statistics _stats;
        private readonly FlowDocument _rtbContents;

        #endregion

        #region Konstruktor okna głównego

        /// <summary>
        /// Konstruktor okna głównego. Inicjalizuje dodatkowe zmienne użyte w aplikacji.
        /// </summary>
        public MainWindow()
        {
            InitializeComponent();

            _bindings = new Bindings { EnabledValue = true, VisibilityValue = Visibility.Hidden, StopButtonVisibilityValue = Visibility.Hidden };

            DataContext = _bindings;

            _stats = new Statistics();
            _viewModel = new ViewModel();

            Chart.DataContext = new
            {
                Data = _viewModel,
                View = _bindings
            };

            var directory = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory()));

            DllFolder = directory + Dll; 
            SetDllDirectory(DllFolder);

            HelpDocPath = directory + DocFile;

            CreateFoldersIfNotExists();

            LogFile = Path.Combine(directory + Output, String.Format("{0}{1:yyyy-MM-dd hh.mm.ss}{2}", Log, DateTime.Now, Type));
            StatsFile = Path.Combine(directory + Output, String.Format("{0}{1:yyyy-MM-dd hh.mm.ss}{2}", Stats, DateTime.Now, Type));

            ContentPanel.Document = new FlowDocument();
            _rtbContents = ContentPanel.Document;
        }

        /// <summary>
        /// Funkcja przygotowuje środowisko pod kątem istnienia folderów.
        /// </summary>
        private void CreateFoldersIfNotExists()
        {
            var directory = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory()));

            Directory.CreateDirectory(DllFolder);
            Directory.CreateDirectory(directory + Doc);
            Directory.CreateDirectory(directory + Output);
        }

        #endregion

        #region Logowanie wiadomości

        /// <summary>
        /// Funkcja ogólna rozpoczynająca zapis wiadomości do konsoli oraz pliku z logiem.
        /// </summary>
        /// <param name="messages">Tablica przechowująca informacje (stringi) do zapisania.</param>
        /// <param name="types">Tablica przechowująca informacje o kolorach wiadomości podanych w pierwszym parametrze, jakie pojawią się w konsoli (domyślnie wszystkie ciemnoszare).</param>
        private void WriteMessage(IReadOnlyList<string> messages, IReadOnlyList<Color> types = null)
        {
            if (!string.IsNullOrEmpty(LogFile))
                AppendToFile(messages.Aggregate((i, j) => i + j));
            
            AddToOutput(messages, types);
        }

        /// <summary>
        /// Funkcja ogólna rozpoczynająca zapis wiadomości do konsoli oraz pliku z logiem (wersja dla obliczeń w tle).
        /// </summary>
        /// <param name="messages">Tablica przechowująca informacje (stringi) do zapisania.</param>
        /// <param name="types">Tablica przechowująca informacje o kolorach wiadomości podanych w pierwszym parametrze, jakie pojawią się w konsoli (domyślnie wszystkie ciemnoszare).</param>
        private void WriteMessageUi(IReadOnlyList<string> messages, IReadOnlyList<Color> types = null)
        {
            Application.Current.Dispatcher.Invoke(() => {
                WriteMessage(messages, types);
            });
        }

        /// <summary>
        /// Funkcja zapisująca wiadomości do pliku. Jeśli plik nie istnieje, zostaje automatycznie utworzony w domyślnej lokalizacji.
        /// </summary>
        /// <param name="message">Wiadomość w postaci string, która zostanie zapisana do pliku tekstowego.</param>
        private void AppendToFile(string message)
        {
            var exist = File.Exists(LogFile);

            using (var sw = new StreamWriter(LogFile, true))
            {
                if (!exist)
                {
                    sw.WriteLine(DateTime.Now.ToString(Markup) + Messages.LogFileNotExist + LogFile);
                    AddToOutput(new[] {Messages.LogFileNotExist, LogFile}, new[] {_cNormal, _cPath});
                }
                    
                sw.WriteLine(DateTime.Now.ToString(Markup) + message);
            }
        }

        /// <summary>
        /// Funkcja zapisująca wiadomości do konsoli aplikacji.
        /// </summary>
        /// <param name="messages">Tablica przechowująca informacje (stringi) do zapisania.</param>
        /// <param name="types">Tablica przechowująca informacje o kolorach wiadomości podanych w pierwszym parametrze, jakie pojawią się w konsoli (domyślnie wszystkie ciemnoszare).</param>
        private void AddToOutput(IReadOnlyList<string> messages, IReadOnlyList<Color> types =  null)
        {
            var paraMarkup = new Paragraph { Margin = new Thickness(0) };

            paraMarkup.Inlines.Add(new Bold(new Run(DateTime.Now.ToString(Markup))));

            for (var i = 0; i < messages.Count; ++i)
            {
                paraMarkup.Inlines.Add(types == null
                    ? new Run(messages[i]) {Foreground = new SolidColorBrush(_cNormal)}
                    : new Run(messages[i]) {Foreground = new SolidColorBrush(types[i])});
            }

            _rtbContents.Blocks.Add(paraMarkup);
       
            ContentPanel.Focus();
            ContentPanel.CaretPosition = ContentPanel.Document.ContentEnd;
            ContentPanel.ScrollToEnd();
        }

        #endregion

        // 0 TODO dodanie blokowania uruchomienia (tylko dla GPU)
        #region Funkcje uruchamiające obliczanie algorytmu różnymi metodami

        /// <summary>
        /// Funkcja pomocniczna wyświetlająca w konsoli przewidywany czas obliczeń dla wybranych ustawień.
        /// </summary>
        /// <param name="g">Graf wejściowy.</param>
        /// <param name="type">Typ algorytmu.</param>
        private void PredictTime(Graph g, int type)
        {
            var predictedTime = _stats.PredictTime(g.VerticesCount, type);
            if (predictedTime.Ticks == 0)
                WriteMessageUi(new[] { Messages.GraphRunPredictTimeNoData }, new[] { _cWarning });
            else
            {
                var output = predictedTime.TotalMilliseconds < 1000 ? 
                    Messages.PredictTimeLessThanSecond : 
                    string.Format("{0:##.###}s", predictedTime.TotalMilliseconds / 1000 * g.Density);

                WriteMessageUi(new[] { Messages.GraphRunPredictTimeStart, output }, new[] { _cNormal, _cPath });
            }

            switch (type)
            {
                case 0:
                    WriteMessageUi(new[] { Messages.PredictTimeMethodInfo, Messages.GraphGPU }, new[] { _cNormal, _cPath });
                    break;
                case 1:
                    WriteMessageUi(new[] { Messages.PredictTimeMethodInfo, Messages.GraphCPUT }, new[] { _cNormal, _cPath });
                    break;
                case 2:
                    WriteMessageUi(new[] { Messages.PredictTimeMethodInfo, Messages.GraphCPUB }, new[] { _cNormal, _cPath });
                    break;
            }
        }

        /// <summary>
        /// Funkcja pomocniczna wyświetlająca komunikat w przypadku braku dostępnej pamięci potrzebnej do wykonania obliczeń bieżącego zadania.
        /// </summary>
        /// <param name="g">Graf wejściowy.</param>
        /// <param name="type">Typ algorytmu.</param>
        /// <returns>False - brak dostępnej pamięci, true - wszystko w porządku.</returns>
        private bool PredictSpace(Graph g, int type)
        {
            var predictedSpace = _stats.PredictSpace(g.VerticesCount, type);
            if (predictedSpace == null) return false;

            WriteMessageUi(new[] { Messages.GraphRunErrorNoSpaceAvailable, predictedSpace[0].ToString(CultureInfo.InvariantCulture), Messages.GraphRunErrorNoSpaceRequired, predictedSpace[1].ToString(CultureInfo.InvariantCulture), Messages.GraphRunErrorNoSpaceEndLine }, new[] { _cNormal, _cError, _cNormal, _cError, _cNormal });

            return true;
        }

        /// <summary>
        /// Funkcja pomocnicza wykonująca przygotowanie danych oraz wyświetlająca wstępne informacje w konsoli.
        /// </summary>
        /// <param name="path">Ścieżka do pliku z reprezentacją grafu.</param>
        /// <returns>Wynikowy graf, bądź NULL w przypadku błędu.</returns>
        private Graph Preprocessing(string path)
        {
            var g = FileProcessing.ReadFile(path);
            if (g == null || g.VerticesCount == 0)
            {
                WriteMessage(new[] {Messages.GraphRunErrorReadingInput, path}, new[] {_cError, _cPath});
                return null;
            }

            WriteMessage(new[] 
            { 
                Messages.PreprocessingFirst, path, 
                Messages.PreprocessingSecond, g.VerticesCount.ToString(), 
                Messages.PreprocessingThird, string.Format("{0:##}%", g.Density * 100)
            }, new[] { _cNormal, _cPath, _cNormal, _cPath, _cNormal, _cPath, _cNormal });

            return g;
        }

        /// <summary>
        /// Funkcja pomocnicza sprawdzająca istnienie pliku DLL dla odpowiedniej metody algorytmu.
        /// </summary>
        /// <param name="type">Typ metody algorytmu.</param>
        /// <returns>True - plik istnieje, false - plik nie istnieje.</returns>
        private bool CheckDll(int type)
        {
            var dllFile = type == 0 ? "\\GraphColoringGPU.dll" : "\\GraphColoringCPU.dll";

            if(File.Exists(DllFolder + dllFile)) return false;

            WriteMessageUi(new[] { Messages.DllFileMissing, DllFolder + dllFile }, new[] { _cError, _cPath });

            return true;
        }

        /// <summary>
        /// Metoda uruchamiająca algorytm w wersji równoległej na procesorze GPU. Obliczenia wykonywane są w tle. Wykorzystuje bibliotekę GraphColoringGPU.dll
        /// </summary>
        /// <param name="g">Wczytany graf wejściowy.</param>
        /// <param name="path">Ścieżka do pliku tekstowego zawierającego reprezentację grafu.</param>
        /// <param name="token">Token wykorzystywany do anulowania rozpoczętych w tle obliczeń.</param>
        private void MethodGpu(Graph g, string path, CancellationToken token)
        {
            try
            {
                const int type = 0;

                if (CheckDll(type)) return;

                PredictTime(g, type);
                    
                var wynik = new int[g.VerticesCount];
                var pamiec = new double[g.VerticesCount + 3];

                var watch = Stopwatch.StartNew();

                token.ThrowIfCancellationRequested();
                FindChromaticNumberGPU(wynik, pamiec, g.Vertices, g.NeighboursCount, g.VerticesCount, g.AllVerticesCount);
                token.ThrowIfCancellationRequested();

                if (pamiec[0] == -666)
                {
                    watch.Stop();
                    WriteMessageUi(new[] { Messages.GraphRunErrorNoSpaceAvailable, pamiec[2].ToString(CultureInfo.InvariantCulture), Messages.GraphRunErrorNoSpaceRequired, pamiec[1].ToString(CultureInfo.InvariantCulture), Messages.GraphRunErrorNoSpaceEndLine }, new[] { _cNormal, _cError, _cNormal, _cError, _cNormal });
                    return;
                }

                var wynikk = -2;

                for (var i = 0; i < g.VerticesCount; i++)
                {
                    if (wynik[i] == -1 || wynik[i] == 0) continue;
                    wynikk = wynik[i] + 1;
                    break;
                }

                watch.Stop();

                _stats.Add(path, g.VerticesCount, g.Density, type, watch.Elapsed, pamiec);

                Application.Current.Dispatcher.Invoke(() => _viewModel.AddData(pamiec, watch.ElapsedMilliseconds, type));

                WriteMessageUi(new[] { Messages.GraphRunResultPartOne, wynikk.ToString(), Messages.GraphRunResultPartTwo, watch.Elapsed.ToString() }, new[] { _cNormal, _cResult, _cNormal, _cResult });
            }
            catch (Exception e)
            {
                WriteMessageUi(new[] { Messages.GraphGPU + Messages.GraphRunErrorPartOne, path, Messages.GraphRunErrorPartTwo, e.Message, Messages.GraphRunErrorPartThree }, new[] { _cNormal, _cError, _cNormal, _cError, _cWarning });
            }
        }

        /// <summary>
        /// Metoda uruchamiająca algorytm w wersji synchronicznej na procesorze CPU (wersja tablicowa). Obliczenia wykonywane są w tle. Wykorzystuje bibliotekę GraphColoringGPU.dll
        /// </summary>
        /// <param name="g">Wczytany graf wejściowy.</param>
        /// <param name="path">Ścieżka do pliku tekstowego zawierającego reprezentację grafu.</param>
        /// <param name="token">Token wykorzystywany do anulowania rozpoczętych w tle obliczeń.</param>
        private void MethodTableCpu(Graph g, string path, CancellationToken token)
        {
            try
            {
                const int type = 1;

                if (CheckDll(type)) return;
                if (PredictSpace(g, type)) return;

                PredictTime(g, type);

                var pamiec = new double[g.VerticesCount + 3];

                var watch = Stopwatch.StartNew();

                token.ThrowIfCancellationRequested();
                var k = FindChromaticNumber(pamiec, g.Vertices, g.NeighboursCount, g.VerticesCount);
                token.ThrowIfCancellationRequested();

                watch.Stop();

                _stats.Add(path, g.VerticesCount, g.Density, type, watch.Elapsed, pamiec);

                Application.Current.Dispatcher.Invoke(() => _viewModel.AddData(pamiec, watch.ElapsedMilliseconds, type));

                WriteMessageUi(new[] { Messages.GraphRunResultPartOne, k.ToString(), Messages.GraphRunResultPartTwo, watch.Elapsed.ToString() }, new[] { _cNormal, _cResult, _cNormal, _cResult });
            }
            catch (Exception e)
            {
                WriteMessageUi(new[] { Messages.GraphCPUT + Messages.GraphRunErrorPartOne, path, Messages.GraphRunErrorPartTwo, e.Message }, new[] { _cNormal, _cError, _cNormal, _cError });
            }
            
        }

        /// <summary>
        /// Metoda uruchamiająca algorytm w wersji synchronicznej na procesorze CPU (wersja bitowa). Obliczenia wykonywane są w tle. Wykorzystuje bibliotekę GraphColoringGPU.dll
        /// </summary>
        /// <param name="g">Wczytany graf wejściowy.</param>
        /// <param name="path">Ścieżka do pliku tekstowego zawierającego reprezentację grafu.</param>
        /// <param name="token">Token wykorzystywany do anulowania rozpoczętych w tle obliczeń.</param>
        private void MethodBitCpu(Graph g, string path, CancellationToken token)
        {
            try
            {
                const int type = 2;

                if (CheckDll(type)) return;
                if (PredictSpace(g, type)) return;

                PredictTime(g, type);

                g = FileProcessing.ConvertToBitVersion(g);

                var pamiec = new double[g.VerticesCount + 3];

                var watch = Stopwatch.StartNew();

                token.ThrowIfCancellationRequested();
                var k = FindChromaticNumber(pamiec, g.Vertices, g.NeighboursCount, g.VerticesCount, 1);
                token.ThrowIfCancellationRequested();

                watch.Stop();

                _stats.Add(path, g.VerticesCount, g.Density, type, watch.Elapsed, pamiec);

                Application.Current.Dispatcher.Invoke(() => _viewModel.AddData(pamiec, watch.ElapsedMilliseconds, type));
           
                WriteMessageUi(new[] { Messages.GraphRunResultPartOne, k.ToString(), Messages.GraphRunResultPartTwo, watch.Elapsed.ToString() }, new[] { _cNormal, _cResult, _cNormal, _cResult });
    
            }
            catch (Exception e)
            {
                WriteMessageUi(new[] { Messages.GraphCPUB + Messages.GraphRunErrorPartOne, path, Messages.GraphRunErrorPartTwo, e.Message }, new[] { _cNormal, _cError, _cNormal, _cError });
            }
        }

        #endregion

        #region Funkcje dodatkowe odpowiadające przyciskom z menu głównego

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Plik". Otwiera okno dialogowe wyboru pliku, a następnie zapisuje wybrany plik do zmiennej "GraphPath". Wartość zmiennej "IsFile" ustawia na "true"
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private void ChooseGraphFile_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = Messages.DialogChooseGraphFilter,
                Title = Messages.DialogChooseGraphTitle
            };

            if (openFileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                IsFile = true;
                GraphPath = openFileDialog.InitialDirectory + openFileDialog.FileName;

                WriteMessage(new[] { Messages.DialogChooseGraphFileResult, GraphPath }, new[] { _cNormal, _cPath });
            }
            else
            {
                WriteMessage(new[] { Messages.DialogChooseGraphNoFile }, new[] { _cError });
            }
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Folder". Otwiera okno dialogowe wyboru folderu, a następnie zapisuje wybrany folder do zmiennej "GraphPath". Wartość zmiennej "IsFile" ustawia na "false"
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private void ChooseGraphFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new FolderBrowserDialog();

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                IsFile = false;
                GraphPath = dialog.SelectedPath;

                WriteMessage(new[] { Messages.DialogChooseGraphFolderResult, GraphPath }, new[] { _cNormal, _cPath });
            }
            else
            {
                WriteMessage(new[] { Messages.DialogChooseGraphNoFolder }, new[] { _cError });
            }
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Wybierz wzorzec nazwy plików w folderze". Otwiera okno dialogowe wprowadzania tekstu, a następnie zapisuje podany tekst do zmiennej "SearchPattern"
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private async void SetPattern_Click(object sender, RoutedEventArgs e)
        {
            var dialog = await this.ShowInputAsync(Messages.SetPatternTitle, Messages.SetPatternDesc);
            if (!string.IsNullOrEmpty(dialog))
            {
                SearchPattern = dialog;

                WriteMessage(new[] { Messages.SetPatternResult, PatternStar + dialog + PatternStar }, new[] { _cNormal, _cPath });
            }
            else
                WriteMessage(new[] { Messages.SetPatternWarning }, new[] { _cWarning });
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Folder z plikami DLL". Otwiera okno dialogowe wyboru folderu, a następnie zapisuje wybrany folder do zmiennej "DllFolder"
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private void SetDLLFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new FolderBrowserDialog();

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                DllFolder = dialog.SelectedPath;
                SetDllDirectory(DllFolder);

                WriteMessage(new[] { Messages.DialogChooseDLLFolder, DllFolder }, new[] { _cNormal, _cPath });
            }
            else
            {
                WriteMessage(new[] { Messages.DialogChooseGraphNoFolder }, new[] { _cError });
            }
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Wyświetl ustawienia". Wypisuje w oknie dialogowym, a następnie konsoli, informacje o bieżących ustawieniach.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private async void DisplaySettings_Click(object sender, RoutedEventArgs e)
        {
            var isfile = string.IsNullOrEmpty(GraphPath) ? Messages.SettingsBlankDefault : (IsFile ? Messages.SettingsIsFileAFile : Messages.SettingsIsFileADirectory);
            var graphPath = string.IsNullOrEmpty(GraphPath) ? Messages.SettingsBlankDefault : GraphPath;
            var searchPattern = string.IsNullOrEmpty(SearchPattern) ? Messages.SettingsBlankDefault : PatternStar + SearchPattern + PatternStar;

            var message = new[]
            {
                EndLine,
                Messages.SettingsGraphPath, graphPath, EndLine,
                Messages.SettingsPattern, searchPattern, EndLine,
                Messages.SettingsIsFile, isfile, EndLine,
                Messages.SettingsLogFilePath, LogFile, EndLine,
                Messages.SettingsStatsFilePath, StatsFile,
                Messages.SettingsDllPath, DllFolder
            };

            await this.ShowMessageAsync(Messages.ActualSettings, message.Aggregate((i, j) => i + j));
            WriteMessage(message, new[] { _cNormal, _cNormal, _cPath, _cNormal, _cNormal, _cPath, _cNormal, _cNormal, _cPath, _cNormal, _cNormal, _cPath, _cNormal, _cNormal, _cPath, _cNormal, _cPath });
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Statystyki - wykresy". Wyświetla wykresy statystyk pamięci oraz czasu obliczceń.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private void DisplayStatCharts_Click(object sender, RoutedEventArgs e)
        {
            if (_stats.IsEmpty())
            {
                WriteMessage(new[] { Messages.StatsZeroRun }, new[] { _cWarning });
                return;
            }

            var tile = sender as Tile;
            if (tile == null) return;

            if (_bindings.VisibilityValue == Visibility.Hidden)
            {
                _bindings.VisibilityValue = Visibility.Visible;

                tile.Tag = Pressed;
                tile.Background = new SolidColorBrush(TileMouseOn);

                ContentPanel.Visibility = Visibility.Hidden;

                SetEnabledValues(false, false);
            }
            else
            {
                _bindings.VisibilityValue = Visibility.Hidden;

                tile.Tag = NotPressed;
                if (tile.IsMouseDirectlyOver)
                    tile.Background = (SolidColorBrush)(new BrushConverter().ConvertFrom(CheckboxPressed));

                ContentPanel.Visibility = Visibility.Visible;

                SetEnabledValues(true, false);
            }
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Statystyki - dane". Wypisuje zawartość pliku ze statystykami na konsolę oraz wyświetla statystyki w postaci wykresu liniowego.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private void DisplayStats_Click(object sender, RoutedEventArgs e)
        {
            if (_stats.IsEmpty())
            {
                WriteMessage(new[] { Messages.StatsZeroRun }, new[] { _cWarning });
                return;
            }

            WriteMessage(new[] { _stats.DisplayStats() });
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Czyść konsolę". Czyści zawartość konsoli (kontrolka "RichTextBox").
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private void ClearLog_Click(object sender, RoutedEventArgs e)
        {
            ContentPanel.Document.Blocks.Clear();
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Pomoc". Uaktywnia kontrolkę WebBrowser i ładuje plik z pomocą. Jeśli takiego pliku nie ma wyświetla stosowny komunikat w konsoli.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private void Help_Click(object sender, RoutedEventArgs e)
        {
            var tile = sender as Tile;
            if (tile == null) return;
  
            if (!AdobeReaderExist())
            {
                WriteMessage(new[] { Messages.ViewHelpNoAdobeReader }, new[] { _cError });
                return;
            } 

            if ((string)tile.Tag == NotPressed || (string)tile.Tag == null)
            {
                if (!File.Exists(HelpDocPath))
                {
                    WriteMessage(new[] { Messages.ViewHelpNoFile }, new[] { _cError });
                    return;
                }

                tile.Tag = Pressed;
                tile.Background = new SolidColorBrush(TileMouseOn);

                Browser.Navigate(new Uri(HelpDocPath));
                Browser.Visibility = Visibility.Visible;

                SetEnabledValues(false);
            }
            else
            {
                tile.Tag = NotPressed;
                if (tile.IsMouseDirectlyOver)
                    tile.Background = (SolidColorBrush)(new BrushConverter().ConvertFrom(CheckboxPressed));

                Browser.Visibility = Visibility.Hidden;
                Browser.Navigate(NavigateBlank);

                SetEnabledValues(true);
            }
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Ustawienia domyślne". Otwiera okno dialogowe wyboru opcji (OK/CANCEL), a następnie przywraca (bądź nie) ustawienia domyślne.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private async void DefaultSettings_Click(object sender, RoutedEventArgs e)
        {
            var result = await this.ShowMessageAsync(Messages.DefaultSettings, Messages.DefaultSettingsConfirmation, MessageDialogStyle.AffirmativeAndNegative);

            if (result == MessageDialogResult.Affirmative)
            {     
                GraphPath = string.Empty;
                SearchPattern = string.Empty;

                DllFolder = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + Dll;
                SetDllDirectory(DllFolder);

                WriteMessage(new[] { Messages.DefaultSettingsOK }, new[] { _cWarning });
            }
            else
            {
                WriteMessage(new[] { Messages.DefaultSettingsNOTOK }, new[] { _cWarning });
            }
        }

        /// <summary>
        /// Pomocnicza funkcja zarządzająca dostępnością przycisków menu w zależności od wybranej akcji.
        /// </summary>
        /// <param name="value">Wartość dostępności poszczególnych przycisków (bool).</param>
        /// <param name="type">Domyślnie wartość true - akcja przycisku Pomoc, wartość false - przycisku Wykresy.</param>
        private void SetEnabledValues(bool value, bool type = true)
        {
            if (type)
            {
                TileCharts.IsEnabled = value;   
                TileResetSettings.IsEnabled = value;
                TilePattern.IsEnabled = value;
                TileSettings.IsEnabled = value;
            }

            if ((string) TileCharts.Tag == Pressed && type) return;

            TileStats.IsEnabled = value;
            TileStart.IsEnabled = value;
        }

        /// <summary>
        /// Pomocnicza funkcja sprawdzająca, czy w systemie jest zainstalowany program Adobe Acrobat Reader.
        /// </summary>
        /// <returns>True - program jest zainstalowany, false - w p.p.</returns>
        private static bool AdobeReaderExist()
        {
            RegistryKey acroRead = null;

            var openSubKey = Registry.LocalMachine.OpenSubKey("Software");

            if (openSubKey == null) return false;

            var adobe = openSubKey.OpenSubKey("Adobe");

            if (adobe == null)
            {
                var policies = openSubKey.OpenSubKey("Policies");

                if (policies == null)
                    return false;

                adobe = policies.OpenSubKey("Adobe");
            }

            if (adobe != null)
                acroRead = adobe.OpenSubKey("Acrobat Reader");

            return acroRead != null;
        }

        #endregion

        #region Główna funkcja uruchamiająca przetwarzanie
        
        /// <summary>
        /// Funkcja wywoływana przez przycisk "Uruchom". Rozpoczyna obliczenie zależnie od wybranych wcześniej opcji.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private async void Run_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(GraphPath))
            {
                WriteMessage(new[] { Messages.RunNoInputData }, new[] { _cError });
                return;
            }

            if ((CpuT.Tag == null || (string)CpuT.Tag == NotPressed) &&
                (CpuB.Tag == null || (string)CpuB.Tag == NotPressed) &&
                (Gpu.Tag == null || (string)Gpu.Tag == NotPressed))
            {
                WriteMessage(new[] { Messages.RunNoMethod }, new[] { _cError });
                return;
            }

            _bindings.EnabledValue = !_bindings.EnabledValue;
            _bindings.StopButtonVisibilityValue = Visibility.Visible;

            try
            {
                var cancellationTokenSource = new CancellationTokenSource();

                _cancelWork = () =>
                {
                    _bindings.StopButtonVisibilityValue = Visibility.Hidden;
                    cancellationTokenSource.Cancel();
                };

                var token = cancellationTokenSource.Token;

                if (IsFile)
                {
                    _viewModel.ClearChart();

                    var g = Preprocessing(GraphPath);
                    if (g == null)
                    {
                        CancelComputation();
                        return;
                    }

                    if ((string) CpuT.Tag == Pressed)
                        await Task.Run(() => MethodTableCpu(g, GraphPath, token), token);       
                    if ((string)CpuB.Tag == Pressed)
                        await Task.Run(() => MethodBitCpu(g, GraphPath, token), token);
                    if ((string)Gpu.Tag == Pressed)
                        await Task.Run(() => MethodGpu(g, GraphPath, token), token);
                }
                else
                {
                    var files = !string.IsNullOrEmpty(SearchPattern) ?
                        Directory.GetFiles(GraphPath) :
                        Directory.GetFiles(GraphPath, PatternStar + SearchPattern + PatternStar, SearchOption.TopDirectoryOnly);

                    foreach (var f in files)
                    {
                        _viewModel.ClearChart();
                        var f1 = f;

                        var g = Preprocessing(f1);
                        if (g == null) continue;

                        if ((string)CpuT.Tag == Pressed)
                            await Task.Run(() => MethodTableCpu(g, f1, token), token);
                        if ((string)CpuB.Tag == Pressed)
                            await Task.Run(() => MethodBitCpu(g, f1, token), token);
                        if ((string)Gpu.Tag == Pressed)
                            await Task.Run(() => MethodGpu(g, f1, token), token);
                    }
                }            
            }
            catch (Exception ee)
            {
                WriteMessage(new[] { ee.Message }, new[] { _cError });
            }

            CancelComputation();
      
            _stats.SaveToFile(StatsFile);
        }

        /// <summary>
        /// Funckja pomocnicza zawierająca zestaw instrukcji do wykonania po zakończeniu obliczeń.
        /// </summary>
        private void CancelComputation()
        {
            _bindings.EnabledValue = !_bindings.EnabledValue;
            _bindings.StopButtonVisibilityValue = Visibility.Hidden;
            _cancelWork = null;
        }

        /// <summary>
        /// Funkcja wywoływana przez przycisk "Przerwij". Przerywa obliczenia rozpoczęte w osobnym wątku.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private void Stop_Click(object sender, RoutedEventArgs e)
        {
            if (_cancelWork != null)
                _cancelWork();
        }

        #endregion

        #region Funkcje obsługujące zdarzenia kafelków

        /// <summary>
        /// Funkcja wywoływana przez przyciski: "GPU", "Table CPU" oraz "Bit CPU". Zaznacza metody, jakie zostaną uruchomione podczas obliczeń.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        private void AlgorithmSelection_Click(object sender, RoutedEventArgs e)
        {
            var tile = sender as Tile;
            if (tile == null) return;

            if ((string)tile.Tag == NotPressed || (string)tile.Tag == null)
            {
                tile.Tag = Pressed;
                tile.Background = new SolidColorBrush(TileMouseOn);
            }
            else
            {
                tile.Tag = NotPressed;
                if (tile.IsMouseDirectlyOver)
                    tile.Background = (SolidColorBrush)(new BrushConverter().ConvertFrom(CheckboxPressed));
            }
        }

        /// <summary>
        /// Zdarzenie wywoływane w momencie znalezienia się kursora myszy nad dowolnym kafelkiem z menu.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (MouseEnter).</param>
        private void Tile_MouseEnter(object sender, MouseEventArgs e)
        {
            var tile = sender as Tile;
            if (tile != null) tile.Background = new SolidColorBrush(TileMouseOn);
        }

        /// <summary>
        /// Zdarzenie wywoływane w momencie zniknięcia kursora myszy znad dowolnego kafelka z menu.
        /// </summary>
        /// <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        /// <param name="e">Domyślne zdarzenie kafelka (MouseLeave).</param>
        private void Tile_MouseLeave(object sender, MouseEventArgs e)
        {
            var tile = sender as Tile;
            if (tile != null && (string)tile.Tag != Pressed) tile.Background = (SolidColorBrush)(new BrushConverter().ConvertFrom(CheckboxPressed));
        }

        #region Tymczasowo nieużywane funkcje obsługi kafelka dwujęzyczności
        ///// <summary>
        ///// Funkcja wywoływana przez przycisk z obrazem flagi. Zmienia język (PL/EN) wyświetlnych komunikatów (nie zmienia języka kafelków menu).
        ///// </summary>
        // <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        // <param name="e">Domyślne zdarzenie kafelka (Click).</param>
        //private void Language_Click(object sender, RoutedEventArgs e)
        //{
        //    var tile = sender as Tile;
        //    if (Thread.CurrentThread.CurrentCulture.Name == CulturePl)
        //    {
        //        Thread.CurrentThread.CurrentUICulture = new CultureInfo(CultureEn);
        //        Thread.CurrentThread.CurrentCulture = new CultureInfo(CultureEn);

        //        if (tile == null) return;
        //        tile.Background = new ImageBrush { ImageSource = new BitmapImage(new Uri(ImageEn)) };
        //        tile.Tag = 1;
        //    }
        //    else
        //    {
        //        Thread.CurrentThread.CurrentUICulture = new CultureInfo(CulturePl);
        //        Thread.CurrentThread.CurrentCulture = new CultureInfo(CulturePl);

        //        if (tile == null) return;
        //        tile.Background = new ImageBrush { ImageSource = new BitmapImage(new Uri(ImagePl)) };
        //        tile.Tag = null;
        //    }
        //}

        ///// <summary>
        ///// Zdarzenie wywoływane w momencie znalezienia się kursora myszy nad kafelkiem z obrazem flagi.
        ///// </summary>
        // <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        // <param name="e">Domyślne zdarzenie kafelka (MouseEnter).</param>
        //private void Language_MouseEnter(object sender, MouseEventArgs e)
        //{
        //    var tile = sender as Tile;
        //    if (tile == null) return;

        //    tile.Background = tile.Tag == null ?
        //        new ImageBrush { ImageSource = new BitmapImage(new Uri(ImageEn)) } :
        //        new ImageBrush { ImageSource = new BitmapImage(new Uri(ImagePl)) };

        //    tile.Opacity = 0.5;
        //}

        ///// <summary>
        ///// Zdarzenie wywoływane w momencie znalezienia się kursora myszy nad kafelkiem z obrazem flagi.
        ///// </summary>
        // <param name="sender">Domyślny obiekt kafelka (Tile).</param>
        // <param name="e">Domyślne zdarzenie kafelka (MouseLeave).</param>
        //private void Language_MouseLeave(object sender, MouseEventArgs e)
        //{
        //    var tile = sender as Tile;
        //    if (tile == null) return;

        //    tile.Background = tile.Tag == null ? 
        //        new ImageBrush { ImageSource = new BitmapImage(new Uri(ImagePl)) } :
        //        new ImageBrush { ImageSource = new BitmapImage(new Uri(ImageEn)) };

        //    tile.Opacity = 1;
        //}

        #endregion

        #endregion
    }
}
