using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Media;
using GraphColoring.Structures;
using MahApps.Metro.Controls;
using MahApps.Metro.Controls.Dialogs;
using Control = System.Windows.Controls.Control;
using MessageBox = System.Windows.MessageBox;
using MouseEventArgs = System.Windows.Input.MouseEventArgs;
using SaveFileDialog = Microsoft.Win32.SaveFileDialog;

namespace GraphColoring
{
    /// <summary>
    /// Klasa obsługująca UI i zdarzenia w nim występujące.
    /// </summary>
    public partial class MainWindow
    {
        #region Definicje metod z załączonych plików DDL

        /// <summary>
        /// 
        /// </summary>
        /// <param name="pamiec"></param>
        /// <param name="vertices"></param>
        /// <param name="neighborsCount"></param>
        /// <param name="n"></param>
        /// <param name="flag"></param>
        /// <returns></returns>
        [DllImport("..\\..\\..\\Debug\\GraphColoringCPU.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int FindChromaticNumber([MarshalAs(UnmanagedType.LPArray)]int[] pamiec,[MarshalAs(UnmanagedType.LPArray)]int[] vertices, [MarshalAs(UnmanagedType.LPArray)]int[] neighborsCount, int n, int flag = 0);

        /// <summary>
        /// Finds the chromatic number gpu.
        /// </summary>
        /// <param name="wynik">The wynik.</param>
        /// <param name="pamiec">Przechowuje statystyki zuzycia pamie</param>
        /// <param name="vertices">The vertices.</param>
        /// <param name="neighborsCount">The neighbors count.</param>
        /// <param name="n">The n.</param>
        /// <param name="allVertices">All vertices.</param>
        /// <returns></returns>
        [DllImport("..\\..\\..\\Debug\\GraphColoringGPU.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int FindChromaticNumberGPU([MarshalAs(UnmanagedType.LPArray)]int[] wynik, [MarshalAs(UnmanagedType.LPArray)]int[] pamiec, [MarshalAs(UnmanagedType.LPArray)]int[] vertices, [MarshalAs(UnmanagedType.LPArray)]int[] neighborsCount, int n, int allVertices);
        
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
        /// Zmienna przechowująca ścieżkę do pliku z logiem, domyślnie plik z logiem znajduje się w folderze z projektem. Ścieżka względna: "/Output/Log_{DATA}.txt", gdzie parametr {DATA} to aktualna data i czas w domyślnym formacie TIMESTAMP systemu WINDOWS
        /// </summary>
        public string LogFile { get; private set; }

        /// <summary>
        /// Zmienna przechowująca ścieżkę do pliku ze statystykami, domyślnie plik ze statystykami znajduje się w folderze z projektem. Ścieżka względna: "/Output/Statistics_{DATA}.txt", gdzie parametr {DATA} to aktualna data i czas w domyślnym formacie TIMESTAMP systemu WINDOWS
        /// </summary>
        public string StatsFile { get; private set; }

        /// <summary>
        /// Zmienna przechowująca element wspólny nazwy plików w przypadku wybrania ścieżki do folderu z plikami. W przypadku podania pustej wartości wybierane są wszystkie pliki z wybranego folderu. 
        /// </summary>
        public string SearchPattern { get; private set; }

        private readonly Statistics _stats;

        #endregion

        #region Konstruktor okna głównego
        /// <summary>
        /// Konstruktor okna głównego. Inicjalizuje dodatkowe zmienne użyte w aplikacji.
        /// </summary>
        public MainWindow()
        {
            InitializeComponent();    
            _stats = new Statistics();
            var directory = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + "\\Output\\";
            LogFile = Path.Combine(directory, String.Format("Log_{0:yyyy-MM-dd hh/mm/ss}.txt", DateTime.Now));
            StatsFile = Path.Combine(directory, "Stats_" + DateTime.Now + ".txt");
        }
        #endregion

        #region Logowanie wiadomości
        private void WriteMessage(string message)
        {
            if (!string.IsNullOrEmpty(LogFile))
            {
                AppendToFile(message);
            }

            AddToOutput(message);
        }

        private void AppendToFile(string message)
        {
            //if (!File.Exists(LogFile))
            //{
            //    using (var fs = File.Create(LogFile))
            //    {
            //        using (var sw = new StreamWriter(fs))
            //        {
            //            sw.WriteLine(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss > ") + message);
            //        }
            //    }
            //}
            //else
            //{
                //using (var fs = new FileStream(LogFile, FileMode.Append , FileAccess.Write))
                using (var sw = new StreamWriter(LogFile, true))
                {
                    sw.WriteLine(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss > ") + message);
                }
            //}
        }

        private void AddToOutput(string message)
        {
            ContentPanel.AppendText(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss > "));
            ContentPanel.AppendText(message+"\r\n");
            ContentPanel.Focus();
            ContentPanel.CaretIndex = ContentPanel.Text.Length;
            ContentPanel.ScrollToEnd();
        }
        #endregion

        #region Funkcje uruchamiające obliczanie algorytmu różnymi metodami
        private void MethodGpu(string path)
        {
            try
            {
                var g = FileProcessing.ReadFile(path);
                var wynik = new int[g.VerticesCount];
                var pamiec = new int[2 * (g.VerticesCount - 1) + 2];

                var watch = Stopwatch.StartNew();

                FindChromaticNumberGPU(wynik, pamiec, g.Vertices, g.NeighboursCount, g.VerticesCount, g.AllVerticesCount);

                var wynikk = -2;

                for (var i = 0; i < g.VerticesCount; i++)
                {
                    if (wynik[i] == -1 || wynik[i] == 0) continue;
                    wynikk = wynik[i] + 1;
                    break;
                }

                watch.Stop();

                _stats.Add(path, 0, watch.Elapsed, pamiec);

                WriteMessage(string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}", wynikk, watch.Elapsed));
            }
            catch (Exception e)
            {
                WriteMessage("[GPU] Nieoczekiwany błąd, plik " + path + " nie został przetworzony\r\nKod błędu: " + e.Message);
            }
        }

        private void MethodTableCpu(string path)
        {
            try
            {
                var g = FileProcessing.ReadFile(path);
                var pamiec = new int[2 * (g.VerticesCount - 1) + 2];

                var watch = Stopwatch.StartNew();

                var k = FindChromaticNumber(pamiec, g.Vertices, g.NeighboursCount, g.VerticesCount);

                watch.Stop();

                _stats.Add(path, 1, watch.Elapsed, pamiec);

                WriteMessage(string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}", k, watch.Elapsed));
            }
            catch (Exception e)
            {
                WriteMessage("[CPU Table] Nieoczekiwany błąd, plik " + path + " nie został przetworzony\r\nKod błędu: " + e.Message);
            }
            
        }

        private void MethodBitCpu(string path)
        {
            try
            {
                var g = FileProcessing.ReadFile(path);
                g = FileProcessing.ConvertToBitVersion(g);
                var pamiec = new int[2 * (g.VerticesCount - 1) + 2];

                var watch = Stopwatch.StartNew();

                var k = FindChromaticNumber(pamiec, g.Vertices, g.NeighboursCount, g.VerticesCount, 1);

                watch.Stop();

                _stats.Add(path, 2, watch.Elapsed, pamiec);

                WriteMessage(string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}", k, watch.Elapsed));
            }
            catch (Exception e)
            {
                WriteMessage("[CPU Bit] Nieoczekiwany błąd, plik " + path + " nie został przetworzony\r\nKod błędu: " + e.Message);
            }

        }
        #endregion

        #region Funkcje dodatkowe odpowiadające przyciskom z menu głównego
        private void ChooseGraphFile_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog1 = new OpenFileDialog
            {
                Filter = @"Pliki tekstowe|*.txt",
                Title = @"Wybierz plik tekstowy zawierający reprezentację grafu"
            };

            if (openFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                IsFile = true;
                GraphPath = openFileDialog1.InitialDirectory + openFileDialog1.FileName;
                WriteMessage("Ścieżka do wybranego pliku: " + GraphPath);
            }
            else
            {
                WriteMessage("Nie wybrałeś żadnego pliku.");
            }
        }

        private void ChooseGraphFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new FolderBrowserDialog();

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                IsFile = false;
                GraphPath = dialog.SelectedPath;
                WriteMessage("Ścieżka do wybranego folderu: " + GraphPath); 
            }
            else
            {
                WriteMessage("Nie wybrałeś żadnego folderu."); 
            }
        }

        private async void SetPattern_Click(object sender, RoutedEventArgs e)
        {
            var dialog = await this.ShowInputAsync("Wpisz wzorzec do wyszukania odpowiednich plików w wybranym folderze.", "Rozróżniane są małe i wielkie litery.");
            if (!string.IsNullOrEmpty(dialog))
                SearchPattern = dialog;
        }

        private void ChooseLogFile_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new SaveFileDialog
            {
                Filter = @"Pliki tekstowe|*.txt",
                Title = @"Wybierz plik do jakiego ma być zapisany log."
            };

            if (dialog.ShowDialog() == true)
            {
                LogFile = dialog.FileName;
                WriteMessage(dialog.InitialDirectory + dialog.FileName);
            }
            else
            {
                WriteMessage("Nie wybrałeś żadnego pliku do zapisu pliku z logiem.");
            }
        }

        private void ChooseStatsFile_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new SaveFileDialog
            {
                Filter = @"Pliki tekstowe|*.txt",
                Title = @"Wybierz plik do jakiego mają być zapisane statystyki."
            };

            if (dialog.ShowDialog() == true)
            {
                StatsFile = dialog.FileName;
                WriteMessage(dialog.InitialDirectory + dialog.FileName);
            }
            else
            {
                WriteMessage("Nie wybrałeś żadnego pliku do zapisu pliku z logiem.");
            }
        }

        private async void DisplaySettings_Click(object sender, RoutedEventArgs e)
        {
            var isfile = string.IsNullOrEmpty(GraphPath) ? "Nie wybrano" : (IsFile ? "P" : "F");

            var message = string.Format("{0}{1}\r\n{2}{3}\r\n{4}{5}\r\n{6}{7}\r\n{8}{9}",
                "Ścieżka do danych z reprezentacją grafu: ", GraphPath,
                "Wzorzec, wg którego będą wybierane pliki z folderu (jeżeli dotyczy) :", SearchPattern,
                "Czy wybrana ścieżka wskazuje na plik (P), czy folder (F): ", isfile,
                "Ścieżka do pliku z logiem: ", LogFile,
                "Ścieżka do pliku ze statystykami: ", StatsFile
                );

            await this.ShowMessageAsync("Aktualne ustawienia", message);
            WriteMessage("Aktualne ustawienia\r\n" + message);
        }

        private void DisplayStats_Click(object sender, RoutedEventArgs e)
        {
            WriteMessage(_stats.DisplayStats());
        }
        
        private void ClearLog_Click(object sender, RoutedEventArgs e)
        {
            ContentPanel.Clear();
        }

        private static void PressTile(Control tile)
        {
            if ((string)tile.Tag == "NotPressed" || (string)tile.Tag == null)
            {
                tile.Tag = "Pressed";
                tile.Background = new SolidColorBrush(Colors.DimGray);
            }
            else
            {
                tile.Tag = "NotPressed";
                if(tile.IsMouseDirectlyOver)
                    tile.Background = (SolidColorBrush)(new BrushConverter().ConvertFrom("#CC119EDA"));
            }
        }

        private void AlgorithmSelection_Click(object sender, RoutedEventArgs e)
        {
            var tile = sender as Tile;
            if (tile == null) return;

            switch (tile.Name)
            {
                case "Gpu":
                    PressTile(tile);
                    break;
                case "CpuT":
                    PressTile(tile);
                    break;
                case "CpuB":
                    PressTile(tile);
                    break;
            }
        }

        #endregion

        #region Główna funkcja uruchamiająca przetwarzanie
        private void Run_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(GraphPath))
            {
                WriteMessage("Błąd: Nie wybrano pliku, ani folderu z plikami zawierającymi reprezentację grafu.");
                return;
            }

            if ((string)CpuT.Tag == "NotPressed" && (string)CpuB.Tag == "NotPressed" && (string)Gpu.Tag == "NotPressed")
            {
                WriteMessage("Błąd: Nie wybrano żadnego algorytmu rozwiązującego problem.");
                return;
            }

            if (IsFile)
            {
                if ((string)CpuT.Tag=="Pressed")
                    MethodTableCpu(GraphPath);
                if ((string)CpuB.Tag == "Pressed")
                    MethodBitCpu(GraphPath);
                if ((string)Gpu.Tag == "Pressed")
                    MethodGpu(GraphPath);
            }
            else
            {
                var files = Directory.GetFiles(GraphPath, "*"+SearchPattern+"*", SearchOption.TopDirectoryOnly);
                foreach (var f in files)
                {
                    if ((string)CpuT.Tag == "Pressed")
                        MethodTableCpu(f);
                    if ((string)CpuB.Tag == "Pressed")
                        MethodBitCpu(f);
                    if ((string)Gpu.Tag == "Pressed")
                        MethodGpu(f);
                }
            }   
            _stats.SaveToFile(StatsFile);
        }
        #endregion

        #region Funkcje obsługujące zdarzenia kafelków
        private void Tile_MouseEnter(object sender, MouseEventArgs e)
        {
            var tile = sender as Tile;
            if (tile != null) tile.Background = new SolidColorBrush(Colors.DimGray);
        }

        private void Tile_MouseLeave(object sender, MouseEventArgs e)
        {
            var tile = sender as Tile;
            if (tile != null && (string)tile.Tag != "Pressed") tile.Background = (SolidColorBrush)(new BrushConverter().ConvertFrom("#CC119EDA"));
        }
        #endregion
    }
}
