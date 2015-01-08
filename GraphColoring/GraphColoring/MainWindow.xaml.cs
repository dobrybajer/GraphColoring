using System.Diagnostics;
using System.Windows;
using GraphColoring.Structures;
using MenuItem = System.Windows.Controls.MenuItem;
using MessageBox = System.Windows.MessageBox;
using OpenFileDialog = System.Windows.Forms.OpenFileDialog;
using System.Runtime.InteropServices;

/// <summary>
/// Przestrzeń nazwy dla aplikacji.
/// </summary>
namespace GraphColoring
{
    /// <summary>
    /// Klasa obsługująca UI i zdarzenia w nim występujące.
    /// </summary>
    public partial class MainWindow
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="vertices"></param>
        /// <param name="neighborsCount"></param>
        /// <param name="n"></param>
        /// <param name="flag"></param>
        /// <returns></returns>
        [DllImport("..\\..\\..\\Debug\\GraphColoringCPU.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int FindChromaticNumber([MarshalAs(UnmanagedType.LPArray)]int[] vertices, [MarshalAs(UnmanagedType.LPArray)]int[] neighborsCount, int n, int flag = 0);

        /// <summary>
        /// Finds the chromatic number gpu.
        /// </summary>
        /// <param name="wynik">The wynik.</param>
        /// <param name="vertices">The vertices.</param>
        /// <param name="neighborsCount">The neighbors count.</param>
        /// <param name="n">The n.</param>
        /// <param name="allVertices">All vertices.</param>
        /// <returns></returns>
        [DllImport("..\\..\\..\\Debug\\GraphColoringGPU.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int FindChromaticNumberGPU([MarshalAs(UnmanagedType.LPArray)]int[] wynik, [MarshalAs(UnmanagedType.LPArray)]int[] vertices, [MarshalAs(UnmanagedType.LPArray)]int[] neighborsCount, int n, int allVertices);

        /// <summary>
        /// Ścieżka ostatnio otworzonego pliku z reprezentacją grafu.
        /// </summary>
        private string _lastPath;

        /// <summary>
        /// Domyślny konstruktor.
        /// </summary>
        public MainWindow()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Metoda obsługująca zdarzenie otworzenie okna dialogowego wyboru pliku.
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void Open_OnClick(object sender, RoutedEventArgs e)
        {
            var openFileDialog1 = new OpenFileDialog
            {
                Filter = @"Pliki tekstowe|*.txt",
                Title = @"Wybierz plik tekstowy zawierający reprezentację grafu"
            };

            if (openFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                _lastPath=openFileDialog1.InitialDirectory + openFileDialog1.FileName;
            }
            else
            {
                MessageBox.Show("Nie wybrałeś żadnego pliku, spróbuj ponownie.");
            }
        }

        /// <summary>
        /// Metoda obsługująca zdarzenie ...
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void Generate_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + ((MenuItem) sender).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        /// <summary>
        /// Metoda obsługująca zdarzenie uruchomienia obliczenia algorytmu napisanego w wersji C++ (zoptymalizowany)
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void CPU_OnClick(object sender, RoutedEventArgs e)
        {
            //_lastPath = "..\\..\\..\\..\\TestFiles\\GraphExample12.txt";
            if (!string.IsNullOrEmpty(_lastPath))
            {
                var g = FileProcessing.ReadFile(_lastPath);
                g = FileProcessing.ConvertToBitVersion(g);
                var watch = Stopwatch.StartNew();

                var k = FindChromaticNumber(g.Vertices, g.NeighboursCount, g.VerticesCount, 1);

                watch.Stop();
                MessageBox.Show(string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}", k, watch.Elapsed));
            }
            else
            {
                MessageBox.Show("Jakbyś podał graf na wejściu, to ja bym policzył :(");
            }
        }

        /// <summary>
        /// Metoda obsługująca zdarzenie uruchomienia obliczenia algorytmu napisanego w wersji CUDA C++ (zoptymalizowany)
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void GPU_OnClick(object sender, RoutedEventArgs e)
        {
            //_lastPath = "..\\..\\..\\..\\TestFiles\\GraphExample12.txt";
            if (!string.IsNullOrEmpty(_lastPath))
            {
                var g = FileProcessing.ReadFile(_lastPath);
                int[] wynik = new int[g.VerticesCount];

                var watch = Stopwatch.StartNew();
                
                FindChromaticNumberGPU(wynik, g.Vertices, g.NeighboursCount, g.VerticesCount, g.AllVerticesCount);

                int wynikk = -2;

                for (int i = 0; i < g.VerticesCount; i++)
                {
                    if (wynik[i] != -1 && wynik[i] != 0)
                    {
                        wynikk = wynik[i] + 1;
                        break;
                    }
                }

                watch.Stop();
                MessageBox.Show(string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}", wynikk, watch.Elapsed));
            }
            else
            {
                MessageBox.Show("Jakbyś podał graf na wejściu, to ja bym policzył :(");
            }
        }

        /// <summary>
        /// Metoda obsługująca zdarzenie otworzenie okna dialogowego z informacjami o programie.
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void AboutProgram_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + ((MenuItem) sender).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        /// <summary>
        /// Metoda obsługująca zdarzenie otworzenie okna dialogowego z informacjami o algorytmie.
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void AboutAlgorithm_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + ((MenuItem) sender).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        /// <summary>
        /// Metoda obsługująca zdarzenie otworzenie okna dialogowego z instrukcją obsługi aplikacji.
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void HowTo_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + ((MenuItem) sender).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        /// <summary>
        /// Metoda obsługująca zdarzenie otworzenie okna dialogowego z aktualną wersją aplikacji.
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void Version_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + ((MenuItem) sender).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }
    }
}
