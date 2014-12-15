using System.Diagnostics;
using System.Windows;
using GraphColoring.Structures;
using MenuItem = System.Windows.Controls.MenuItem;
using MessageBox = System.Windows.MessageBox;
using OpenFileDialog = System.Windows.Forms.OpenFileDialog;
using System.Runtime.InteropServices; 

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
        [DllImport("..\\..\\..\\Release\\GraphColoringCPU.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int FindChromaticNumber([MarshalAs(UnmanagedType.LPArray)]int[] vertices, [MarshalAs(UnmanagedType.LPArray)]int[] neighborsCount, int n, int flag = 0);

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
        /// Metoda obsługująca zdarzenie uruchomienia obliczenia algorytmu napisanego w wersji C# (niezoptymalizowany)
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void CPU1_OnClick(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(_lastPath))
            {
                var g = FileProcessing.ReadFile(_lastPath);

                var watch = Stopwatch.StartNew();

                var k = g.GetChromaticNumber();

                watch.Stop();
                MessageBox.Show(string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}ms", k, watch.ElapsedMilliseconds));
            }
            else
            {
                MessageBox.Show("Jakbyś podał graf na wejściu, to ja bym policzył :(");
            }
        }

        /// <summary>
        /// Metoda obsługująca zdarzenie uruchomienia obliczenia algorytmu napisanego w wersji C++ (zoptymalizowany)
        /// </summary>
        /// <param name="sender">Obiekt, który wywołał zdarzenie.</param>
        /// <param name="e">Parametry zdarzenia.</param>
        private void CPU2_OnClick(object sender, RoutedEventArgs e)
        {
            _lastPath = "..\\..\\..\\..\\TestFiles\\GraphExample22.txt";
            if (!string.IsNullOrEmpty(_lastPath))
            {
                var g = FileProcessing.ReadFile(_lastPath);
                g = FileProcessing.ConvertToBitVersion(g);
                var watch = Stopwatch.StartNew();

                var k = FindChromaticNumber(g.Vertices, g.NeighboursCount, g.VerticesCount,1);

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
            MessageBox.Show("Element '" + ((MenuItem) sender).Header + "' nie jest obsługiwany w bieżącej wersji.");
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
