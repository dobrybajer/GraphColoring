using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using GraphColoring.Structures;
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
        public static extern int FindChromaticNumberGPU([MarshalAs(UnmanagedType.LPArray)]int[] wynik,[MarshalAs(UnmanagedType.LPArray)]int[] pamiec, [MarshalAs(UnmanagedType.LPArray)]int[] vertices, [MarshalAs(UnmanagedType.LPArray)]int[] neighborsCount, int n, int allVertices);

        readonly ConsoleContent _dc = new ConsoleContent();

        private bool isFile = false;

        public MainWindow()
        {
            InitializeComponent();
            Loaded += MainWindow_Loaded;          
            DataContext = _dc;
            _dc.Path = "Nie wybrano.";   
        }

        private void GPU(string path)
        {
            var g = FileProcessing.ReadFile(path);
            var wynik = new int[g.VerticesCount];
            var pamiec = new int[2 * (g.VerticesCount-1) + 2];
          
            var watch = Stopwatch.StartNew();

            FindChromaticNumberGPU(wynik, pamiec,  g.Vertices, g.NeighboursCount, g.VerticesCount, g.AllVerticesCount);
            var tmp = pamiec;
            var wynikk = -2;

            for (var i = 0; i < g.VerticesCount; i++)
            {
                if (wynik[i] == -1 || wynik[i] == 0) continue;
                wynikk = wynik[i] + 1;
                break;
            }

            _dc.RunCommandType(0, string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}", wynikk, watch.Elapsed));

            InputBlock.Focus();
            ContentPanel.ScrollToBottom();
        }

        private void CPUT(string path)
        {
            var g = FileProcessing.ReadFile(path);
            var watch = Stopwatch.StartNew();

            var k = FindChromaticNumber(g.Vertices, g.NeighboursCount, g.VerticesCount);

            watch.Stop();

            _dc.RunCommandType(0, string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}", k, watch.Elapsed));

            InputBlock.Focus();
            ContentPanel.ScrollToBottom();
        }

        private void CPUB(string path)
        {
            var g = FileProcessing.ReadFile(path);
            g = FileProcessing.ConvertToBitVersion(g);
            var watch = Stopwatch.StartNew();

            var k = FindChromaticNumber(g.Vertices, g.NeighboursCount, g.VerticesCount, 1);

            watch.Stop();

            _dc.RunCommandType(0, string.Format("Graf jest co najwyżej {0}-kolorowalny.\nCzas obliczeń: {1}", k, watch.Elapsed));

            InputBlock.Focus();
            ContentPanel.ScrollToBottom();
        }

        void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            InputBlock.KeyDown += InputBlock_KeyDown;
            InputBlock.Focus();
        }

        void InputBlock_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key != Key.Enter) return;
            _dc.ConsoleInput = InputBlock.Text;
            _dc.RunCommand();
            InputBlock.Focus();
            ContentPanel.ScrollToBottom();
        }

        private void ChooseFile_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog1 = new OpenFileDialog
            {
                Filter = @"Pliki tekstowe|*.txt",
                Title = @"Wybierz plik tekstowy zawierający reprezentację grafu"
            };

            if (openFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                isFile = true;
                _dc.Path = openFileDialog1.InitialDirectory + openFileDialog1.FileName;
                _dc.RunCommandType(0, "Ścieżka do wybranego pliku: " + _dc.Path);
                InputBlock.Focus();
                ContentPanel.ScrollToBottom();
            }
            else
            {
                _dc.RunCommandType(0, "Nie wybrałeś żadnego pliku.");
                InputBlock.Focus();
                ContentPanel.ScrollToBottom();
            }
        }

        private void ChooseFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
       
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                isFile = false;
                _dc.Path = dialog.SelectedPath;
                _dc.RunCommandType(0, "Ścieżka do wybranego folderu: " + _dc.Path);
                InputBlock.Focus();
                ContentPanel.ScrollToBottom();
            }
            else
            {
                _dc.RunCommandType(0, "Nie wybrałeś żadnego folderu.");
                InputBlock.Focus();
                ContentPanel.ScrollToBottom();
            }
        }

        private void ClearLog_Click(object sender, RoutedEventArgs e)
        {
            _dc.RunCommandType(1);
            InputBlock.Focus();
            ContentPanel.ScrollToBottom();
        }

        private void MakeLog_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = @"Pliki tekstowe|*.txt",
                Title = @"Wybierz plik do jakiego ma być zapisany log."
            };

            if (dialog.ShowDialog() == true)
            {
                _dc.RunCommandType(2, dialog.InitialDirectory + dialog.FileName);        
            }
            else
            {
                _dc.RunCommandType(0, "Nie wybrałeś żadnego pliku do zapisu pliku z logiem.");
                InputBlock.Focus();
                ContentPanel.ScrollToBottom();
            }
        }

        private void Run_Click(object sender, RoutedEventArgs e)
        {
            if (_dc.Path != "Nie wybrano." && (CpuT.IsChecked == true || CpuB.IsChecked == true || Gpu.IsChecked == true))
            {
                if (isFile)
                {
                    if (CpuT.IsChecked == true)
                        CPUT(_dc.Path);
                    if (CpuB.IsChecked == true)
                        CPUB(_dc.Path);
                    if (Gpu.IsChecked == true)
                        GPU(_dc.Path);
                }
                else
                {
                    var files = Directory.GetFiles(_dc.Path, "*GraphExample*", SearchOption.TopDirectoryOnly);
                    foreach (var f in files)
                    {
                        if (CpuT.IsChecked == true)
                            CPUT(f);
                        if (CpuB.IsChecked == true)
                            CPUB(f);
                        if (Gpu.IsChecked == true)
                            GPU(f);
                    }
                }   
            }
            else
            {
                _dc.RunCommandType(0, "Błąd: Nie wybrano pliku, ani folderu lub nie wybrano metody algorytmu.");
                InputBlock.Focus();
                ContentPanel.ScrollToBottom();
            }
        }
    }

    public class ConsoleContent : INotifyPropertyChanged
    {
        private string _path;

        public string Path
        {
            get { return _path; }
            set
            {
                if (value == _path) return;
                _path = value;
                OnPropertyChanged("Path");
            }
        }

        string _consoleInput = string.Empty;
        ObservableCollection<string> _consoleOutput = new ObservableCollection<string>() { "Hello!" };

        public string ConsoleInput
        {
            get
            {
                return _consoleInput;
            }
            set
            {
                _consoleInput = value;
                OnPropertyChanged("ConsoleInput");
            }
        }

        public ObservableCollection<string> ConsoleOutput
        {
            get
            {
                return _consoleOutput;
            }
            set
            {
                _consoleOutput = value;
                OnPropertyChanged("ConsoleOutput");
            }
        }

        public void RunCommand()
        {
            ConsoleOutput.Add(ConsoleInput);
            // do your stuff here.
            ConsoleInput = String.Empty;
        }

        public void RunCommandType(int type=0, string message="")
        {
            switch (type)
            {
                case 0:
                    ConsoleOutput.Add(message);
                    break;
                case 1:
                    ConsoleOutput.Clear();
                    break;
                case 2:
                    File.WriteAllLines(message, ConsoleOutput.ToList());
                    break;
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;
        void OnPropertyChanged(string propertyName)
        {
            if (null != PropertyChanged)
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
