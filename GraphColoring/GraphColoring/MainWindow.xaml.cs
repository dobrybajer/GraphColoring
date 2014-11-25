using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Forms;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using GraphColoring.Properties;
using GraphColoring.UserControls;
using Microsoft.Win32;
using Help = GraphColoring.UserControls.Help;
using MenuItem = System.Windows.Controls.MenuItem;
using MessageBox = System.Windows.MessageBox;
using OpenFileDialog = System.Windows.Forms.OpenFileDialog;


namespace GraphColoring
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private GraphOld graph;
        private string lastPath;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void Tile_Click(object sender, RoutedEventArgs e)
        {
            var s = sender as Tile;
   
            if (s == null) return;
            switch ((string)s.Tag)
            {
                case "M1":
                    Content.Children.Clear();
                    Content.Children.Add(new LoadData());
                
       
                    break;
                case "M2":
                    Content.Children.Clear();
                    //Content.Children.Add();
                    break;
                case "M3":
                    Content.Children.Clear();
                    Content.Children.Add(new Help());
                    break;
                case "M4":
                    Close();
                    break;
                default:
                   
                    break;
            }
        }

        private void Open_OnClick(object sender, RoutedEventArgs e)
        {
            var openFileDialog1 = new OpenFileDialog
            {
                Filter = @"Pliki tekstowe|*.txt",
                Title = @"Wybierz plik tekstowy zawierający reprezentację grafu"
            };

            if (openFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                lastPath=openFileDialog1.InitialDirectory + openFileDialog1.FileName;
                graph = new GraphOld(lastPath);
            }
            else
            {
                MessageBox.Show("Nie wybrałeś żadnego pliku, spróbuj ponownie.");
            }
        }

        private void Generate_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + (sender as MenuItem).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        private void CPU1_OnClick(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(lastPath))
            {
                Graph g = Graph.ReadGraph(lastPath);
                Algorithm.ChromaticNumber.BuildingIndependentSets(g);
                int k = Algorithm.ChromaticNumber.FindChromaticNumber(g);
                MessageBox.Show(string.Format("Kolorowanie grafu jest nie większe niż: {0}", k));
            }
            else
            {
                MessageBox.Show("Jak byś podał graf na wejściu, to ja bym policzył :(");
            }
            
        }

        private void CPU2_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + (sender as MenuItem).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        private void GPU_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + (sender as MenuItem).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        private void AboutProgram_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + (sender as MenuItem).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        private void AboutAlgorithm_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + (sender as MenuItem).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        private void HowTo_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + (sender as MenuItem).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }

        private void Version_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Element '" + (sender as MenuItem).Header + "' nie jest obsługiwany w bieżącej wersji.");
        }
    }
}
