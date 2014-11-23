using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace GraphColoring
{
    class FileReader
    {
        public async void ReadFileButton_Click(object sender, RoutedEventArgs e)
        {
            string text;
            try
            {
                using (StreamReader sr = new StreamReader("C:/Users/Kamil/Documents/GitHub/GraphColoring/GraphColoring/GraphColoring/Files/GraphExample.txt"))
                {
                    String line = await sr.ReadToEndAsync();
                    
                }
            }
            catch (Exception ex)
            {
                text = "Could not read the file";
            }
            

        }
    }

}
