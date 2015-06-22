using System;
using System.ComponentModel;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace GraphColoring.Structures
{
    [ValueConversion(typeof(bool), typeof(SolidColorBrush))]
    class Bindings : INotifyPropertyChanged, IValueConverter
    {
        #region Imlementacja interfejsu INotifyPropertyChanged

        /// <summary>
        /// Zdarzenie implementujace interfejs INotifyPropertyChanged.
        /// </summary>
        public event PropertyChangedEventHandler PropertyChanged;

        /// <summary>
        /// Funkcja wywołująca zdarzenie na zmiennej o podanej w parametrze nazwie.
        /// </summary>
        /// <param name="propertyName">Nazwa zmiennej.</param>
        protected void OnPropertyChanged(string propertyName)
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName)); 
        }

        #endregion

        #region Zmienne bazowe

        private bool _isEnabled;
        private Visibility _isVisible;
        private Visibility _isVisibleStopBtn;

        #endregion

        #region Właściwości

        /// <summary>
        /// Właściwość reagująca na zmianę wartości zmiennej na którą wskazuje.
        /// </summary>
        public bool EnabledValue
        {
            get
            {
                return _isEnabled;
            }
            set
            {
                _isEnabled = value; 
                OnPropertyChanged("EnabledValue");
            }
        }

        /// <summary>
        /// Właściwość reagująca na zmianę wartości zmiennej na którą wskazuje.
        /// </summary>
        public Visibility VisibilityValue
        {
            get
            {
                return _isVisible;
            }
            set
            {
                _isVisible = value; 
                OnPropertyChanged("VisibilityValue");
            }
        }

        /// <summary>
        /// Właściwość reagująca na zmianę wartości zmiennej na którą wskazuje.
        /// </summary>
        public Visibility StopButtonVisibilityValue
        {
            get
            {
                return _isVisibleStopBtn;
            }
            set
            {
                _isVisibleStopBtn = value;
                OnPropertyChanged("StopButtonVisibilityValue");
            }
        }

        #endregion

        #region Konwertery

        /// <summary>
        /// Funkcja konwertująca wartość bool na odpowiedni kolor tekstu wyświetlanego na przycisku. True - biały, false - szary.
        /// </summary>
        /// <param name="value">Wartość, na podstawie której zostanie dokonana konwersja</param>
        /// <param name="targetType"></param>
        /// <param name="parameter"></param>
        /// <param name="culture"></param>
        /// <returns>Kolor, w jakim zostanie wyświetlony tekst na przyciskach.</returns>
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return (bool)value ? new SolidColorBrush(Colors.White) : new SolidColorBrush(Colors.Gray);
        }

        /// <summary>
        /// Funkcja konwertująca wstecz - nie zaimplementowano.
        /// </summary>
        /// <param name="value">Wartość, na podstawie której zostanie dokonana konwersja</param>
        /// <param name="targetType"></param>
        /// <param name="parameter"></param>
        /// <param name="culture"></param>
        /// <returns>NULL.</returns>
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }

        #endregion
    }
}
