using System.ComponentModel;
using System.Windows;

namespace GraphColoring.Structures
{
    class Bindings : INotifyPropertyChanged
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

        #endregion
    }
}
