namespace GraphColoring.Structures
{
    /// <summary>
    /// Klasa zawierająca stałe oraz metody używane w wielu miejscach całego projektu
    /// </summary>
    internal static class Common
    {
        #region Stałe reprezentujące konkretne wartości

        internal const double ToMb = 1048576 / (double)sizeof(int);

        #endregion

        #region Metody
        /// <summary>
        /// Funckja obliczająca liczbę kombinacji bez powtórzeń liczby n z liczby k.
        /// </summary>
        /// <param name="n">Górna liczba w Dwumianie Newtona.</param>
        /// <param name="k">Dolna liczba w Dwumianie Newtona.</param>
        /// <returns>Liczba kombinacji.</returns>
        internal static ulong Combination_n_of_k(ulong n, ulong k)
        {
            if (k > n) return 0;
            if (k == 0 || k == n) return 1;

            if (k * 2 > n) k = n - k;

            ulong r = 1;
            for (ulong d = 1; d <= k; ++d)
            {
                r *= n--;
                r /= d;
            }
            return r;
        }

        /// <summary>
        /// Funckja licząca potęgi liczby 2 (zarówno dodatnie jak i ujemne tj. ułamki)
        /// </summary>
        /// <param name="n">Wykładnik potęgi.</param>
        /// <returns>Liczba 2 podniesiona do odpowiedniej potęgi.</returns>
        internal static double Pow(int n)
        {
            double result = 1;

            if (n > 0)
            {
                while (n > 0)
                {
                    result *= 2;
                    n--;
                }
            }
            else
            {
                while (n < 0)
                {
                    result /= 2;
                    n++;
                }
            }

            return result;
        }
        #endregion
    }
}
