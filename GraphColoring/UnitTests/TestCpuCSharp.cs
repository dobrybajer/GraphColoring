using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests
{
    [TestClass]
    public class TestGpuCpp
    {
        [TestMethod]
        public void ChromaticNumber_Pow()
        {
            // arrange
            const double a = 2;
            const double n = 10;
            const double expected = 1024;
        
            // act
            var result = Math.Pow(a, n);

            // assert
            Assert.AreEqual(expected, result, "Exponentation failure.");
        }
    }
}
