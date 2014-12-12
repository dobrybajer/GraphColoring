using System;
using GraphColoring.Structures;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using GraphColoring;
using GraphColoring.Algorithm;

namespace UnitTests
{
    [TestClass]
    public class TestCpuCSharp
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
