using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices; 

namespace UnitTests
{
    [TestClass]
    public class TestCpuCpp
    {
        [DllImport("C:\\Users\\Kamil\\Documents\\GitHub\\GraphColoring\\GraphColoring\\Debug\\GraphColoringCPU.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong Pow(int a, int n);

        [TestMethod]
        public void TestMethod1()
        {
            // arrange
            const int a = 2;
            const int n = 10;
            const ulong expected = 1024;

            // act
            var result = Pow(a, n);

            // assert
            Assert.AreEqual(expected, result, "Exponentation failure.");
        }
    }
}
