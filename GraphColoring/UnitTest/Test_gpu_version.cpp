#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\GraphColoringGPU\Algorithm.cuh"

using namespace version_gpu;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest
{		
	TEST_CLASS(UnitTestGPU)
	{
	public:
		
		TEST_METHOD(Pow_Exponentiation_Basic)
		{
			Assert::AreEqual(170859375, Pow(15, 7), 0.0, L"B³¹d podczas standardowego potêgowania.", LINE_INFO());
		}

		TEST_METHOD(Pow_Exponentiation_Critical)
		{
			Assert::AreEqual(2147483648, Pow(2, 31), 0.0, L"B³¹d podczas potêgowania du¿ej liczby.", LINE_INFO());
			Assert::AreEqual(1, Pow(52345243, 0), 0.0, L"B³¹d podczas podnoszenia do potegi 0.", LINE_INFO());
		}

		TEST_METHOD(sgnPow_MinusOne_Fast_Exponantiation)
		{
			Assert::AreEqual(1, sgnPow(0), L"B³¹d podczas podnoszenia -1 do potêgi 0.", LINE_INFO());
			Assert::AreEqual(1, sgnPow(10), L"B³¹d podczas podnoszenia -1 do potêgi parzystej.", LINE_INFO());
			Assert::AreEqual(-1, sgnPow(11), L"B³¹d podczas podnoszenia -1 do potêgi nieparzystej.", LINE_INFO());
			Assert::AreEqual(1, sgnPow(1073741824), L"B³¹d podczas podnoszenia -1 do du¿ej potêgi.", LINE_INFO());
		}

		TEST_METHOD(BitCount_Counting_Subset_Elements_Basic)
		{
			Assert::AreEqual(3, BitCount(41), L"B³¹d podczas zliczania bitów w standardowej liczbie (1).", LINE_INFO());
			Assert::AreEqual(14, BitCount(523535), L"B³¹d podczas zliczania bitów w standardowej liczbie (2).", LINE_INFO());
			Assert::AreEqual(15, BitCount(107374183), L"B³¹d podczas zliczania bitów w du¿ej liczbie.", LINE_INFO());
		}

		TEST_METHOD(BitCount_Counting_Subset_Elements_Critical)
		{
			Assert::AreEqual(0, BitCount(0), L"B³¹d podczas zliczania bitów w liczbie 0.", LINE_INFO());
			Assert::AreEqual(1, BitCount(1073741824), L"B³¹d podczas zliczania bitów w maksymalnej (int) liczbie.", LINE_INFO());
		}

		TEST_METHOD(Combination_n_of_k_All) 
		{
			Assert::AreEqual((unsigned int)1, Combination_n_of_k(10, 0), L"B³¹d podczas zliczania kombinacji 0 elementowej.", LINE_INFO());
			Assert::AreEqual((unsigned int)1, Combination_n_of_k(10, 10), L"B³¹d podczas zliczania kombinacji n elementowej.", LINE_INFO());
			Assert::AreEqual((unsigned int)0, Combination_n_of_k(10, 11), L"B³¹d podczas zliczania kombinacji o z³ych parametrach.", LINE_INFO());
			Assert::AreEqual((unsigned int)3108105, Combination_n_of_k(28, 21), L"B³¹d podczas zliczania du¿ej iloœci kombinacji (1).", LINE_INFO());
			Assert::AreEqual((unsigned int)563921995, Combination_n_of_k(43, 34), L"B³¹d podczas zliczania du¿ej iloœci kombinacji (2).", LINE_INFO());
		}
	};
}