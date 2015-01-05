#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\GraphColoringCPU\Algorithm.h"

using namespace version_cpu;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest
{		
	TEST_CLASS(UnitTestCPU)
	{
	public:
		
		TEST_METHOD(Pow_Exponentiation_Basic)
		{
			Assert::AreEqual(170859375, Pow(15, 7), 0.0, L"B³¹d podczas standardowego potêgowania.", LINE_INFO());
		}

		TEST_METHOD(Pow_Exponentiation_Critical)
		{
			Assert::AreEqual((unsigned int)2147483648, Pow(2, 31), L"B³¹d podczas potêgowania du¿ej liczby.", LINE_INFO());
			Assert::AreEqual((unsigned int)1, Pow(52345243, 0), L"B³¹d podczas podnoszenia do potegi 0.", LINE_INFO());
			Assert::AreEqual((unsigned int)0, Pow(-35345, 123), L"B³¹d podczas podnoszenia do potegi ujemnej liczby.", LINE_INFO());
			Assert::AreEqual((unsigned int)0, Pow(0, 123), L"B³¹d podczas podnoszenia do potegi liczby 0.", LINE_INFO());
			Assert::AreEqual((unsigned int)1, Pow(52345243, -234), L"B³¹d podczas podnoszenia do ujemnej potegi.", LINE_INFO());
			Assert::AreEqual((unsigned int)1234, Pow(1234, 1), L"B³¹d podczas podnoszenia do potegi 1.", LINE_INFO());
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
			Assert::AreEqual(0, BitCount(-456646), L"B³¹d podczas zliczania bitów w ujemnej liczbie.", LINE_INFO());
		}

		// poprawiæ sam¹ funkcjê - b³¹d przepe³nienia nawet dla 28 wierzcho³ków
		TEST_METHOD(Combination_n_of_k_All) 
		{
			Assert::AreEqual((unsigned int)1, Combination_n_of_k(10, 0), L"B³¹d podczas zliczania kombinacji 0 elementowej.", LINE_INFO());
			Assert::AreEqual((unsigned int)1, Combination_n_of_k(10, 10), L"B³¹d podczas zliczania kombinacji n elementowej.", LINE_INFO());
			Assert::AreEqual((unsigned int)0, Combination_n_of_k(10, 11), L"B³¹d podczas zliczania kombinacji o z³ych parametrach.", LINE_INFO());
			Assert::AreEqual((unsigned int)3108105, Combination_n_of_k(28, 21), L"B³¹d podczas zliczania du¿ej iloœci kombinacji (1).", LINE_INFO());
			Assert::AreEqual((unsigned int)563921995, Combination_n_of_k(43, 34), L"B³¹d podczas zliczania du¿ej iloœci kombinacji (2).", LINE_INFO());
		}

		TEST_METHOD(GetFirstBitPosition_Basic)
		{
			Assert::AreEqual((unsigned int)5, GetFirstBitPosition(41), L"B³¹d podczas wyznaczania pozycji pierwszego bitu dla liczby standardowej (1).", LINE_INFO());
			Assert::AreEqual((unsigned int)17, GetFirstBitPosition(132414), L"B³¹d podczas wyznaczania pozycji pierwszego bitu dla liczby standardowej (2).", LINE_INFO());
		}

		TEST_METHOD(GetFirstBitPosition_Critical)
		{
			Assert::AreEqual((unsigned int)0, GetFirstBitPosition(0), L"B³¹d podczas wyznaczania pozycji pierwszego bitu dla liczby 0.", LINE_INFO());
			Assert::AreEqual((unsigned int)31, GetFirstBitPosition(2147483648), L"B³¹d podczas wyznaczania pozycji pierwszego bitu dla liczby 2^31.", LINE_INFO());
			//Assert::AreEqual((unsigned int)32, GetFirstBitPosition(4294967296), L"B³¹d podczas wyznaczania pozycji pierwszego bitu dla liczby 2^32.", LINE_INFO());
		}

		TEST_METHOD(GetBitPosition_Basic)
		{
			Assert::AreEqual((unsigned int)1, GetBitPosition(12323, 1), L"B³¹d podczas wyznaczania sumy dla 1 bitu dla liczby 12323.", LINE_INFO());
			Assert::AreEqual((unsigned int)4131, GetBitPosition(12323, 4), L"B³¹d podczas wyznaczania sumy dla 4 bitów dla liczby 12323.", LINE_INFO());
		}

		TEST_METHOD(GetBitPosition_Critical)
		{
			Assert::AreEqual((unsigned int)0, GetBitPosition(-4636, 5), L"B³¹d podczas wyznaczania sumy dla z³ych danych (1 parametr).", LINE_INFO());
			Assert::AreEqual((unsigned int)0, GetBitPosition(12323, -456), L"B³¹d podczas wyznaczania sumy dla z³ych danych (1 parametr).", LINE_INFO());
			Assert::AreEqual((unsigned int)0, GetBitPosition(12323, 0), L"B³¹d podczas wyznaczania sumy dla 0 bitów dla liczby 12323.", LINE_INFO());
			Assert::AreEqual((unsigned int)0, GetBitPosition(0, 1), L"B³¹d podczas wyznaczania sumy dla 1 bitu dla liczby 0.", LINE_INFO());
			Assert::AreEqual((unsigned int)12323, GetBitPosition(12323, 5), L"B³¹d podczas wyznaczania sumy dla 5 bitów dla liczby 12323.", LINE_INFO());
			Assert::AreEqual((unsigned int)12323, GetBitPosition(12323, 123), L"B³¹d podczas wyznaczania sumy dla 123 bitów dla liczby 12323 (liczba ma tylko 5 bitów równych 1).", LINE_INFO());
		}

		TEST_METHOD(BuildingIndependentSets_TableVersion_All)
		{
			int independent_set [16] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5 };
			int vertices [10] = { 1, 2, 3, 0, 2, 0, 1, 3, 0, 2 };
			int offset [4] = { 3, 5, 8, 10 };
			int n = 4;
			int* wynik = BuildingIndependentSets_TableVersion(*&vertices, *&offset, n);

			for (int i = 0; i < (1 << n); i++)
				Assert::AreEqual(independent_set[i], wynik[i], L"B³¹d podczas tablicy zbiorów niezale¿nych dla wersji tablicowej.", LINE_INFO());
		}

		TEST_METHOD(BuildingIndependentSets_BitVersion_All)
		{
			int independent_set [16] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5 };
			int vertices_bit [4] = { 14, 5, 11, 5 };
			int n = 4;
			int* wynik = BuildingIndependentSets_BitVersion(*&vertices_bit, n);

			for (int i = 0; i < (1 << n); i++)
				Assert::AreEqual(independent_set[i], wynik[i], L"B³¹d podczas tablicy zbiorów niezale¿nych dla wersji bitowej.", LINE_INFO());
		}

		TEST_METHOD(FindChromaticNumber_All)
		{
			int vertices [10] = { 1, 2, 3, 0, 2, 0, 1, 3, 0, 2 };
			int vertices_bit [4] = { 14, 5, 11, 5 };
			int offset [4] = { 3, 5, 8, 10 };
			int n = 4;
			
			Assert::AreEqual(3, FindChromaticNumber(*&vertices, *&offset, n), L"B³¹d podczas wyznaczania kolorowalnoœci grafu dla wersji tablicowej (parametr domyœlny).", LINE_INFO());
			Assert::AreEqual(3, FindChromaticNumber(*&vertices, *&offset, n, 0), L"B³¹d podczas wyznaczania kolorowalnoœci grafu dla wersji tablicowej.", LINE_INFO());
			Assert::AreEqual(3, FindChromaticNumber(*&vertices_bit, *&offset, n, 1), L"B³¹d podczas wyznaczania kolorowalnoœci grafu dla wersji tablicowej.", LINE_INFO());
		}
	};
}