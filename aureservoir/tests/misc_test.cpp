/***************************************************************************/
/*!
 *  \file   misc_test.cpp
 *
 *  \brief  unit tests for random and denormal functions
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   Sept 2007
 *
 *   ::::_aureservoir_::::
 *   C++ library for analog reservoir computing neural networks
 *
 *   This library is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU Lesser General Public
 *   License as published by the Free Software Foundation; either
 *   version 2.1 of the License, or (at your option) any later version.
 *
 ***************************************************************************/

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

#include "aureservoir/aureservoir.h"
using namespace aureservoir;

template <typename T>
class MiscTest : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( MiscTest );
  CPPUNIT_TEST( randomTest );
  CPPUNIT_TEST( denormalTest );
  CPPUNIT_TEST( stringConvertTest );
  CPPUNIT_TEST_SUITE_END();

  typedef typename DEMatrix<T>::Type DEMatrix;
  typedef typename DEVector<T>::Type DEVector;

  public:
    void setUp (void)
    {
      A_.resize(4,4);
      x_.resize(4);

      Rand<T>::initSeed();
    }

    void tearDown (void)
    {}

  protected:
    void randomTest(void);
    void denormalTest(void);
    void stringConvertTest(void);

  private:
    DEMatrix A_;
    DEVector x_;
};

template <typename T>
void MiscTest<T>::randomTest(void)
{
  T test = Rand<T>::uniform();
  CPPUNIT_ASSERT( (test > -1) && (test < 1) );

  test = Rand<T>::uniform(2,3);
  CPPUNIT_ASSERT( (test > 2) && (test < 3) );

  test = Rand<T>::uniform(-4,-3);
  CPPUNIT_ASSERT( (test > -4) && (test < -3) );

  test = Rand<T>::uniform(0.2, 0.5);
  CPPUNIT_ASSERT( (test > 0.2) && (test < 0.5) );

  test = Rand<T>::uniform(-0.1, 0.1);
  CPPUNIT_ASSERT( (test > -0.1) && (test < 0.1) );
}

template <typename T>
void MiscTest<T>::denormalTest(void)
{
  set_denormal_flags();

  // initialization to 0
  CPPUNIT_ASSERT( A_(1,1) == 0 );
  CPPUNIT_ASSERT( A_(4,4) == 0 );
  CPPUNIT_ASSERT( x_(1) == 0 );
  CPPUNIT_ASSERT( x_(4) == 0 );

  denormals_add_dc( A_.data(), A_.numRows() * A_.numCols() );
  denormals_add_dc( x_.data(), x_.length() );

  // now with small DC offset
  CPPUNIT_ASSERT( A_(1,1) > 0 );
  CPPUNIT_ASSERT( A_(4,4) > 0 );
  CPPUNIT_ASSERT( x_(1) > 0 );
  CPPUNIT_ASSERT( x_(4) > 0 );
}

template <typename T>
void MiscTest<T>::stringConvertTest(void)
{
  // DOUBLE conversion

  T testval = 3.1456;
  std::string str("3.1456");
  CPPUNIT_ASSERT_DOUBLES_EQUAL( stringToDouble(str), testval, 1E-06 );

  testval = 1E-06;
  str = "1E-06";
  CPPUNIT_ASSERT_DOUBLES_EQUAL( stringToDouble(str), testval, 1E-06 );


  // INTEGER conversion

  str = "35";
  CPPUNIT_ASSERT( stringToInt(str) == 35 );

  str = "-335";
  CPPUNIT_ASSERT( stringToInt(str) == -335 );


  // wrong input test

  str = "testit";
  CPPUNIT_ASSERT_THROW( stringToDouble(str), AUExcept );
  CPPUNIT_ASSERT_THROW( stringToInt(str), AUExcept );

  str = "3.14A56";
  CPPUNIT_ASSERT_THROW( stringToDouble(str), AUExcept );
  CPPUNIT_ASSERT_THROW( stringToInt(str), AUExcept );

  str = "12a4";
  CPPUNIT_ASSERT_THROW( stringToDouble(str), AUExcept );
  CPPUNIT_ASSERT_THROW( stringToInt(str), AUExcept );

  str = "3.14";
  CPPUNIT_ASSERT_THROW( stringToInt(str), AUExcept );
}

// register float and double version in test suite
CPPUNIT_TEST_SUITE_REGISTRATION ( MiscTest<float> );
CPPUNIT_TEST_SUITE_REGISTRATION ( MiscTest<double> );
