/***************************************************************************/
/*!
 *  \file   activations_test.cpp
 *
 *  \brief  unit tests for activation functions
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
class ActivationsTest : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( ActivationsTest );
  CPPUNIT_TEST( linearTest );
  CPPUNIT_TEST( tanhTest );
  CPPUNIT_TEST_SUITE_END();

  typedef typename DEMatrix<T>::Type DEMatrix;
  typedef typename DEVector<T>::Type DEVector;

  public:
    void setUp (void)
    {
      A_.resize(4,4);
      x_.resize(4);

      A_ = 1, 2, 3, 4,
           5, 6, 7, 8,
           9, 8, 7, 6,
           5, 4, 3, 2;

      x_ = 0.1, 0.2, 0.3, 0.4;
    }

    void tearDown (void)
    {}

  protected:
    void linearTest(void);
    void tanhTest(void);

  private:
    DEMatrix A_;
    DEVector x_;
};

template <typename T>
void ActivationsTest<T>::linearTest(void)
{
  act_linear( A_.data(), A_.numRows() * A_.numCols() );
  act_invlinear( x_.data(), x_.length() );

  CPPUNIT_ASSERT_DOUBLES_EQUAL( A_(1,1), 1, 1E-6 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( A_(4,4), 2, 1E-6 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( A_(2,3), 7, 1E-6 );

  CPPUNIT_ASSERT_DOUBLES_EQUAL( x_(1), 0.1, 1E-6 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( x_(4), 0.4, 1E-6 );
}

template <typename T>
void ActivationsTest<T>::tanhTest(void)
{
  act_tanh( A_.data(), A_.numRows() * A_.numCols() );
  act_invtanh( x_.data(), x_.length() );

  CPPUNIT_ASSERT_DOUBLES_EQUAL( A_(1,1), 0.76159, 1E-4 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( A_(4,4), 0.96403, 1E-4 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( A_(2,3), 1.00000, 1E-4 );

  CPPUNIT_ASSERT_DOUBLES_EQUAL( x_(1), 0.10034, 1E-4 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( x_(4), 0.42365, 1E-4 );
}

// register float and double version in test suite
CPPUNIT_TEST_SUITE_REGISTRATION ( ActivationsTest<float> );
CPPUNIT_TEST_SUITE_REGISTRATION ( ActivationsTest<double> );
