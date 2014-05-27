/***************************************************************************/
/*!
 *  \file   init_test.cpp
 *
 *  \brief  unit tests for initialization algorithms
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

#include "test_utils.h"

/// \todo move the init tests to python

template <typename T>
class InitTest : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( InitTest );
  CPPUNIT_TEST( exceptionTest );
  CPPUNIT_TEST( sizeTest );
  CPPUNIT_TEST( connectivityTest );
  CPPUNIT_TEST( rangeTest );
  CPPUNIT_TEST( reservoirTest );
  CPPUNIT_TEST_SUITE_END();

  typedef typename DEMatrix<T>::Type DEMatrix;
  typedef typename DEVector<T>::Type DEVector;

  public:
    void setUp (void)
    { net_ = new ESN<T>; }

    void tearDown (void)
    { delete net_; }

  protected:
    void exceptionTest(void);
    void sizeTest(void);
    void connectivityTest(void);
    void rangeTest(void);
    void reservoirTest(void);

  private:
    ESN<T> *net_;
};

template <typename T>
void InitTest<T>::exceptionTest(void)
{
  net_->setInitParam(CONNECTIVITY, 0);
  CPPUNIT_ASSERT_THROW( net_->init(), AUExcept );

  net_->setInitParam(CONNECTIVITY, 0.2);
  net_->setInitParam(ALPHA, 0);
  CPPUNIT_ASSERT_THROW( net_->init(), AUExcept );

  net_->setInitParam(ALPHA, 0.8);
  net_->setInitParam(IN_CONNECTIVITY, -0.1);
  CPPUNIT_ASSERT_THROW( net_->init(), AUExcept );

  net_->setInitParam(IN_CONNECTIVITY, 0.6);
  net_->setInitParam(FB_CONNECTIVITY, -0.1);
  CPPUNIT_ASSERT_THROW( net_->init(), AUExcept );
}

template <typename T>
void InitTest<T>::sizeTest(void)
{
  int n, in, out;

  for(int i=0; i<20; ++i)
  {
    n = (int) Rand<float>::uniform(10,30);
    in = (int) Rand<float>::uniform(1,10);
    out = (int) Rand<float>::uniform(1,10);

    net_->setInitAlgorithm(INIT_STD);
    net_->setSize(n);
    net_->setInputs(in);
    net_->setOutputs(out);
    net_->setInitParam(CONNECTIVITY, 0.9);

    net_->init();

    CPPUNIT_ASSERT( net_->getWin().numRows() == n );
    CPPUNIT_ASSERT( net_->getWin().numCols() == in );
    CPPUNIT_ASSERT( net_->getWback().numRows() == n );
    CPPUNIT_ASSERT( net_->getWback().numCols() == out );
    CPPUNIT_ASSERT( net_->getWout().numRows() == out );
    CPPUNIT_ASSERT( net_->getWout().numCols() == n+in );
    CPPUNIT_ASSERT( net_->getX().length() == n );
  }
}

template <typename T>
void InitTest<T>::connectivityTest(void)
{
  int n, in, out, res, should;
  T conn, in_conn, fb_conn;

  try
  {
    for(int i=0; i<20; ++i)
    {
      n = (int) Rand<float>::uniform(30,60);
      in = (int) Rand<float>::uniform(1,30);
      out = (int) Rand<float>::uniform(1,30);
      in_conn = Rand<T>::uniform(0.01,0.98);
      fb_conn = Rand<T>::uniform(0.01,0.98);
      conn = Rand<T>::uniform(0.2,0.98);

      net_->setInitAlgorithm(INIT_STD);
      net_->setSize(n);
      net_->setInputs(in);
      net_->setOutputs(out);
      net_->setInitParam(CONNECTIVITY, conn);
      net_->setInitParam(IN_CONNECTIVITY, in_conn);
      net_->setInitParam(FB_CONNECTIVITY, fb_conn);

      net_->init();

      // Win
      res = get_nonzero_elements( net_->getWin().data(),
            net_->getWin().numRows() * net_->getWin().numCols(), 1E-12 );
      should = (int)(n*in*in_conn + 0.5);
      CPPUNIT_ASSERT( res == should );

      // Wback
      res = get_nonzero_elements( net_->getWback().data(),
            net_->getWback().numRows() * net_->getWback().numCols(), 1E-12 );
      should = (int)(n*out*fb_conn + 0.5);
      CPPUNIT_ASSERT( res == should );

      // Reservoir
      res = get_nonzero_elements_sparse( net_->getW(), 1E-12 );
      should = (int)(n*n*conn + 0.5);
      CPPUNIT_ASSERT( res == should );
    }
  }
  // we could get a zero eigenvalue, then an exception is thrown
  catch(AUExcept e)
  {
    std::cerr << "AUExcept: " << e.what() << std::endl;
    std::cerr << "This can happen with random values ... ;)" << std::endl;
  }
}

template <typename T>
void InitTest<T>::rangeTest(void)
{
  T in_scale, in_shift, fb_scale, fb_shift;
  double min, max;

  for(int i=0; i<20; ++i)
  {
    in_scale = Rand<T>::uniform(-3,3);
    in_shift = Rand<T>::uniform(-3,3);
    fb_scale = Rand<T>::uniform(-5,5);
    fb_shift = Rand<T>::uniform(-5,5);

    net_->setInitAlgorithm(INIT_STD);
    net_->setSize(10);
    net_->setInitParam(CONNECTIVITY, 0.9);
    net_->setInitParam(IN_CONNECTIVITY, 1);
    net_->setInitParam(FB_CONNECTIVITY, 1);
    net_->setInitParam(IN_SCALE, in_scale);
    net_->setInitParam(IN_SHIFT, in_shift);
    net_->setInitParam(FB_SCALE, fb_scale);
    net_->setInitParam(FB_SHIFT, fb_shift);

    net_->init();

    // Win
    min = -1*in_scale + in_shift;
    max = in_scale + in_shift;
    CPPUNIT_ASSERT( is_in_range( net_->getWin().data(),
              net_->getWin().numRows() * net_->getWin().numCols(),
              min, max ) );

    // Wback
    min = -1*fb_scale + fb_shift;
    max = fb_scale + fb_shift;
    CPPUNIT_ASSERT( is_in_range( net_->getWback().data(),
              net_->getWback().numRows() * net_->getWback().numCols(),
              min, max ) );
  }
}

template <typename T>
void InitTest<T>::reservoirTest(void)
{
  int n;
  T conn, alpha, max_ew;
  DEMatrix W;

  for(int i=0; i<20; ++i)
  {
    n = (int) Rand<float>::uniform(10,20);
    conn = Rand<T>::uniform(0.8,0.98);
    alpha = Rand<T>::uniform(0.1,0.98);

    net_->setInitAlgorithm(INIT_STD);
    net_->setSize(n);
    net_->setInitParam(CONNECTIVITY, conn);
    net_->setInitParam(ALPHA, alpha);

    net_->init();

    W = net_->getW();
    max_ew = calc_largest_eigenvalue( W );

    CPPUNIT_ASSERT_DOUBLES_EQUAL( max_ew, alpha, 0.05 );
  }
}

// register float and double version in test suite
CPPUNIT_TEST_SUITE_REGISTRATION ( InitTest<float> );
CPPUNIT_TEST_SUITE_REGISTRATION ( InitTest<double> );
