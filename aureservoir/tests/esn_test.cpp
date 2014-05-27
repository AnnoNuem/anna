/***************************************************************************/
/*!
 *  \file   esn_test.cpp
 *
 *  \brief  unit tests for the ESN class
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

template <typename T>
class ESNTest : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( ESNTest );
  CPPUNIT_TEST( exceptionTest );
  CPPUNIT_TEST_SUITE_END();

  public:
    void setUp (void)
    { net_ = new ESN<T>; }

    void tearDown (void)
    { delete net_; }

  protected:
    void exceptionTest(void);

  private:
    ESN<T> *net_;
};

template <typename T>
void ESNTest<T>::exceptionTest(void)
{
  CPPUNIT_ASSERT_THROW( net_->setInputs(0), AUExcept );
  CPPUNIT_ASSERT_THROW( net_->setInputs(-4), AUExcept );

  CPPUNIT_ASSERT_THROW( net_->setSize(0), AUExcept );
  CPPUNIT_ASSERT_THROW( net_->setSize(-3), AUExcept );

  CPPUNIT_ASSERT_THROW( net_->setOutputs(0), AUExcept );
  CPPUNIT_ASSERT_THROW( net_->setOutputs(-4), AUExcept );

  typename ESN<T>::DEMatrix A(5,3);
  CPPUNIT_ASSERT_THROW( net_->setWout(A), AUExcept );
}

// register float and double version in test suite
CPPUNIT_TEST_SUITE_REGISTRATION ( ESNTest<float> );
CPPUNIT_TEST_SUITE_REGISTRATION ( ESNTest<double> );
