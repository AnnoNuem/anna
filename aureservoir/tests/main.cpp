/***************************************************************************/
/*!
 *  \file   main.cpp
 *
 *  \brief  main file of unit tests for aureservoir
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

#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>

int main (int argc, char* argv[])
{
  // stores test results (listener)
  CppUnit::TestResult testresult;
  CppUnit::TestResultCollector collectedresults;
  testresult.addListener (&collectedresults);

  // get testsuit and add it to our TestRunner
  CppUnit::TestRunner tester;
  tester.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

  // run unit tests
 tester.run(testresult);

  // format results in a nice compiler friendly format
  CppUnit::CompilerOutputter compileroutputter( &collectedresults, 
                                                std::cout);
  compileroutputter.write ();

  // returns 0 on success
  return collectedresults.wasSuccessful () ? 0 : 1;
}
