/***************************************************************************/
/*!
 *  \file   utilities.h
 *
 *  \brief  some global defines/includes and utility functions
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

#ifndef AURESERVOIR_UTILITIES_H__
#define AURESERVOIR_UTILITIES_H__

// external includes
#include <flens/flens.h>
#include <string>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
// #include <complex>

#include "auexcept.h"
#include "denormal.h"

namespace aureservoir
{

using std::string;
using std::complex;
using flens::_;

/// typedef trait class of a real sparse matrix with type T
template<typename T = float>
struct SPMatrix
{
  typedef flens::SparseGeMatrix<flens::CRS<T> > Type;
};

/// typedef trait class of a real dense matrix with type T
template<typename T = float>
struct DEMatrix
{
  typedef flens::GeMatrix<flens::FullStorage<T, flens::ColMajor> > Type;
};

/// typedef trait class of a real dense vector with type T
template<typename T = float>
struct DEVector
{
  typedef flens::DenseVector<flens::Array<T> > Type;
};

/// typedef trait class of a complex dense vector with type T
template<typename T = float>
struct CDEVector
{
  typedef flens::DenseVector<flens::Array< complex<T> > > Type;
};


/*!
 * \class Rand
 *
 * \brief template class for random number generation
 *
 * This class is used as random number generator for various
 * distributions.
 *
 * \todo add more distributions -> boost random number library
 */
template <typename T=float>
class Rand
{
 public:

  /// inits the random seed
  static void initSeed()
  { srand(time(0)); }

  /*!
   * generates a pseudo random number from a uniform distribution
   * @param min minimum value
   * @param max maximum value
   * @return value between [min|max)
   */
  static T uniform(float min=-1, float max=1)
  {
   T tmp = std::rand() / (T(RAND_MAX)+1); // between [0|1)
   return tmp*(max-min) + min;
  }

  /*!
   * generates a pseudo random number vector from a uniform distribution
   * @param vec fills this vector with rand values between [min|max)
   * @param min minimum value
   * @param max maximum value
   */
  static void uniform(typename DEVector<T>::Type &vec, float min=-1, float max=1)
  {
    for(int i=0; i<vec.length(); ++i)
      vec.data()[i] = std::rand() / (T(RAND_MAX)+1); // between [0|1)

    vec *= (max-min);
    vec += min;
  }
};

/// specialization for complex
// template<>
// static dcplx Rand<dcplx>::uniform(float min, float max)
// {
//   double real = std::rand() / (double(RAND_MAX)+1);
//   double imag = std::rand() / (double(RAND_MAX)+1);
//   dcplx tmp(real,imag);
//   return tmp*(dcplx(max)-dcplx(min)) + dcplx(min);
// }


/*!
 * converts a value from a string to a double
 * used to set parameters from strings
 */
inline double stringToDouble(const string& s)
  throw(AUExcept)
{
  std::istringstream is(s);
  double val;
  char c;

  // throw error if conversion is wrong or if we still have values
  if( !(is >> val) || is.get(c) )
    throw AUExcept("stringToDouble: could not convert parameter: " + s);

  return val;
}

/*!
 * converts a value from a string to a integer
 * used to set parameters from strings
 */
inline int stringToInt(const string& s)
  throw(AUExcept)
{
  std::istringstream is(s);
  int val;
  char c;

  // throw error if conversion is wrong or if we still have values
  if( !(is >> val) || is.get(c) )
    throw AUExcept("stringToInt: could not convert parameter: " + s);

  return val;
}

} // end of namespace aureservoir

#endif // AURESERVOIR_UTILITIES_H__
