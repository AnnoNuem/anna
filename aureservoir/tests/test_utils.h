/***************************************************************************/
/*!
 *  \file   test_utils.h
 *
 *  \brief  some utility functions for the unit tests
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

#ifndef TEST_UTILS_H__
#define TEST_UTILS_H__

#include "aureservoir/utilities.h"

// use GNU scientific library
/// \todo remove dependency from gsl -> move init test to python
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>

/*!
 * inits the data with random values
 * @param data data array
 * @param size size of the data array
 */
template <typename T>
inline void data_set_random(T *data, int size, double min=-1., double max=1.)
{
  for(int i=0; i<size; ++i)
    data[i] = Rand<T>::uniform(min,max);
}

/*!
 * @param data data array
 * @param size size of the data array
 * @param eps double number under which all values are counted as zero
 * @return number of non-zero data elements in the array
 */
template <typename T>
inline int get_nonzero_elements(const T *data, int size, double eps)
{
  int result = 0;

  for(int i=0; i<size; ++i)
    if( std::abs( data[i] ) > eps ) result++;

  return result;
}

/*!
 * @param mtx a FLENS sparse matrix
 * @param eps double number under which all values are counted as zero
 * @return number of non-zero data elements in the matrix
 */
template <typename T>
inline int get_nonzero_elements_sparse(const T &mtx, double eps)
{
  int result = 0;

  typedef typename T::const_iterator It;
  for (It it=mtx.begin(); it!=mtx.end(); ++it) {
      if( std::abs( it->second ) > eps ) result++;
  }

  return result;
}

/*!
 * tests if all the data is between min and max
 * @return true if all data in [min|max]
 */
template <typename T>
inline int is_in_range(const T *data, int size, double min, double max)
{
  bool result = true;
  double min_, max_;

  if(min>max)
  { min_=max; max_=min; }
  else
  { min_=min; max_=max; }

  for(int i=0; i<size; ++i)
    if( data[i]>max_ || data[i]<min_ ) result=false;

  return result;
}

/*!
 * calculates the largest eigenvalue of a dense matrix
 * @param mtx FLENS matrix (must be quadratic !)
 * @return largest eigenvalue
 */
template <typename T>
inline double calc_largest_eigenvalue(const T &mtx)
{
  int n = mtx.numRows();

  // need to convert to double
  double data[n*n];
  for(int i=0; i<n*n; ++i)
    data[i] = mtx.data()[i];

  gsl_matrix_view m = gsl_matrix_view_array( data, n, n );
  gsl_vector_complex *eval = gsl_vector_complex_alloc( n );
  gsl_matrix_complex *evec = gsl_matrix_complex_alloc( n, n );
  gsl_eigen_nonsymmv_workspace *w = gsl_eigen_nonsymmv_alloc( n );

  gsl_eigen_nonsymmv(&m.matrix, eval, evec, w);
  gsl_eigen_nonsymmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_DESC);

  // get largest EW
  gsl_complex ew = gsl_vector_complex_get(eval, 0);

  gsl_eigen_nonsymmv_free (w);
  gsl_vector_complex_free(eval);
  gsl_matrix_complex_free(evec);

  return gsl_complex_abs(ew);
}

#endif // TEST_UTILS_H__
