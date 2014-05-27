/***************************************************************************/
/*!
 *  \file   simulate.hpp
 *
 *  \brief  simulation algorithms for Echo State Networks
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

#include <assert.h>

namespace aureservoir
{

//! @name class BPFilter Implementation
//@{

template <typename T>
void BPFilter<T>::setBPCutoff(const typename DEVector<T>::Type &f1,
                              const typename DEVector<T>::Type &f2)
  throw(AUExcept)
{
  if( f1.length() != f2.length() )
    throw AUExcept("BPFilter: f1 must be same size as f2!");

  int size = f1.length();

  // allocate data
  ema1_.resizeOrClear(size);
  ema2_.resizeOrClear(size);
  f1_.resize(size);
  f2_.resize(size);
  scale_.resize(size);

  f1_ = f1;
  f2_ = f2;

  // calculate scaler values:
  // scale = 1 + f2/f1;
  for(int i=1; i<=size; ++i)
    scale_(i) = 1 + f2_(i)/f1_(i);
}

template <typename T>
void BPFilter<T>::calc(typename DEVector<T>::Type &x)
{
  // Bandpass Filtering: new activation = ema1(act) - ema2(ema1(act))
  // ema1 += f1 * (activation - ema1)
  // ema2  += f2 * (ema1 - ema2)
  // activation = (ema1 - ema2) * scale
  for(int i=1; i<=ema1_.length(); ++i)
  {
    ema1_(i) += f1_(i) * ( x(i) - ema1_(i) );
    ema2_(i) += f2_(i) * ( ema1_(i) - ema2_(i) );
    x(i) = (ema1_(i) - ema2_(i)) * scale_(i);
  }
}

//@}
//! @name class IIRFilter Implementation
//@{

template <typename T>
void IIRFilter<T>::setIIRCoeff(const typename DEMatrix<T>::Type &B,
                   const typename DEMatrix<T>::Type &A)
  throw(AUExcept)
{
  if( B.numRows() != A.numRows() )
    throw AUExcept("BPFilter: B and A must have same rows!");

  int cols = B.numCols() > A.numCols() ? B.numCols() : A.numCols();
  int rows = A.numRows();

  // resize and clear old coefficients, so it is possible to set matrices
  // A and B which don't have the same size !
  A_.resizeOrClear( rows, cols );
  B_.resizeOrClear( rows, cols );
  std::fill_n( A_.data(), rows*cols, 0 );
  std::fill_n( B_.data(), rows*cols, 0 );

  S_.resizeOrClear(rows, cols-1);
  y_.resizeOrClear(rows);

  // divide coefficients through gains a[0]
  // and make assignment
  for(int i=1; i<=rows; ++i)
  {
    for(int j=1; j<=A.numCols(); ++j)
      A_(i,j) = A(i,j) / A(i,1);

    for(int j=1; j<=B.numCols(); ++j)
      B_(i,j) = B(i,j) / A(i,1);
  }
}

template <typename T>
void IIRFilter<T>::calc(typename DEVector<T>::Type &x)
{
  assert( x.length() == S_.numRows() );

  int neurons = S_.numRows();
  int coeffs = S_.numCols();

  for(int i=1; i<=neurons; ++i)
  {
    // calc new output
    y_(i) = B_(i,1) * x(i) + S_(i,1);

    // update internal storage
    for(int j=1; j<=(coeffs-1); ++j)
      S_(i,j) = B_(i,j+1) * x(i) - A_(i,j+1) * y_(i) + S_(i,j+1);

    S_(i,coeffs) = B_(i,coeffs+1) * x(i) - A_(i,coeffs+1) * y_(i);
  }

  x = y_;
}

//@}
//! @name class SerialIIRFilter Implementation
//@{

template <typename T>
void SerialIIRFilter<T>::setIIRCoeff(const typename DEMatrix<T>::Type &B,
                                     const typename DEMatrix<T>::Type &A,
                                     int series)
  throw(AUExcept)
{
  // simple init for 1 filter
  if(series==1)
  {
    IIRFilter<T> filter;
    filter.setIIRCoeff(B,A);
    filters_.push_back(filter);
    return;
  }

  // else split up matrix B and A in series
  int bsize = B.numCols();
  int asize = A.numCols();
  if( asize != bsize )
    throw AUExcept("SerialIIRFilter: serial filters must have same columns A and B!");

  /// \todo check for odd size !
  int nr = bsize / series;
  IIRFilter<T> filter;
  typename DEMatrix<T>::Type a,b;

  for(int i=1; i<=series; ++i)
  {
    a = A( _, _((i-1)*nr+1, i*nr) );
    b = B( _, _((i-1)*nr+1, i*nr) );
    filter.setIIRCoeff(b,a);
    filters_.push_back(filter);
  }
}

template <typename T>
void SerialIIRFilter<T>::calc(typename DEVector<T>::Type &x)
{
  int size = filters_.size();
  for(int i=0; i<size; ++i)
    filters_[i].calc( x );
}

//@}

} // end of namespace aureservoir
