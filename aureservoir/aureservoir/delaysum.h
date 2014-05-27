/***************************************************************************/
/*!
 *  \file   delaysum.h
 *
 *  \brief  utilities for delay and sum readout
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   March 2008
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

#ifndef AURESERVOIR_DELAYSUM_H__
#define AURESERVOIR_DELAYSUM_H__

#include "utilities.h"
#include <complex>
#include <math.h>
#include <fftw3.h>

namespace aureservoir
{

//! @name FFT routines using FFTW
//@{

/*!
 * calculates real fft with zero padding in double precision
 * @param x real input vector
 * @param X complex FFT output vector, will be resized to fftsize/2+1
 * @param fftsize fftsize, x will be zero-padded to this size
 */
void rfft(const DEVector<double>::Type &x,
          CDEVector<double>::Type &X, int fftsize);

/*!
 * calculates real fft with zero padding in single precision
 * @param x real input vector
 * @param X complex FFT output vector, will be resized to fftsize/2+1
 * @param fftsize fftsize, x will be zero-padded to this size
 */
void rfft(const DEVector<float>::Type &x,
          CDEVector<float>::Type &X, int fftsize);

/*!
 * calculates inverse real fft in double precision
 * @param X complex frequency domain input vector
 * @param x real IFFT output vector, will be resized to correct size
 */
void irfft(CDEVector<double>::Type &X, DEVector<double>::Type &x);

/*!
 * calculates inverse real fft in single precision
 * @param X complex frequency domain input vector
 * @param x real IFFT output vector, will be resized to correct size
 */
void irfft(CDEVector<float>::Type &X, DEVector<float>::Type &x);

//@}

/*!
 * \class CalcDelay
 * \brief template class for delay calculation
 */
template <typename T>
class CalcDelay
{
 public:
  /*!
   * calculates delay between x and y using the
   * generalized cross calculation (GCC)
   * @param X complex input vector1 in frequency domain
   * @param Y complex input vector2 in frequency domain
   * @param maxdelay maximum delay size for calculation
   * @param filter pre-whitening filter type:
   *               0 = standard cross correlation
   *               1 = phase transform (PHAT)
   * @return delay between the two signals
   */
  static int gcc(const typename CDEVector<T>::Type &X,
                 const typename CDEVector<T>::Type &Y,
                 int maxdelay=1000, int filter=0);
};

/*!
 * \class DelayLine
 * \brief template class for a signal delay line
 */
template <typename T>
class DelayLine
{
 public:

  DelayLine()
  { delay_=0; readpt_=1; }

  virtual ~DelayLine() {}

  /*!
   * allocates the delay line
   * @param initbuf are the initial values of the delayline
   *                the delay is the size of this vector
   *                if this vector is uninitialized the delay is 0
   */
  void initBuffer(const typename DEVector<T>::Type &initbuf);

  /*!
   * perform one step of the delay line
   * @param sample will be stored in ringbuffer and
   * @return delayed sample
   */
  T tic(T sample);

  /// assignment operator
  const DelayLine& operator= (const DelayLine<T>& src)
  {
    readpt_ = src.readpt_;
    delay_ = src.delay_;
    buffer_ = src.buffer_;
    return *this;
  }

  /// ringbuffer for the delay line
  typename DEVector<T>::Type buffer_;

  /// current readpointer in ringbuffer
  long readpt_;

  /// the delay of this delay line
  long delay_;
};

} // end of namespace aureservoir

#include <aureservoir/delaysum.hpp>

#endif // AURESERVOIR_DELAYSUM_H__
