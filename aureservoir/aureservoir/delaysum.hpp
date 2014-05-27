/***************************************************************************/
/*!
 *  \file   delaysum.hpp
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

#include <assert.h>

namespace aureservoir
{

//! @name FFT routines
//@{

void rfft(const DEVector<double>::Type &x,
          CDEVector<double>::Type &X, int fftsize)
{
  // zero pad to fftsize
  DEVector<double>::Type xpad(fftsize);
  std::fill_n(xpad.data(), fftsize, 0);
  xpad(_(1,x.length())) = x;

  // calc FFTs
  fftw_plan p1;
  X.resize(fftsize/2+1);
  p1 = fftw_plan_dft_r2c_1d(fftsize, &xpad(1),
                            reinterpret_cast<fftw_complex*>( &X(1) ),
                            FFTW_ESTIMATE);
  fftw_execute(p1);
  fftw_destroy_plan(p1);
}

void rfft(const DEVector<float>::Type &x,
          CDEVector<float>::Type &X, int fftsize)
{
  // zero pad to fftsize
  DEVector<float>::Type xpad(fftsize);
  std::fill_n(xpad.data(), fftsize, 0);
  xpad(_(1,x.length())) = x;

  // calc FFTs
  fftwf_plan p1;
  X.resize(fftsize/2+1);
  p1 = fftwf_plan_dft_r2c_1d(fftsize, &xpad(1),
                            reinterpret_cast<fftwf_complex*>( &X(1) ),
                            FFTW_ESTIMATE);
  fftwf_execute(p1);
  fftwf_destroy_plan(p1);
}

void irfft(CDEVector<double>::Type &X, DEVector<double>::Type &x)
{
  int fftsize = 2*(X.length()-1);

  // calc IFFT
  x.resize(fftsize);
  fftw_plan p1 = fftw_plan_dft_c2r_1d(fftsize, 
                          reinterpret_cast<fftw_complex*>( &X(1) ),
                          &x(1), FFTW_ESTIMATE);
  fftw_execute(p1);
  fftw_destroy_plan(p1);
}

void irfft(CDEVector<float>::Type &X, DEVector<float>::Type &x)
{
  int fftsize = 2*(X.length()-1);

  // calc IFFT
  x.resize(fftsize);
  fftwf_plan p1 = fftwf_plan_dft_c2r_1d(fftsize, 
                          reinterpret_cast<fftwf_complex*>( &X(1) ),
                          &x(1), FFTW_ESTIMATE);
  fftwf_execute(p1);
  fftwf_destroy_plan(p1);
}

//@}
//! @name class CalcDelay
//@{

template <typename T>
int CalcDelay<T>::gcc(const typename CDEVector<T>::Type &X,
                      const typename CDEVector<T>::Type &Y, int maxdelay, int filter)
{
  assert( X.length() == Y.length() );

  typename CDEVector<T>::Type tmp( X.length() );

  int fftsize = 2*(X.length()-1);

  // multiplication in frequency domain
  for(int i=1; i<=X.length(); ++i)
    tmp(i) = conj( X(i) ) * Y(i);

  // calc phase transform if needed
  if( filter == 1 )
  {
    for(int i=1; i<=X.length(); ++i)
      if( std::abs(tmp(i)) != 0) tmp(i) = tmp(i) / std::abs(tmp(i));
  }

  // calc crosscorr with IFFT
  typename DEVector<T>::Type crosscorr;
  irfft(tmp,crosscorr);

  // calc delay
  for(int i=1; i<=fftsize; ++i)
    crosscorr(i) = std::abs( crosscorr(i) );
  int mdelay = (fftsize < maxdelay+1) ? fftsize : maxdelay+1;

  int delay = (int) ( std::max_element( crosscorr.data(),
                                        crosscorr.data()+mdelay )
                      - crosscorr.data() );

//   cout << crosscorr << "fftsize:" << fftsize << ",mdelay:" << mdelay 
//        << "maxdelay;" << maxdelay << endl;

  return delay;
}

//@}
//! @name class DelayLine
//@{

template <typename T>
void DelayLine<T>::initBuffer(const typename DEVector<T>::Type &initbuf)
{
  delay_ = initbuf.length();
  buffer_ = initbuf;
  readpt_ = 1;
}

template <typename T>
T DelayLine<T>::tic(T sample)
{
  // check for no delay
  if( delay_ == 0 )
    return sample;

  T outsample = buffer_( readpt_ );
  buffer_( readpt_ ) = sample;
  readpt_ = (readpt_ % delay_) + 1;

  return outsample;
}

//@}

} // end of namespace aureservoir
