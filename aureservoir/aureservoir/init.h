/***************************************************************************/
/*!
 *  \file   init.h
 *
 *  \brief  initialization algorithms for Echo State Networks
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

#ifndef AURESERVOIR_INIT_H__
#define AURESERVOIR_INIT_H__

#include "utilities.h"

namespace aureservoir
{

/*!
 * \enum InitAlgorithm
 *
 * all possible initialization algorithms
 */
enum InitAlgorithm
{
  INIT_STD      //!< standard initialization, \sa class InitStd
};

/*!
 * \enum InitParameter
 *
 * possible parameters of the initialization algorithms
 * \note not every algorithm must use all of them !
 */
enum InitParameter
{
  CONNECTIVITY,     //!< connectivity of the reservoir weight matrix
  ALPHA,            //!< spectral radius of the reservoir weight matrix
  IN_CONNECTIVITY,  //!< connectivity of the input weight matrix
  IN_SCALE,         //!< scale input weight matrix random vaules
  IN_SHIFT,         //!< shift input weight matrix random vaules
  FB_CONNECTIVITY,  //!< connectivity of the feedback weight matrix
  FB_SCALE,         //!< scale feedback weight matrix random vaules
  FB_SHIFT,         //!< shift feedback weight matrix random vaules
  LEAKING_RATE,     //!< leaking rate for Leaky Integrator ESNs
  TIKHONOV_FACTOR,  //!< regularization factor for TrainRidgeReg
  DS_USE_CROSSCORR, //!< use simple cross-correlation for delay calculation
  DS_USE_GCC,       //!< use generalized cross-correlation for delay calculation
  DS_MAXDELAY,      //!< maximum delay for delay&sum readout
  IP_LEARNRATE,     //!< learnrate for Gaussian-IP reservoir adaptation
  IP_MEAN,          //!< desired mean for Gaussian-IP reservoir adaptation
  IP_VAR            //!< desired variance for Gaussian-IP reservoir adaptation
};

template <typename T> class ESN;

/*!
 * \class InitBase
 *
 * \brief abstract base class for initialization algorithms
 *
 * This class is an abstract base class for all different kinds of
 * initialization algorithms.
 * The idea behind this system is that the algorithms can be exchanged
 * at runtime (strategy pattern).
 *
 * Simply derive from this class if you want to add a new init algorithm.
 */
template <typename T>
class InitBase
{
 public:

  /// Constructor
  InitBase(ESN<T> *esn) { esn_=esn; }

  /// Destructor
  virtual ~InitBase() {}

  /// initialization algorithm
  virtual void init() throw(AUExcept) = 0;

 protected:

  /// checks if the init parameters have the right values
  virtual void checkInitParams() throw(AUExcept);

  /// allocates working data for algorithms
  virtual void allocateWorkData();

  /// reference to the data of the network
  ESN<T> *esn_;
};

/*!
 * \class InitStd
 *
 * \brief standard initialization as described in Jaeger's initial paper
 *
 * Initializes all matrices with normal distributed random values in
 * a specific connectivity.
 * Then it scales the weight matrix with the help of the largest
 * eigenvalue to the spectral radius alpha.
 */
template <typename T>
class InitStd : public InitBase<T>
{
  using InitBase<T>::esn_;

 public:
  InitStd(ESN<T> *esn) : InitBase<T>(esn) {}
  virtual ~InitStd() {}

  /// the algorithm
  virtual void init() throw(AUExcept);
};

} // end of namespace aureservoir

#endif // AURESERVOIR_INIT_H__
