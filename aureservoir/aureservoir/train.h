/***************************************************************************/
/*!
 *  \file   train.h
 *
 *  \brief  training algorithms for Echo State Networks
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

#ifndef AURESERVOIR_TRAIN_H__
#define AURESERVOIR_TRAIN_H__

#include "utilities.h"
#include "delaysum.h"

namespace aureservoir
{

/*!
 * \enum TrainAlgorithm
 *
 * all possible training algorithms
 */
enum TrainAlgorithm
{
  TRAIN_PI,        //!< offline, pseudo inverse based \sa class TrainPI
  TRAIN_LS,        //!< offline least square algorithm, \sa class TrainLS
  TRAIN_RIDGEREG,  //!< with ridge regression, \sa class TrainRidgeReg
  TRAIN_DS_PI      //!< trains a delay&sum readout with PI \sa class TrainDSPI
};

template <typename T> class ESN;

/*!
 * \class TrainBase
 *
 * \brief abstract base class for training algorithms
 *
 * This class is an abstract base class for all different kinds of
 * training algorithms.
 * The idea behind this system is that the algorithms can be exchanged
 * at runtime (strategy pattern).
 *
 * Simply derive from this class if you want to add a new algorithm.
 */
template <typename T>
class TrainBase
{
 public:

  /// Constructor
  TrainBase(ESN<T> *esn) { esn_=esn; }

  /// Destructor
  virtual ~TrainBase() {}

  /*!
   * training algorithm
   *
   * @param in matrix of input values (inputs x timesteps)
   * @param out matrix of desired output values (outputs x timesteps)
   *            for teacher forcing
   * @param washout washout time in samples, used to get rid of the
   *                transient dynamics of the network starting state
   */
  virtual void train(const typename ESN<T>::DEMatrix &in,
                     const typename ESN<T>::DEMatrix &out,
                     int washout) throw(AUExcept) = 0;

 protected:

  /// check parameters
  void checkParams(const typename ESN<T>::DEMatrix &in,
                   const typename ESN<T>::DEMatrix &out,
                   int washout) throw(AUExcept);


  /// collect network states with simulation algorithm
  void collectStates(const typename ESN<T>::DEMatrix &in,
                     const typename ESN<T>::DEMatrix &out,
                     int washout);

  /// squares states for SIM_SQUARE
  void squareStates();

  /// frees allocated data for M and O
  void clearData()
  { M.resize(1,1); O.resize(1,1); }

  /// reference to the data of the network
  ESN<T> *esn_;

  /// matrix for network states and inputs over all timesteps
  typename ESN<T>::DEMatrix M;
  /// matrix for outputs over all timesteps
  typename ESN<T>::DEMatrix O;
};

/*!
 * \class TrainPI
 *
 * \brief offline trainig algorithm using the pseudo inverse
 *
 * Trains the ESN offline in two steps, as described in Jaeger's
 * "Tutorial on training recurrent neural networks"
 * \sa http://www.faculty.iu-bremen.de/hjaeger/pubs/ESNTutorial.pdf
 *
 * 1. teacher-forcing/sampling: collects the internal states and
 *    desired outputs <br/>
 * 2. computes output weights usings LAPACK's xGELSS routing, which
 *    performs a singular value decomposition and gets the 
 *    pseudo inverse
 *    \sa http://www.netlib.org/lapack/single/sgelss.f
 *
 * The difference to TrainLS is, that TrainPI is computationally
 * more expansive, but TrainLeastSquare can have stability problems.
 * \sa class TrainLeastSquare
 *
 * For a more mathematical description:
 * \sa http://en.wikipedia.org/wiki/Linear_least_squares
 * \sa http://www.netlib.org/lapack/lug/node27.html
 *
 * \example "slow_sine.py"
 */
template <typename T>
class TrainPI : public TrainBase<T>
{
  using TrainBase<T>::esn_;
  using TrainBase<T>::M;
  using TrainBase<T>::O;

 public:
  TrainPI(ESN<T> *esn) : TrainBase<T>(esn) {}
  virtual ~TrainPI() {}

  /// training algorithm
  virtual void train(const typename ESN<T>::DEMatrix &in,
                     const typename ESN<T>::DEMatrix &out,
                     int washout) throw(AUExcept);
};

/*!
 * \class TrainLS
 *
 * \brief offline trainig algorithm using the least square solution
 *
 * trains the ESN offline in two steps, as described in Jaeger's
 * "Tutorial on training recurrent neural networks"
 * \sa http://www.faculty.iu-bremen.de/hjaeger/pubs/ESNTutorial.pdf
 *
 * 1. teacher-forcing/sampling: collects the internal states and
 *    desired outputs <br/>
 * 2. computes output weights usings LAPACK's least square algorithm
 *    \sa http://www.netlib.org/lapack/single/sgels.f
 *
 * The differences to the TrainPI algorithm is explained here:
 * \sa class TrainPI
 */
template <typename T>
class TrainLS : public TrainBase<T>
{
  using TrainBase<T>::esn_;
  using TrainBase<T>::M;
  using TrainBase<T>::O;

 public:
  TrainLS(ESN<T> *esn) : TrainBase<T>(esn) {}
  virtual ~TrainLS() {}

  /// training algorithm
  virtual void train(const typename ESN<T>::DEMatrix &in,
                     const typename ESN<T>::DEMatrix &out,
                     int washout) throw(AUExcept);
};

/*!
 * \class TrainRidgeReg
 *
 * \brief offline algorithm with Ridge Regression / Tikhonov Regularization
 *
 * Trains an ESN offline in the same way as TrainLeastSquare or TrainPI.
 * \sa class TrainLeastSquare
 *
 * The difference compared to TrainLeastSquare is, that it uses an
 * regularization factor to calculate the output weigths, where it
 * tries to get them as small as possible.
 * This is called Ridge Regression or Tikhonov Regularization.
 *
 * The regularization factor can be set with the TIKHONOV_FACTOR parameter.
 * If TIKHONOV_FACTOR=0, one gets the unregularized least squares solution.
 * The higher the parameter, the stronger the smoothing/regularization effect.
 *
 * For ridge regression with ESNs see:
 * \sa http://scholarpedia.org/article/Echo_State_Network
 *
 * A general mathematical describtion can be found at:
 * \sa http://en.wikipedia.org/wiki/Tikhonov_regularization
 */
template <typename T>
class TrainRidgeReg : public TrainBase<T>
{
  using TrainBase<T>::esn_;
  using TrainBase<T>::M;
  using TrainBase<T>::O;

 public:
  TrainRidgeReg(ESN<T> *esn) : TrainBase<T>(esn) {}
  virtual ~TrainRidgeReg() {}

  /// training algorithm
  virtual void train(const typename ESN<T>::DEMatrix &in,
                     const typename ESN<T>::DEMatrix &out,
                     int washout) throw(AUExcept);
};

/*!
 * \class TrainDSPI
 *
 * \brief offline algorithm for delay&sum readout with PI
 *
 * Training like in class TrainPI, put an additional delay is learned
 * in the readout.
 * \sa class TrainPI
 * \sa class SimFilterDS
 *
 * See "Echo State Networks with Filter Neurons and a Delay&Sum Readout"
 * (Georg Holzmann, 2008).
 * \sa http://grh.mur.at/misc/ESNsWithFilterNeuronsAndDSReadout.pdf
 *
 * \example "singleneuron_sinosc.py"
 */
template <typename T>
class TrainDSPI : public TrainBase<T>
{
  using TrainBase<T>::esn_;
  using TrainBase<T>::M;
  using TrainBase<T>::O;

 public:
  TrainDSPI(ESN<T> *esn) : TrainBase<T>(esn) {}
  virtual ~TrainDSPI() {}

  /// training algorithm
  virtual void train(const typename ESN<T>::DEMatrix &in,
                     const typename ESN<T>::DEMatrix &out,
                     int washout) throw(AUExcept);
};

} // end of namespace aureservoir

#endif // AURESERVOIR_TRAIN_H__
