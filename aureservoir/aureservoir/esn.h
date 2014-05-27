/***************************************************************************/
/*!
 *  \file   esn.h
 *
 *  \brief  implements the base class of an echo state network
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

#ifndef AURESERVOIR_ESN_H__
#define AURESERVOIR_ESN_H__

#include <iostream>
#include <map>
#include <algorithm>

#include "utilities.h"
#include "activations.h"
#include "init.h"
#include "simulate.h"
#include "train.h"

namespace aureservoir
{

/*!
 * \class ESN
 *
 * \brief class for a basic Echo State Network
 *
 * This class implements a basic Echo State Network as described
 * in articles by Herbert Jaeger on the following page:
 * \sa http://www.scholarpedia.org/article/Echo_State_Network
 *
 * The template argument T can be float or double. Single Precision
 * (float) saves quite some computation time.
 *
 * The "echo state" approach looks at RNNs from a new angle. Large RNNs
 * are interpreted as "reservoirs" of complex, excitable dynamics.
 * Output units "tap" from this reservoir by linearly combining the
 * desired output signal from the rich variety of excited reservoir signals.
 * This idea leads to training algorithms where only the network-to-output 
 * connection weights have to be trained. This can be done with known,
 * highly efficient linear regression algorithms.
 * from \sa http://www.faculty.iu-bremen.de/hjaeger/esn_research.html
 *
 * For more information and a complete documentation of this library
 * see \sa http://aureservoir.sourceforge.net
 *
 * \example "esn_example.cpp"
 * \example "slow_sine.py"
 */
template <typename T = float>
class ESN
{
 public:

  /// typedef of a Parameter Map
  typedef std::map<InitParameter,T> ParameterMap;

  typedef typename SPMatrix<T>::Type SPMatrix;
  typedef typename DEMatrix<T>::Type DEMatrix;
  typedef typename DEVector<T>::Type DEVector;

  /// Constructor
  ESN();

  /// Copy Constructor
  ESN(const ESN<T> &src);

  /// assignement operator
  const ESN& operator= (const ESN<T>& src);

  /// Destructor
  ~ESN();

  //! @name Algorithm interface
  //@{

  /*!
   * Initialization Algorithm for an Echo State Network
   * \sa class InitBase
   */
  void init()
    throw(AUExcept)
  { init_->init(); }

  /*!
   * Reservoir Adaptation Algorithm Interface
   * At the moment this is only the Gaussian-IP reservoir adaptation method
   * for tanh neurons.
   * \sa "Adapting reservoirs to get Gaussian distributions" by David Verstraeten,
   *      Benjamin Schrauwen and Dirk Stroobandt
   *
   * @param in matrix of input values (inputs x timesteps),
   *           the reservoir will be adapted by this number of timesteps.
   * @return mean value of differences between all parameters before and after
   *         adaptation, can be used to see if learning still makes an progress.
   */
  double adapt(const DEMatrix &in)
    throw(AUExcept);

  /*!
   * Training Algorithm Interface
   * \sa class TrainBase
   *
   * @param in matrix of input values (inputs x timesteps)
   * @param out matrix of desired output values (outputs x timesteps)
   *            for teacher forcing
   * @param washout washout time in samples, used to get rid of the
   *                transient dynamics of the network starting state
   */
  inline void train(const DEMatrix &in, const DEMatrix &out, int washout)
    throw(AUExcept)
  { train_->train(in, out, washout); }

  /*!
   * Simulation Algorithm Interface
   * \sa class SimBase
   *
   * @param in matrix of input values (inputs x timesteps)
   * @param out matrix for output values (outputs x timesteps)
   */
  inline void simulate(const DEMatrix &in, DEMatrix &out)
  { sim_->simulate(in, out); }

   /*!
   * resets the internal state vector x of the reservoir to zero
   */
  void resetState()
  {
    std::fill_n( x_.data(), x_.length(), 0 );
    std::fill_n( sim_->last_out_.data(), outputs_, 0 );
  }

  //@}
  //! @name C-style Algorithm interface
  //@{

  /*!
   * C-style Reservoir Adaptation Algorithm Interface
   * (data will be copied into a FLENS matrix)
   * At the moment this is only the Gaussian-IP reservoir adaptation method
   * for tanh neurons.
   * \sa "Adapting reservoirs to get Gaussian distributions" by David Verstraeten,
   *      Benjamin Schrauwen and Dirk Stroobandt
   *
   * @param inmtx matrix of input values (inputs x timesteps),
   *              the reservoir will be adapted by this number of timesteps.
   * @return mean value of differences between all parameters before and after
   *         adaptation, can be used to see if learning still makes an progress.
   */
  double adapt(T *inmtx, int inrows, int incols) throw(AUExcept);

  /*!
   * C-style Training Algorithm Interface
   * (data will be copied into a FLENS matrix)
   * \sa class TrainBase
   *
   * @param inmtx input matrix in row major storage (usual C array)
   *              (inputs x timesteps)
   * @param outmtx output matrix in row major storage (outputs x timesteps)
   *               for teacher forcing
   * @param washout washout time in samples, used to get rid of the
   *                transient dynamics of the network starting state
   */
  inline void train(T *inmtx, int inrows, int incols,
                    T *outmtx, int outrows, int outcols,
                    int washout) throw(AUExcept);

  /*!
   * C-style Simulation Algorithm Interface with some additional
   * error checking.
   * (data will be copied into a FLENS matrix)
   * \sa class SimBase
   *
   * @param inmtx input matrix in row major storage (usual C array)
   *              (inputs x timesteps)
   * @param outmtx output matrix in row major storage (outputs x timesteps),
   *               \attention Data must be already allocated!
   */
  inline void simulate(T *inmtx, int inrows, int incols,
                       T *outmtx, int outrows, int outcols) throw(AUExcept);

  /*!
   * C-style Simulation Algorithm Interface, for single step simulation
   * \sa class SimBase
   * \todo see if we can do this in python without this additional method
   * @param inmtx input vector, size = inputs
   * @param outmtx output vector, size = outputs
   *               \attention Data must be already allocated!
   */
  inline void simulateStep(T *invec, int insize, T *outvec, int outsize)
    throw(AUExcept);

  //@}
  //! @name Additional Interface for Bandpass and IIR-Filter Neurons
  /// \todo rethink if this is consistent -> in neue klasse tun ?
  //@{

  /*!
   * Set lowpass/highpass cutoff frequencies for bandpass style neurons.
   " \sa class SimBP
   *
   * @param f1 vector with lowpass cutoff for all neurons (size = neurons)
   * @param f2 vector with highpass cutoffs (size = neurons)
   */
  void setBPCutoff(const DEVector &f1, const DEVector &f2) throw(AUExcept);

  /*!
   * Set lowpass/highpass cutoff frequencies for bandpass style neurons
   " (C-style Interface).
   *
   * @param f1 vector with lowpass cutoff for all neurons (size = neurons)
   * @param f2 vector with highpass cutoffs (size = neurons)
   */
  void setBPCutoff(T *f1vec, int f1size, T *f2vec, int f2size)
    throw(AUExcept);

  /*!
   * sets the IIR-Filter coefficients, like Matlabs filter object.
   *
   * @param B matrix with numerator coefficient vectors (m x nb)
   *          m  ... nr of parallel filters (neurons)
   *          nb ... nr of filter coefficients
   * @param A matrix with denominator coefficient vectors (m x na)
   *          m  ... nr of parallel filters (neurons)
   *          na ... nr of filter coefficients
   * @param seris nr of serial IIR filters, e.g. if series=2 the coefficients
   *              B and A will be divided in its half and calculated with
   *              2 serial IIR filters
   */
  void setIIRCoeff(const DEMatrix &B, const DEMatrix &A, int series=1)  
    throw(AUExcept);

  /*!
   * sets the IIR-Filter coefficients, like Matlabs filter object.
   *
   * @param B matrix with numerator coefficient vectors (m x nb)
   *          m  ... nr of parallel filters (neurons)
   *          nb ... nr of filter coefficients
   * @param A matrix with denominator coefficient vectors (m x na)
   *          m  ... nr of parallel filters (neurons)
   *          na ... nr of filter coefficients
   * @param seris nr of serial IIR filters, e.g. if series=2 the coefficients
   *              B and A will be divided in its half and calculated with
   *              2 serial IIR filters
   */
  void setIIRCoeff(T *bmtx, int brows, int bcols,
                   T *amtx, int arows, int acols,
                   int series=1) throw(AUExcept);

  //@}
  //! @name GET parameters
  //@{

  /*!
   * posts current parameters to stdout
   * \todo maybe return a outputstream (if stdout is not useful)
   *       or just use the << operator ?
   */
  void post();

  /// @return reservoir size (nr of neurons)
  int getSize() const { return neurons_; };
  /// @return nr of inputs to the reservoir
  int getInputs() const { return inputs_; };
  /// @return nr of outputs from the reservoir
  int getOutputs() const { return outputs_; };
  /// @return current noise level
  double getNoise() const { return noise_; }

  /*!
   * returns an initialization parametern from the parameter map
   * @param key the requested parameter
   * @return the value of the parameter
   */
  T getInitParam(InitParameter key) { return init_params_[key]; }

  /// @return initialization algorithm
  InitAlgorithm getInitAlgorithm() const
  { return static_cast<InitAlgorithm>(net_info_.at(INIT_ALG)); }
  /// @return training algorithm
  TrainAlgorithm getTrainAlgorithm() const
  { return static_cast<TrainAlgorithm>(net_info_.at(TRAIN_ALG)); }
  /// @return simulation algorithm
  SimAlgorithm getSimAlgorithm() const
  { return static_cast<SimAlgorithm>(net_info_.at(SIMULATE_ALG)); }

  /// @return reservoir activation function
  ActivationFunction getReservoirAct() const
  { return static_cast<ActivationFunction>(net_info_.at(RESERVOIR_ACT)); }
  /// @return output activation function
  ActivationFunction getOutputAct() const
  { return static_cast<ActivationFunction>(net_info_.at(OUTPUT_ACT)); }

  //@}
  //! @name GET internal data
  //@{

  /// @return input weight matrix (neurons x inputs)
  const DEMatrix &getWin() { return Win_; }
  /// @return reservoir weight matrix (neurons x neurons)
  const SPMatrix &getW() { return W_; }
  /// @return feedback (output to reservoir) weight matrix (neurons x outputs)
  const DEMatrix &getWback() { return Wback_; }
  /// @return output weight matrix (outputs x neurons+inputs)
  const DEMatrix &getWout() { return Wout_; }
  /// @return internal state vector x (size = neurons)
  const DEVector &getX() { return x_; }
  /**
   * query the trained delays in delay&sum readout \sa class SimFilterDS
   * @return matrix with delay form neurons+inputs to all outputs
   *         size = (output x neurons+inputs)
   */
  DEMatrix getDelays() throw(AUExcept) { return sim_->getDelays(); }

  //@}
  //! @name GET internal data C-style interface
  //@{

  /// get pointer to input weight matrix data and dimensions
  /// (neurons x inputs)
  /// \warning This data is in fortran style column major storage !
  void getWin(T **mtx, int *rows, int *cols);
  /// get pointer to feedback weight matrix data and dimensions 
  /// (neurons x outputs)
  /// \warning This data is in fortran style column major storage !
  void getWback(T **mtx, int *rows, int *cols);
  /// get pointer to output weight matrix data and dimensions
  /// (outputs x neurons+inputs)
  /// \warning This data is in fortran style column major storage !
  void getWout(T **mtx, int *rows, int *cols);
  /// get pointer to internal state vector x data and length
  void getX(T **vec, int *length);
  /*!
   * Copies data of the sparse reservoir weight matrix
   * into a dense C-style matrix.
   * \attention Memory of the C array must be allocated before!
   * @param wmtx pointer to matrix of size (neurons_ x neurons_)
   */
  void getW(T *wmtx, int wrows, int wcols) throw(AUExcept);
  /**
   * query the trained delays in delay&sum readout \sa class SimFilterDS
   * and copies the data into a C-style matrix
   * \attention Memory of the C array must be allocated before!
   * @param wmtx matrix with delay form neurons+inputs to all outputs
   *        size = (output x neurons+inputs)
   */
  void getDelays(T *wmtx, int wrows, int wcols) throw(AUExcept);

  //@}
  //! @name SET methods
  //@{

  /// set initialization algorithm
  void setInitAlgorithm(InitAlgorithm alg=INIT_STD)
    throw(AUExcept);
  /// set training algorithm
  void setTrainAlgorithm(TrainAlgorithm alg=TRAIN_PI)
    throw(AUExcept);
  /// set simulation algorithm
  void setSimAlgorithm(SimAlgorithm alg=SIM_STD)
    throw(AUExcept);

  /// set reservoir size (nr of neurons)
  void setSize(int neurons=10) throw(AUExcept);
  /// set nr of inputs to the reservoir
  void setInputs(int inputs=1) throw(AUExcept);
  /// set nr of outputs from the reservoir
  void setOutputs(int outputs=1) throw(AUExcept);

  /// set noise level for training/simulation algorithm
  /// @param noise with uniform distribution within [-noise|+noise]
  void setNoise(double noise) throw(AUExcept);

  /// set initialization parameter
  void setInitParam(InitParameter key, T value=0.);

  /// set reservoir activation function
  void setReservoirAct(ActivationFunction f=ACT_TANH) throw(AUExcept);
  /// set output activation function
  void setOutputAct(ActivationFunction f=ACT_LINEAR) throw(AUExcept);

  /*!
   * Additional method to set all parameters with string key-value
   * pairs, which can be used for bindings from other languages
   * @param param the parameter to set
   * @param value the value of that parameter
   */
//   void setParameter(string param, string value) throw(AUExcept);

  //@}
  //! @name SET internal data
  //@{

  /// set input weight matrix (neurons x inputs)
  void setWin(const DEMatrix &Win) throw(AUExcept);
  /// set reservoir weight matrix (neurons x neurons)
  void setW(const DEMatrix &W) throw(AUExcept);
  /// set feedback weight matrix (neurons x outputs)
  void setWback(const DEMatrix &Wback) throw(AUExcept);
  /// set output weight matrix (outputs x neurons+inputs)
  void setWout(const DEMatrix &Wout) throw(AUExcept);
  /// set internal state vector (size = neurons)
  void setX(const DEVector &x) throw(AUExcept);

  /*!
   * set last output, stored by the simulation algorithm
   * needed in singleStep simulation with feedback
   * @param last vector with length = outputs
   */
  void setLastOutput(const DEVector &last) throw(AUExcept);

  //@}
  //! @name SET internal data C-style interface
  //@{

  /*!
   * set input weight matrix C-style interface (neurons x inputs)
   * (data will be copied into a FLENS matrix)
   * @param inmtx pointer to win matrix in row major storage
   */
  void setWin(T *inmtx, int inrows, int incols) throw(AUExcept);

  /*!
   * set reservoir weight matrix C-style interface (neurons x neurons)
   * (data will be copied into a FLENS matrix)
   * @param inmtx pointer to a dense reservoir matrix in row major storage
   */
  void setW(T *inmtx, int inrows, int incols) throw(AUExcept);

  /*!
   * set feedback weight matrix C-style interface (neurons x outputs)
   * (data will be copied into a FLENS matrix)
   * @param inmtx pointer to wback matrix in row major storage
   */
  void setWback(T *inmtx, int inrows, int incols) throw(AUExcept);

  /*!
   * set output weight matrix C-style interface (outputs x neurons+inputs)
   * (data will be copied into a FLENS matrix)
   * @param inmtx pointer to wout matrix in row major storage
   */
  void setWout(T *inmtx, int inrows, int incols) throw(AUExcept);

  /*!
   * set internal state vector C-style interface (size = neurons)
   * (data will be copied into a FLENS matrix)
   * @param invec pointer to state vector
   */
  void setX(T *invec, int insize) throw(AUExcept);

  /*!
   * set last output, stored by the simulation algorithm
   * needed in singleStep simulation with feedback
   * @param last vector with size = outputs
   */
  void setLastOutput(T *last, int size) throw(AUExcept);

  //@}

 protected:

  /// function object for initialization algorithm
  InitBase<T> *init_;

  /// function object for training algorithm
  TrainBase<T> *train_;

  /// function object for simulation algorithm
  SimBase<T> *sim_;


  /// input weight matrix
  /// \todo also sparse version !?
  DEMatrix Win_;

  /// reservoir weight matrix
  SPMatrix W_;

  /// feedback (output to reservoir) weight matrix
  /// \todo also sparse version !?
  DEMatrix Wback_;

  /// output weight matrix (this will be trained)
  /// \todo also sparse version !?
  DEMatrix Wout_;

  /// internal state vector holding the current value of each
  /// neuron in the reservoir
  DEVector x_;


  /*!
   * activation function for the reservoir
   * \sa activations.h
   */
  void (*reservoirAct_)(T *data, int size);

  /*!
   * activation function for the outputs
   * \sa activations.h
   */
  void (*outputAct_)(T *data, int size);

  /*!
   * inverse activation function for the outputs
   * \sa activations.h
   */
  void (*outputInvAct_)(T *data, int size);


  /// nr of neurons in the reservoir (= reservoir size)
  int neurons_;
  /// nr of inputs to the reservoir
  int inputs_;
  /// nr of outputs from the reservoir
  int outputs_;

  /// noise level
  double noise_;


  /// parameter map for initialization arguments
  ParameterMap init_params_;

  /// enum used in the InfoMap to query network
  enum NetInfo
  {
    RESERVOIR_ACT,  //!< reservoir activation function
    OUTPUT_ACT,     //!< output activation function
    INIT_ALG,       //!< initialization algorithm
    TRAIN_ALG,      //!< training algorithm
    SIMULATE_ALG    //!< simulation algorithm
  };
  typedef std::map<NetInfo, int> InfoMap;

  /// holds strings of various ESN settings
  InfoMap net_info_;

  /// @return string of activation function enum
  string getActString(int act);
  /// @return string of init algorithm enum
  string getInitString(int alg);
  /// @return string of simulation algorithm enum
  string getSimString(int alg);
  /// @return string of training algorithm enum
  string getTrainString(int alg);


  //! @name algorithms are friends
  //@{
  friend class InitBase<T>;
  friend class InitStd<T>;
  friend class TrainBase<T>;
  friend class TrainPI<T>;
  friend class TrainLS<T>;
  friend class TrainRidgeReg<T>;
  friend class TrainDSPI<T>;
  friend class SimBase<T>;
  friend class SimStd<T>;
  friend class SimSquare<T>;
  friend class SimLI<T>;
  friend class SimBP<T>;
  friend class SimFilter<T>;
  friend class SimFilter2<T>;
  friend class SimFilterDS<T>;
  //@}
};

} // end of namespace aureservoir

#include <aureservoir/esn.hpp>
#include <aureservoir/init.hpp>
#include <aureservoir/simulate.hpp>
#include <aureservoir/train.hpp>

#endif // AURESERVOIR_ESN_H__
