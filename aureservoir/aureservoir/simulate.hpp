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

//! @name class SimBase Implementation
//@{

template <typename T>
SimBase<T>::SimBase(ESN<T> *esn)
{
  esn_=esn;
  reallocate();
}

template <typename T>
void SimBase<T>::reallocate()
{
  last_out_.resize(esn_->outputs_, 1);
  t_.resize(esn_->neurons_);
}

template <typename T>
void SimBase<T>::setBPCutoffConst(T f1, T f2) throw(AUExcept)
{
  std::string str = "SimBase::setBPCutoffConst: ";
  str += "this is not implemented in standard ESNs, ";
  str += "use e.g. SIM_BP !";

  throw AUExcept( str );
}

template <typename T>
void SimBase<T>::setBPCutoff(const typename ESN<T>::DEVector &f1,
                             const typename ESN<T>::DEVector &f2)
  throw(AUExcept)
{
  std::string str = "SimBase::setBPCutoff: ";
  str += "this is not implemented in standard ESNs, ";
  str += "use e.g. SIM_BP !";

  throw AUExcept( str );
}

template <typename T>
void SimBase<T>::setIIRCoeff(const typename DEMatrix<T>::Type &B,
                             const typename DEMatrix<T>::Type &A,
                             int series)
  throw(AUExcept)
{
  std::string str = "SimBase::setIIRCoeff: ";
  str += "this is not implemented in standard ESNs, ";
  str += "use e.g. SIM_FILTER !";

  throw AUExcept( str );
}

template <typename T>
void SimBase<T>::initDelayLine(int index,
                               const typename DEVector<T>::Type &initbuf)
  throw(AUExcept)
{
  std::string str = "SimBase::initDelayLines: ";
  str += "this is not implemented in standard ESNs, ";
  str += "use ESN with delay&sum readout, e.g. SIM_FILTER_DS !";

  throw AUExcept( str );
}

template <typename T>
typename DEMatrix<T>::Type SimBase<T>::getDelays()
  throw(AUExcept)
{
  std::string str = "SimBase::getDelays: ";
  str += "this is not implemented in standard ESNs, ";
  str += "use ESN with delay&sum readout, e.g. SIM_FILTER_DS !";

  throw AUExcept( str );
}

template <typename T>
typename DEVector<T>::Type &SimBase<T>::getDelayBuffer(int output, int nr)
    throw(AUExcept)
{
  std::string str = "SimBase::getDelayBuffer: ";
  str += "this is not implemented in standard ESNs, ";
  str += "use ESN with delay&sum readout, e.g. SIM_FILTER_DS !";

  throw AUExcept( str );
}

//@}
//! @name class SimStd Implementation
//@{

template <typename T>
void SimStd<T>::simulate(const typename ESN<T>::DEMatrix &in,
                         typename ESN<T>::DEMatrix &out)
{
  assert( in.numRows() == esn_->inputs_ );
  assert( out.numRows() == esn_->outputs_ );
  assert( in.numCols() == out.numCols() );
  assert( last_out_.numRows() == esn_->outputs_ );

  int steps = in.numCols();
  typename ESN<T>::DEMatrix::View
    Wout1 = esn_->Wout_(_,_(1, esn_->neurons_)),
    Wout2 = esn_->Wout_(_,_(esn_->neurons_+1, esn_->neurons_+esn_->inputs_));

  /// \todo direkte blas implementation hier machen ?
  ///       -> ist nicht immer dieses ganze error checking
  ///       -> vielleicht auch direkt floats übergeben ?

  /// \todo optimierte version für vektor/einzelwerte auch machen ?
  ///       -> das wirklich sinnvoll ?
  /// \todo optimierte version ohne Wback_ !
  /// \todo optimierte version ohne noise


  // First run with output from last simulation

  t_ = esn_->x_; // temp object needed for BLAS
  esn_->x_ = esn_->Win_*in(_,1) + esn_->W_*t_ + esn_->Wback_*last_out_(_,1);
  // add noise
  Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
  esn_->x_ += t_;
  esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

  // output = Wout * [x; in]
  last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,1);

  // output activation
  esn_->outputAct_( last_out_.data(),
                    last_out_.numRows()*last_out_.numCols() );
  out(_,1) = last_out_(_,1);


  // the rest

  for(int n=2; n<=steps; ++n)
  {
    t_ = esn_->x_; // temp object needed for BLAS
    esn_->x_ = esn_->Win_*in(_,n) + esn_->W_*t_ + esn_->Wback_*out(_,n-1);
    // add noise
    Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
    esn_->x_ += t_;
    esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

    // output = Wout * [x; in]
    last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,n);

    // output activation
    esn_->outputAct_( last_out_.data(),
                      last_out_.numRows()*last_out_.numCols() );
    out(_,n) = last_out_(_,1);
  }
}

//@}
//! @name class SimLI Implementation
//@{

template <typename T>
void SimLI<T>::simulate(const typename ESN<T>::DEMatrix &in,
                        typename ESN<T>::DEMatrix &out)
{
  assert( in.numRows() == esn_->inputs_ );
  assert( out.numRows() == esn_->outputs_ );
  assert( in.numCols() == out.numCols() );
  assert( last_out_.numRows() == esn_->outputs_ );

  int steps = in.numCols();
  typename ESN<T>::DEMatrix::View
    Wout1 = esn_->Wout_(_,_(1, esn_->neurons_)),
    Wout2 = esn_->Wout_(_,_(esn_->neurons_+1, esn_->neurons_+esn_->inputs_));

  /// \todo see SimStd


  // First run with output from last simulation

  t_ = esn_->x_; // temp object needed for BLAS
  esn_->x_ = (1. - esn_->init_params_[LEAKING_RATE])*t_ +
             esn_->Win_*in(_,1) + esn_->W_*t_ + esn_->Wback_*last_out_(_,1);
  // add noise
  Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
  esn_->x_ += t_;
  esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

  // output = Wout * [x; in]
  last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,1);

  // output activation
  esn_->outputAct_( last_out_.data(),
                    last_out_.numRows()*last_out_.numCols() );
  out(_,1) = last_out_(_,1);


  // the rest

  for(int n=2; n<=steps; ++n)
  {
    t_ = esn_->x_; // temp object needed for BLAS
    esn_->x_ = (1. - esn_->init_params_[LEAKING_RATE])*t_ +
               esn_->Win_*in(_,n) + esn_->W_*t_ + esn_->Wback_*out(_,n-1);
    // add noise
    Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
    esn_->x_ += t_;
    esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

    // output = Wout * [x; in]
    last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,n);

    // output activation
    esn_->outputAct_( last_out_.data(),
                      last_out_.numRows()*last_out_.numCols() );
    out(_,n) = last_out_(_,1);
  }
}

//@}
//! @name class SimBP Implementation
//@{

template <typename T>
void SimBP<T>::setBPCutoffConst(T f1, T f2) throw(AUExcept)
{
  typename ESN<T>::DEVector f1vec(esn_->neurons_);
  typename ESN<T>::DEVector f2vec(esn_->neurons_);

  std::fill_n( f1vec.data(), f1vec.length(), f1 );
  std::fill_n( f2vec.data(), f2vec.length(), f2 );
  filter_.setBPCutoff(f1vec,f2vec);
}

template <typename T>
void SimBP<T>::setBPCutoff(const typename ESN<T>::DEVector &f1,
                           const typename ESN<T>::DEVector &f2)
  throw(AUExcept)
{
  if( f1.length() != esn_->neurons_ )
    throw AUExcept("SimBP: f1 must have same length as reservoir neurons!");
  if( f2.length() != esn_->neurons_ )
    throw AUExcept("SimBP: f2 must have same length as reservoir neurons!");

  filter_.setBPCutoff(f1,f2);
}

template <typename T>
void SimBP<T>::simulate(const typename ESN<T>::DEMatrix &in,
                        typename ESN<T>::DEMatrix &out)
{
  assert( in.numRows() == esn_->inputs_ );
  assert( out.numRows() == esn_->outputs_ );
  assert( in.numCols() == out.numCols() );
  assert( last_out_.numRows() == esn_->outputs_ );

  int steps = in.numCols();
  typename ESN<T>::DEMatrix::View
    Wout1 = esn_->Wout_(_,_(1, esn_->neurons_)),
    Wout2 = esn_->Wout_(_,_(esn_->neurons_+1, esn_->neurons_+esn_->inputs_));

  /// \todo see SimStd

  // First run with output from last simulation

  // calc neuron activation
  t_ = esn_->x_;
  esn_->x_ = esn_->Win_*in(_,1) + esn_->W_*t_ + esn_->Wback_*last_out_(_,1);
  // add noise
  Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
  esn_->x_ += t_;
  esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

  // Bandpass Filtering
  filter_.calc(esn_->x_);

  // output = Wout * [x; in]
  last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,1);

  // output activation
  esn_->outputAct_( last_out_.data(),
                    last_out_.numRows()*last_out_.numCols() );
  out(_,1) = last_out_(_,1);


  // the rest

  for(int n=2; n<=steps; ++n)
  {
    t_ = esn_->x_; // temp object needed for BLAS
    esn_->x_ = esn_->Win_*in(_,n) + esn_->W_*t_ + esn_->Wback_*out(_,n-1);
    // add noise
    Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
    esn_->x_ += t_;
    esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

    // Bandpass Filtering
    filter_.calc(esn_->x_);

    // output = Wout * [x; in]
    last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,n);

    // output activation
    esn_->outputAct_( last_out_.data(),
                      last_out_.numRows()*last_out_.numCols() );
    out(_,n) = last_out_(_,1);
  }
}

//@}
//! @name class SimFilter Implementation
//@{

template <typename T>
void SimFilter<T>::setIIRCoeff(const typename DEMatrix<T>::Type &B,
                           const typename DEMatrix<T>::Type &A,
                           int series)
  throw(AUExcept)
{
  if( B.numRows() != esn_->neurons_ )
    throw AUExcept("SimFilter: B must have same rows as reservoir neurons!");
  if( A.numRows() != esn_->neurons_ )
    throw AUExcept("SimFilter: A must have same rows as reservoir neurons!");

  filter_.setIIRCoeff(B,A,series);
}

template <typename T>
void SimFilter<T>::simulate(const typename ESN<T>::DEMatrix &in,
                            typename ESN<T>::DEMatrix &out)
{
  assert( in.numRows() == esn_->inputs_ );
  assert( out.numRows() == esn_->outputs_ );
  assert( in.numCols() == out.numCols() );
  assert( last_out_.numRows() == esn_->outputs_ );

  int steps = in.numCols();
  typename ESN<T>::DEMatrix::View
    Wout1 = esn_->Wout_(_,_(1, esn_->neurons_)),
    Wout2 = esn_->Wout_(_,_(esn_->neurons_+1, esn_->neurons_+esn_->inputs_));

  /// \todo see SimStd

  // First run with output from last simulation

  // calc neuron activation
  t_ = esn_->x_;
  esn_->x_ = esn_->Win_*in(_,1) + esn_->W_*t_ + esn_->Wback_*last_out_(_,1);
  // add noise
  Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
  esn_->x_ += t_;
  esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

  // IIR Filtering
  filter_.calc(esn_->x_);

  // output = Wout * [x; in]
  last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,1);

  // output activation
  esn_->outputAct_( last_out_.data(),
                    last_out_.numRows()*last_out_.numCols() );
  out(_,1) = last_out_(_,1);


  // the rest

  for(int n=2; n<=steps; ++n)
  {
    t_ = esn_->x_; // temp object needed for BLAS
    esn_->x_ = esn_->Win_*in(_,n) + esn_->W_*t_ + esn_->Wback_*out(_,n-1);
    // add noise
    Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
    esn_->x_ += t_;
    esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

    // IIR Filtering
    filter_.calc(esn_->x_);

    // output = Wout * [x; in]
    last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,n);

    // output activation
    esn_->outputAct_( last_out_.data(),
                      last_out_.numRows()*last_out_.numCols() );
    out(_,n) = last_out_(_,1);
  }
}

//@}
//! @name class SimFilter2 Implementation
//@{

template <typename T>
void SimFilter2<T>::simulate(const typename ESN<T>::DEMatrix &in,
                            typename ESN<T>::DEMatrix &out)
{
  assert( in.numRows() == esn_->inputs_ );
  assert( out.numRows() == esn_->outputs_ );
  assert( in.numCols() == out.numCols() );
  assert( last_out_.numRows() == esn_->outputs_ );

  int steps = in.numCols();
  typename ESN<T>::DEMatrix::View
    Wout1 = esn_->Wout_(_,_(1, esn_->neurons_)),
    Wout2 = esn_->Wout_(_,_(esn_->neurons_+1, esn_->neurons_+esn_->inputs_));

  /// \todo see SimStd

  // First run with output from last simulation

  t_ = esn_->x_; // temp object needed for BLAS
  esn_->x_ = esn_->Win_*in(_,1) + esn_->W_*t_ + esn_->Wback_*last_out_(_,1);

  // IIR Filtering
  filter_.calc(esn_->x_);

  // add noise
  Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
  esn_->x_ += t_;
  esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

  // output = Wout * [x; in]
  last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,1);

  // output activation
  esn_->outputAct_( last_out_.data(),
                    last_out_.numRows()*last_out_.numCols() );
  out(_,1) = last_out_(_,1);


  // the rest

  for(int n=2; n<=steps; ++n)
  {
    t_ = esn_->x_; // temp object needed for BLAS
    esn_->x_ = esn_->Win_*in(_,n) + esn_->W_*t_ + esn_->Wback_*out(_,n-1);

    // IIR Filtering
    filter_.calc(esn_->x_);

    // add noise
    Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
    esn_->x_ += t_;
    esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

    // output = Wout * [x; in]
    last_out_(_,1) = Wout1*esn_->x_ + Wout2*in(_,n);

    // output activation
    esn_->outputAct_( last_out_.data(),
                      last_out_.numRows()*last_out_.numCols() );
    out(_,n) = last_out_(_,1);
  }
}

//@}
//! @name class SimFilterDS Implementation
//@{

template <typename T>
void SimFilterDS<T>::reallocate()
{
  last_out_.resize(esn_->outputs_, 1);
  t_.resize(esn_->neurons_);
  dellines_.resize( (esn_->neurons_+esn_->inputs_)*esn_->outputs_ );
  intmp_.resize(esn_->inputs_,1);
}

template <typename T>
void SimFilterDS<T>::initDelayLine(int index,
                               const typename DEVector<T>::Type &initbuf)
  throw(AUExcept)
{
  assert( index >= 0 );
  assert( index < esn_->outputs_*(esn_->inputs_+esn_->neurons_) );

  dellines_[index].initBuffer(initbuf);
}

template <typename T>
typename DEMatrix<T>::Type SimFilterDS<T>::getDelays() throw(AUExcept)
{
  typename DEMatrix<T>::Type del(esn_->outputs_,
                                 esn_->inputs_+esn_->neurons_);

  for(int i=1; i<=esn_->outputs_;++i) {
  for(int j=1; j<=esn_->inputs_+esn_->neurons_; ++j) {
    del(i,j) = T( dellines_[(i-1)*(esn_->neurons_+esn_->inputs_)+j-1].delay_ );
  } }

  return del;
}

template <typename T>
typename DEVector<T>::Type &SimFilterDS<T>::getDelayBuffer(int output, int nr)
    throw(AUExcept)
{
  return dellines_[output*(esn_->neurons_+esn_->inputs_)+nr].buffer_;
}

template <typename T>
void SimFilterDS<T>::simulate(const typename ESN<T>::DEMatrix &in,
                              typename ESN<T>::DEMatrix &out)
{
  assert( in.numRows() == esn_->inputs_ );
  assert( out.numRows() == esn_->outputs_ );
  assert( in.numCols() == out.numCols() );
  assert( last_out_.numRows() == esn_->outputs_ );

  int steps = in.numCols();
  typename ESN<T>::DEMatrix::View
    Wout1 = esn_->Wout_(_,_(1, esn_->neurons_)),
    Wout2 = esn_->Wout_(_,_(esn_->neurons_+1, esn_->neurons_+esn_->inputs_));

  /// \todo see SimStd

  // First run with output from last simulation

  // calc neuron activation
  t_ = esn_->x_;
  esn_->x_ = esn_->Win_*in(_,1) + esn_->W_*t_ + esn_->Wback_*last_out_(_,1);
  // add noise
  Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
  esn_->x_ += t_;
  esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

  // IIR Filtering
  filter_.calc(esn_->x_);

  // delay states and inputs for all individual outputs
  for(int i=1; i<=esn_->outputs_; ++i)
  {
    // delay x_ vector and store into t_
    for(int j=1; j<=esn_->neurons_; ++j)
      t_(j) =  dellines_[(i-1)*(esn_->neurons_+esn_->inputs_)+j-1].tic(
                          esn_->x_(j) );

    // store correct delayed input vector in intmp_
    for(int j=1; j<=esn_->inputs_; ++j)
      intmp_(j,1) = dellines_[ (i-1)*(esn_->neurons_+esn_->inputs_)
                                +esn_->neurons_+j-1 ].tic( in(j,1) );

    // calc  Wout * [x; in] for current output with delayed values
    last_out_(i,1) = Wout1(i,_)*t_ + Wout2(i,_)*intmp_(_,1);
  }

  // output activation
  esn_->outputAct_( last_out_.data(),
                    last_out_.numRows()*last_out_.numCols() );
  out(_,1) = last_out_(_,1);


  // the rest

  for(int n=2; n<=steps; ++n)
  {
    t_ = esn_->x_; // temp object needed for BLAS
    esn_->x_ = esn_->Win_*in(_,n) + esn_->W_*t_ + esn_->Wback_*out(_,n-1);
    // add noise
    Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
    esn_->x_ += t_;
    esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

    // IIR Filtering
    filter_.calc(esn_->x_);

    // delay states and inputs for all individual outputs
    for(int i=1; i<=esn_->outputs_; ++i)
    {
      // delay x_ vector and store into t_
      for(int j=1; j<=esn_->neurons_; ++j)
        t_(j) =  dellines_[(i-1)*(esn_->neurons_+esn_->inputs_)+j-1].tic(
                            esn_->x_(j) );

      // store correct delayed input vector in intmp_
      for(int j=1; j<=esn_->inputs_; ++j)
        intmp_(j,1) = dellines_[ (i-1)*(esn_->neurons_+esn_->inputs_)
                                  +esn_->neurons_+j-1 ].tic( in(j,n) );

      // calc  Wout * [x; in] for current output with delayed values
      last_out_(i,1) = Wout1(i,_)*t_ + Wout2(i,_)*intmp_(_,1);
    }

    // output activation
    esn_->outputAct_( last_out_.data(),
                      last_out_.numRows()*last_out_.numCols() );
    out(_,n) = last_out_(_,1);
  }
}

//@}
//! @name class SimSquare Implementation
//@{

template <typename T>
void SimSquare<T>::reallocate()
{
  last_out_.resize(esn_->outputs_, 1);
  t_.resize(esn_->neurons_);
  t2_.resize(esn_->neurons_);
  dellines_.resize( (esn_->neurons_+esn_->inputs_)*esn_->outputs_ );
  intmp_.resize(esn_->inputs_,1);
  insq_.resize(esn_->inputs_);
}

template <typename T>
void SimSquare<T>::simulate(const typename ESN<T>::DEMatrix &in,
                            typename ESN<T>::DEMatrix &out)
{
  assert( in.numRows() == esn_->inputs_ );
  assert( out.numRows() == esn_->outputs_ );
  assert( in.numCols() == out.numCols() );
  assert( last_out_.numRows() == esn_->outputs_ );

  // we have to resize Wout_ to also have connections for
  // the squared states
  esn_->Wout_.resize(esn_->outputs_, 2*(esn_->neurons_+esn_->inputs_));

  int steps = in.numCols();
  typename ESN<T>::DEMatrix::View
    Wout1 = esn_->Wout_(_,_(1, esn_->neurons_)),
    Wout2 = esn_->Wout_(_,_(esn_->neurons_+1,esn_->neurons_+esn_->inputs_)),
    Wout3 = esn_->Wout_(_,_(esn_->neurons_+esn_->inputs_+1,
                            2*esn_->neurons_+esn_->inputs_)),
    Wout4 = esn_->Wout_(_,_(2*esn_->neurons_+esn_->inputs_+1,
                            2*(esn_->neurons_+esn_->inputs_)));


  // First run with output from last simulation

  t_ = esn_->x_; // temp object needed for BLAS
  esn_->x_ = esn_->Win_*in(_,1) + esn_->W_*t_ + esn_->Wback_*last_out_(_,1);
  // add noise
  Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
  esn_->x_ += t_;
  esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

  // IIR Filtering
  filter_.calc(esn_->x_);

  // delay states and inputs for all individual outputs
  for(int i=1; i<=esn_->outputs_; ++i)
  {
    // delay x_ vector and store into t_
    for(int j=1; j<=esn_->neurons_; ++j)
      t_(j) =  dellines_[(i-1)*(esn_->neurons_+esn_->inputs_)+j-1].tic(
                          esn_->x_(j) );

    // store correct delayed input vector in intmp_
    for(int j=1; j<=esn_->inputs_; ++j)
      intmp_(j,1) = dellines_[ (i-1)*(esn_->neurons_+esn_->inputs_)
                                +esn_->neurons_+j-1 ].tic( in(j,1) );

    // calculate squared state version
    for(int j=1; j<=t_.length(); ++j)
      t2_(j) = pow( t_(j), 2 );
    // calculate squared input version
    for(int j=1; j<=insq_.length(); ++j)
      insq_(j) = pow( intmp_(j,1), 2 );

    // output = Wout * [x; in; x^2; in^2]
    last_out_(i,1) = Wout1(i,_)*t_ + Wout2(i,_)*intmp_(_,1)
                     + Wout3(i,_)*t2_ + Wout4(i,_)*insq_;
  }

  // output activation
  esn_->outputAct_( last_out_.data(),
                    last_out_.numRows()*last_out_.numCols() );
  out(_,1) = last_out_(_,1);


  // the rest

  for(int n=2; n<=steps; ++n)
  {
    t_ = esn_->x_; // temp object needed for BLAS
    esn_->x_ = esn_->Win_*in(_,n) + esn_->W_*t_ + esn_->Wback_*out(_,n-1);
    // add noise
    Rand<T>::uniform(t_, -1.*esn_->noise_, esn_->noise_);
    esn_->x_ += t_;
    esn_->reservoirAct_( esn_->x_.data(), esn_->x_.length() );

    // IIR Filtering
    filter_.calc(esn_->x_);

    // delay states and inputs for all individual outputs
    for(int i=1; i<=esn_->outputs_; ++i)
    {
      // delay x_ vector and store into t_
      for(int j=1; j<=esn_->neurons_; ++j)
        t_(j) =  dellines_[(i-1)*(esn_->neurons_+esn_->inputs_)+j-1].tic(
                            esn_->x_(j) );

      // store correct delayed input vector in intmp_
      for(int j=1; j<=esn_->inputs_; ++j)
        intmp_(j,1) = dellines_[ (i-1)*(esn_->neurons_+esn_->inputs_)
                                +esn_->neurons_+j-1 ].tic( in(j,n) );

      // calculate squared state version
      for(int j=1; j<=t_.length(); ++j)
        t2_(j) = pow( t_(j), 2 );
      // calculate squared input version
      for(int j=1; j<=insq_.length(); ++j)
        insq_(j) = pow( intmp_(j,1), 2 );

      // output = Wout * [x; in; x^2; in^2]
      last_out_(i,1) = Wout1(i,_)*t_ + Wout2(i,_)*intmp_(_,1)
                       + Wout3(i,_)*t2_ + Wout4(i,_)*insq_;
    }

    // output activation
    esn_->outputAct_( last_out_.data(),
                      last_out_.numRows()*last_out_.numCols() );
    out(_,n) = last_out_(_,1);
  }
}

//@}

} // end of namespace aureservoir
