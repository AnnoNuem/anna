/***************************************************************************/
/*!
 *  \file   train.hpp
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

namespace aureservoir
{

//! @name class TrainBase Implementation
//@{

template <typename T>
void TrainBase<T>::checkParams(const typename ESN<T>::DEMatrix &in,
                               const typename ESN<T>::DEMatrix &out,
                               int washout)
  throw(AUExcept)
{
  if( in.numCols() != out.numCols() )
    throw AUExcept("TrainBase::train: input and output must be same column size!");
  if( in.numRows() != esn_->inputs_ )
    throw AUExcept("TrainBase::train: wrong input row size!");
  if( out.numRows() != esn_->outputs_ )
    throw AUExcept("TrainBase::train: wrong output row size!");

  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_SQUARE )
  {
    if( (in.numCols()-washout) < esn_->neurons_+esn_->inputs_ )
    throw AUExcept("TrainBase::train: too few training data!");
  }
  else
  {
    if( (in.numCols()-washout) < 2*(esn_->neurons_+esn_->inputs_) )
    throw AUExcept("TrainBase::train: too few training data!");
  }

  /// \todo check also for the right algorithm combination
  ///       -> or better do that in init()

  // reallocate data buffer for simulation algorithm
  esn_->sim_->reallocate();
}

template <typename T>
void TrainBase<T>::collectStates(const typename ESN<T>::DEMatrix &in,
                                 const typename ESN<T>::DEMatrix &out,
                                 int washout)
{
  int steps = in.numCols();

  // collects output of all timesteps in O
  O.resize(steps-washout, esn_->outputs_);

  // collects reservoir activations and inputs of all timesteps in M
  // (for squared algorithm we need a bigger matrix)
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_SQUARE )
    M.resize(steps-washout, esn_->neurons_+esn_->inputs_);
  else
    M.resize(steps-washout, 2*(esn_->neurons_+esn_->inputs_));


  typename ESN<T>::DEMatrix sim_in(esn_->inputs_ ,1),
                            sim_out(esn_->outputs_ ,1);
  for(int n=1; n<=steps; ++n)
  {
    sim_in(_,1) = in(_,n);
    esn_->simulate(sim_in, sim_out);

    // for teacherforcing with feedback in single step simulation
    // we need to set the correct last output
    esn_->sim_->last_out_(_,1) = out(_,n);

//     std::cout << esn_->x_ << std::endl;

    // store internal states, inputs and outputs after washout
    if( n > washout )
    {
      M(n-washout,_(1,esn_->neurons_)) = esn_->x_;
      M(n-washout,_(esn_->neurons_+1,esn_->neurons_+esn_->inputs_)) =
      sim_in(_,1);
    }
  }

  // collect desired outputs
  O = flens::transpose( out( _,_(washout+1,steps) ) );
}

template <typename T>
void TrainBase<T>::squareStates()
{
  // add additional squared states and inputs
  /// \todo vectorize that
  int Msize = esn_->neurons_+esn_->inputs_;
  int Mrows = M.numRows();
  for(int i=1; i<=Mrows; ++i) {
  for(int j=1; j<=Msize; ++j) {
    M(i,j+Msize) = pow( M(i,j), 2 );
  } }
}

//@}
//! @name class TrainPI Implementation
//@{

template <typename T>
void TrainPI<T>::train(const typename ESN<T>::DEMatrix &in,
                       const typename ESN<T>::DEMatrix &out,
                       int washout)
  throw(AUExcept)
{
  this->checkParams(in,out,washout);

  // 1. teacher forcing, collect states
  this->collectStates(in,out,washout);

  // add additional squared states when using SIM_SQUARE
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] == SIM_SQUARE )
    this->squareStates();


  // 2. offline weight computation

  // undo output activation function
  esn_->outputInvAct_( O.data(), O.numRows()*O.numCols() );

  // calc weights with pseudo inv: Wout_ = (M^-1) * O
  flens::lss( M, O );
  esn_->Wout_ = flens::transpose( O(_( 1, M.numCols() ),_) );

  this->clearData();
}

//@}
//! @name class TrainLS Implementation
//@{

template <typename T>
void TrainLS<T>::train(const typename ESN<T>::DEMatrix &in,
                       const typename ESN<T>::DEMatrix &out,
                       int washout)
  throw(AUExcept)
{
  this->checkParams(in,out,washout);

  // 1. teacher forcing, collect states
  this->collectStates(in,out,washout);

  // add additional squared states when using SIM_SQUARE
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] == SIM_SQUARE )
    this->squareStates();


  // 2. offline weight computation

  // undo output activation function
  esn_->outputInvAct_( O.data(), O.numRows()*O.numCols() );

  // calc weights with least square solver: Wout_ = (M^-1) * O
  flens::ls( flens::NoTrans, M, O );
  esn_->Wout_ = flens::transpose( O(_( 1, M.numCols() ),_) );

  this->clearData();
}

//@}
//! @name class TrainRidgeReg Implementation
//@{

template <typename T>
void TrainRidgeReg<T>::train(const typename ESN<T>::DEMatrix &in,
                       const typename ESN<T>::DEMatrix &out,
                       int washout)
  throw(AUExcept)
{
  this->checkParams(in,out,washout);

  // 1. teacher forcing, collect states
  this->collectStates(in,out,washout);

  // add additional squared states when using SIM_SQUARE
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] == SIM_SQUARE )
    this->squareStates();


  // 2. offline weight computation

  // undo output activation function
  esn_->outputInvAct_( O.data(), O.numRows()*O.numCols() );


  // calc weights with ridge regression (.T = transpose):
  // Wout = ( (M.T*M + alpha^2*I)^-1 *M.T * O )

  // get regularization factor and square it
  T alpha = pow(esn_->init_params_[TIKHONOV_FACTOR],2);

  // temporal objects
  typename ESN<T>::DEMatrix T1(esn_->neurons_+esn_->inputs_,
                               esn_->neurons_+esn_->inputs_);
  flens::DenseVector<flens::Array<int> > t2( M.numCols() );

  // M.T * M
  T1 = flens::transpose(M)*M;

  // ans + alpha^2*I
  for(int i=1; i<=T1.numRows(); ++i)
    T1(i,i) += alpha;

  // calc inverse: (ans)^-1
  flens::trf(T1, t2);
  flens::tri(T1, t2);

  // ans * M.T
  esn_->Wout_ = T1 * flens::transpose(M);

  // ans * O
  T1 = esn_->Wout_ * O;

  // result = ans.T
  esn_->Wout_ = flens::transpose(T1);


  this->clearData();
}

//@}
//! @name class TrainDSPI Implementation
//@{

template <typename T>
void TrainDSPI<T>::train(const typename ESN<T>::DEMatrix &in,
                       const typename ESN<T>::DEMatrix &out,
                       int washout)
  throw(AUExcept)
{
  this->checkParams(in,out,washout);


  // 1. teacher forcing, collect states

  int steps = in.numCols();

  // collects output of all timesteps in O
  O.resize(steps-washout, 1);

  // collects reservoir activations and inputs of all timesteps in M
  M.resize(steps, esn_->neurons_+esn_->inputs_);

  typename ESN<T>::DEMatrix sim_in(esn_->inputs_ ,1),
                            sim_out(esn_->outputs_ ,1);
  for(int n=1; n<=steps; ++n)
  {
    sim_in(_,1) = in(_,n);
    esn_->simulate(sim_in, sim_out);

    // for teacherforcing with feedback in single step simulation
    // we need to set the correct last output
    esn_->sim_->last_out_(_,1) = out(_,n);

    // store internal states, inputs and outputs
    M(n,_(1,esn_->neurons_)) = esn_->x_;
    M(n,_(esn_->neurons_+1,esn_->neurons_+esn_->inputs_)) =
    sim_in(_,1);
  }


  // 2. delay calculation for delay&sum readout

  // check for right simulation algorithm
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_FILTER_DS &&
      esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_SQUARE )
    throw AUExcept("TrainDSPI::train: you need to use SIM_FILTER_DS or SIM_SQUARE for this training algorithm!");

  // get maxdelay
  int maxdelay;
  if( esn_->init_params_.find(DS_MAXDELAY) == esn_->init_params_.end() )
  {
    // set maxdelay to 0 if we have squared state updates
    if( esn_->net_info_[ESN<T>::SIMULATE_ALG] == SIM_SQUARE )
      maxdelay = 0;
    else
      maxdelay = 1000;
  }
  else
    maxdelay = (int) esn_->init_params_[DS_MAXDELAY];

  // see if we use GCC or simple crosscorr, standard is GCC
  int filter;
  if( esn_->init_params_.find(DS_USE_CROSSCORR) == esn_->init_params_.end() )
    filter = 1;
  else
    filter = 0;

  // delay calculation

  int delay = 0;
  int fftsize = (int) pow( 2, ceil(log(steps)/log(2)) ); // next power of 2
  typename CDEVector<T>::Type X,Y;
  typename DEVector<T>::Type x,y,rest;
  typename DEMatrix<T>::Type T1(1,esn_->neurons_+esn_->inputs_);
  typename DEMatrix<T>::Type Mtmp(M.numRows(),M.numCols()); /// \todo memory !!!

  for(int i=1; i<=esn_->outputs_; ++i)
  {
    // calc FFT of target vector
    y = out(i,_);
    rfft( y, Y, fftsize );

    // calc delays to reservoir neurons and inputs
    for(int j=1; j<=esn_->neurons_+esn_->inputs_; ++j)
    {
      // calc FFT of neuron/input vector
      x = M(_,j);
      rfft( x, X, fftsize );

      // calc delay with GCC
      delay = CalcDelay<T>::gcc(X,Y,maxdelay,filter);

      if( delay != 0 )
      {
        // shift signal the right amount
        rest = M( _(M.numRows()-delay+1,M.numRows()), j );
        Mtmp( _(1,delay), j ) = 0.;
        Mtmp( _(delay+1,M.numRows()), j ) = M( _(1,M.numRows()-delay), j );

        // init delay lines with the rest of the buffer
        esn_->sim_->initDelayLine((i-1)*(esn_->neurons_+esn_->inputs_)+j-1, rest);
      }
      else
        Mtmp(_,j) = M(_,j);
    }


    // 3. offline weight computation for each output extra

    // collect desired outputs
    O(_,1) = out( i ,_(washout+1,steps) );

    // undo output activation function
    esn_->outputInvAct_( O.data(), O.numRows()*O.numCols() );

    // square and double state if we have additional squared state updates
    if( esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_SQUARE )
    {
      M = Mtmp( _(washout+1,steps), _);
    }
    else
    {
      M.resize( steps-washout, Mtmp.numCols()*2 );
      M( _, _(1,Mtmp.numCols()) ) = Mtmp( _(washout+1,steps), _);
      this->squareStates();
    }

    // calc weights with pseudo inv: Wout_ = (M^-1) * O
    flens::lss( M, O );
    T1 = flens::transpose( O(_( 1, M.numCols() ),_) );
    esn_->Wout_(i,_) = T1(1,_);


    // 4. restore simulation matrix M

    if( i < esn_->outputs_ )
    {
      M.resize( Mtmp.numRows(), Mtmp.numCols() );

      // undo the delays and store it again into M
      for(int j=1; j<=esn_->neurons_+esn_->inputs_; ++j)
      {
        rest = esn_->sim_->getDelayBuffer(i-1,j-1);
        delay = rest.length();

        if( delay != 0 )
        {
          M( _(1,steps-delay), j ) = Mtmp( _(delay+1,steps), j );
          M( _(steps-delay+1,steps), j ) = rest;
        }
        else
          M(_,j) = Mtmp(_,j);
      }
    }
  }

  this->clearData();
}

//@}

} // end of namespace aureservoir
