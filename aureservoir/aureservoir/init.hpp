/***************************************************************************/
/*!
 *  \file   init.hpp
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

#include <algorithm>

namespace aureservoir
{

//! @name class InitBase Implementation
//@{

template <typename T>
void InitBase<T>::checkInitParams()
  throw(AUExcept)
{
  T tmp;

  tmp = esn_->init_params_[CONNECTIVITY];
  if( tmp<=0 || tmp>1 )
    throw AUExcept("InitBase::checkInitParams: CONNECTIVITY must be within [0|1]");

  tmp = esn_->init_params_[ALPHA];
  if( tmp<0 )
    throw AUExcept("InitBase::checkInitParams: ALPHA must be >= 0");

  tmp = esn_->init_params_[IN_CONNECTIVITY];
  if( tmp<0 || tmp>1 )
    throw AUExcept("InitBase::checkInitParams: IN_CONNECTIVITY must be within [0|1]");

  tmp = esn_->init_params_[FB_CONNECTIVITY];
  if( tmp<0 || tmp>1 )
    throw AUExcept("InitBase::checkInitParams: FB_CONNECTIVITY must be within [0|1]");

  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] == SIM_LI )
  {
    if( esn_->init_params_.find(LEAKING_RATE) == esn_->init_params_.end() )
      throw AUExcept("InitBase::checkInitParams: No LEAKING_RATE given !");

    tmp = esn_->init_params_[LEAKING_RATE];
    if( tmp<0 )
      throw AUExcept("InitBase::checkInitParams: LEAKING_RATE must be >= 0 !");
  }

  if( esn_->net_info_[ESN<T>::TRAIN_ALG] == TRAIN_RIDGEREG )
  {
    if( esn_->init_params_.find(TIKHONOV_FACTOR) == esn_->init_params_.end() )
      throw AUExcept("InitBase::checkInitParams: No TIKHONOV_FACTOR given !");

    tmp = esn_->init_params_[TIKHONOV_FACTOR];
    if( tmp<0 )
      throw AUExcept("InitBase::checkInitParams: TIKHONOV_FACTOR must be >= 0 !");
  }
}

template <typename T>
void InitBase<T>::allocateWorkData()
{
  // to allocate sim class data, if we don't use training
  esn_->sim_->reallocate();

  // calc params for tanh2 activation function
  if( esn_->net_info_[ESN<T>::RESERVOIR_ACT] == ACT_TANH2 )
  {
    tanh2_a_.resize( esn_->neurons_ );
    tanh2_b_.resize( esn_->neurons_ );
    std::fill_n( tanh2_a_.data(), esn_->neurons_, 1. );
    std::fill_n( tanh2_b_.data(), esn_->neurons_, 0. );
  }

  // set filtered neurons to standard ESN calculation
  int simalg = esn_->net_info_[ESN<T>::SIMULATE_ALG];
  if( simalg==SIM_FILTER || simalg==SIM_FILTER2 || simalg==SIM_FILTER_DS || simalg==SIM_SQUARE )
  {
    typename DEMatrix<T>::Type B(esn_->neurons_, 2), A(esn_->neurons_, 2);

    for(int i=1; i<=esn_->neurons_; ++i)
    {
      B(i,1) = 1.; B(i,2) = 0.;
      A(i,1) = 1.; A(i,2) = 0.;
    }
    esn_->setIIRCoeff(B,A);
  }
}

//@}
//! @name class InitStd Implementation
//@{

template <typename T>
void InitStd<T>::init()
  throw(AUExcept)
{
  this->checkInitParams();
  this->allocateWorkData();

  esn_->Win_.resizeOrClear(esn_->neurons_, esn_->inputs_);
  esn_->Wback_.resizeOrClear(esn_->neurons_, esn_->outputs_);
  esn_->Wout_.resizeOrClear(esn_->outputs_, esn_->neurons_+esn_->inputs_);
  esn_->x_.resizeOrClear(esn_->neurons_);


  // INPUT WEIGHTS

  // generate random weigths within [-1,1] with given connectivity
  T nrn = esn_->neurons_*esn_->init_params_[IN_CONNECTIVITY]
          * esn_->inputs_ - 0.5;
  int i;
  for (i=0; i<nrn; ++i)
    esn_->Win_.data()[i] = Rand<T>::uniform();

  // shuffle elements within the matrix
  int mtxsize = esn_->Win_.numRows() * esn_->Win_.numCols();
  std::random_shuffle( esn_->Win_.data(),
                       esn_->Win_.data() + mtxsize );

  // scale and shift elemets
  esn_->Win_ *= esn_->init_params_[IN_SCALE];
  esn_->Win_ += esn_->init_params_[IN_SHIFT];


  // FEEDBACK WEIGHTS

  // generate random weigths within [-1,1] with given connectivity
  nrn = esn_->neurons_*esn_->init_params_[FB_CONNECTIVITY]
        * esn_->outputs_ - 0.5;
  for (int i=0; i<nrn; ++i)
    esn_->Wback_.data()[i] = Rand<T>::uniform();

  // shuffle elements within the matrix
  mtxsize = esn_->Wback_.numRows() * esn_->Wback_.numCols();
  std::random_shuffle( esn_->Wback_.data(),
                       esn_->Wback_.data() + mtxsize );

  // scale and shift elemets
  esn_->Wback_ *= esn_->init_params_[FB_SCALE];
  esn_->Wback_ += esn_->init_params_[FB_SHIFT];


  // RESERVOIR WEIGHTS

  // temporal dense reservoir weight matrix
  typename DEMatrix<T>::Type Wtmp(esn_->neurons_, esn_->neurons_),
                             tmp(esn_->neurons_, esn_->neurons_);

  // generate random weigths within [-1,1] with given connectivity
  nrn = esn_->neurons_*esn_->neurons_*esn_->init_params_[CONNECTIVITY] - 0.5;
  for (int i=0; i<nrn; ++i)
    Wtmp.data()[i] = Rand<T>::uniform();

  // shuffle elements within the matrix
  std::random_shuffle( Wtmp.data(), 
                       Wtmp.data() + (Wtmp.numRows()*Wtmp.numCols()) );

  // calc eigenvalues of Wtmp
  typename DEVector<T>::Type wr(esn_->neurons_),
                             wi(esn_->neurons_),
                             w(esn_->neurons_);
  tmp = Wtmp; // needs a deep copy for ev

  flens::ev(false, false, tmp, wr, wi, tmp, tmp); // last 2 are dummy for EV

  // calc largest absolut eigenvalue
  for(int i=1; i<=esn_->neurons_; ++i)
    w(i) = sqrt( pow(wr(i),2) + pow(wi(i),2) );
  T max_ew = *std::max_element( w.data(), w.data()+w.length() );

  // check if we have zero max_ew
  if( max_ew == 0 )
    throw AUExcept("InitStd::init: maximum eigenvalue is zero ! Try init again!");

  // scale matrix to spectral radius alpha
  T alpha = esn_->init_params_[ALPHA];
  Wtmp *= (alpha / max_ew);

  // finally convert it to sparse matrix
//   esn_->W_.initWith(Wtmp, 1E-9);
  esn_->W_ = Wtmp; /// \todo check initialization with epsillon again in flens !!!
}

//@}

} // end of namespace aureservoir
