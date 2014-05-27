/***************************************************************************/
/*!
 *  \file   esn.hpp
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

namespace aureservoir
{

template <typename T>
ESN<T>::ESN()
{
  set_denormal_flags();
  Rand<T>::initSeed();

  init_=0;
  train_=0;
  sim_=0;

  // set some standard parameters

  setSize(10);
  setInputs(1);
  setOutputs(1);
  noise_=0;

  setInitParam(CONNECTIVITY, 0.8);
  setInitParam(ALPHA, 0.8);
  setInitParam(IN_CONNECTIVITY, 0.8);
  setInitParam(IN_SCALE, 1.);
  setInitParam(IN_SHIFT, 0.);
  setInitParam(FB_CONNECTIVITY, 0.);
  setInitParam(FB_SCALE, 1.);
  setInitParam(FB_SHIFT, 0.);

  setReservoirAct(ACT_TANH);
  setOutputAct(ACT_LINEAR);

  setInitAlgorithm(INIT_STD);
  setTrainAlgorithm(TRAIN_PI);
  setSimAlgorithm(SIM_STD);
}

template <typename T>
ESN<T>::ESN(const ESN<T> &src)
{
  init_=0;
  train_=0;
  sim_=0;
  noise_=0;

  neurons_ = src.neurons_;
  inputs_ = src.inputs_;
  outputs_ = src.outputs_;
  noise_ = src.noise_;

  /// \todo check if maps operator= performs a deep copy !
  init_params_ = src.init_params_;

  InitAlgorithm init = src.getInitAlgorithm();
  setInitAlgorithm( init );
  TrainAlgorithm train = src.getTrainAlgorithm();
  setTrainAlgorithm(train);

  // copy simulation alg and its data
  net_info_[SIMULATE_ALG] = src.getSimAlgorithm();
  sim_ = src.sim_->clone(this);

  Win_ = src.Win_;
  W_ = src.W_;
  Wback_ = src.Wback_;
  Wout_ = src.Wout_;
  x_ = src.x_;

  ActivationFunction tmp = src.getReservoirAct();
  setReservoirAct(tmp);
  tmp = src.getOutputAct();
  setOutputAct(tmp);
}

template <typename T>
const ESN<T> &ESN<T>::operator= (const ESN<T> &src)
{
  ESN<T>::ESN(src);
  return *this;
}

template <typename T>
ESN<T>::~ESN()
{
  if(init_) delete init_;
  if(train_) delete train_;
  if(sim_) delete sim_;
}

template <typename T>
double ESN<T>::adapt(const DEMatrix &in)
  throw(AUExcept)
{
  // this allocates the data, so we really need ACT_TANH2 activation function
  if( net_info_[RESERVOIR_ACT] != ACT_TANH2 )
    throw AUExcept("adapt: You need to set ACT_TANH2 activation function!");

  // check if we have all parameters
  if( init_params_.find(IP_LEARNRATE) == init_params_.end() )
    throw AUExcept("adapt: You need IP_LEARNRATE init parameter for adaptation");
  if( init_params_.find(IP_VAR) == init_params_.end() )
    throw AUExcept("adapt: You need IP_VAR init parameter for adaptation");
  if( init_params_.find(IP_MEAN) == init_params_.end() )
    init_params_[IP_MEAN] = 0.;

  // difference vectors to see learn progress
  flens::DenseVector<flens::Array<double> > a_diff = tanh2_a_,
                                            b_diff = tanh2_b_;

  // adaptation algorithm

  DEVector y(neurons_);
  DEMatrix sim_in(inputs_,1), sim_out(outputs_,1);
  int steps = in.numCols();
  T lr = init_params_[IP_LEARNRATE];
  T mean = init_params_[IP_MEAN];
  T var = init_params_[IP_VAR];
  T db;

  // set linear activation function to get values before nonlinearity
  setReservoirAct(ACT_LINEAR);

  for(int n=1; n<=steps; ++n)
  {
    // simulate one step
    sim_in(_,1) = in(_,n);
    simulate(sim_in, sim_out);
    // now we have our network state x_ without activation function

    // calc activation
    y = x_;
    act_tanh2( y.data(), y.length() );

    for(int m=1; m<=neurons_; ++m)
    {
      db = -lr * ( -mean/var + (y(m)/var)*(2*var+1-pow(y(m),2)+y(m)*mean) );
      tanh2_b_(m) += db;
      tanh2_a_(m) += lr/tanh2_a_(m) + db*x_(m);
    }
  }

  // re-set original reservoir activation
  setReservoirAct(ACT_TANH2);

  // calc difference / learn progress
  double progress=0;
  a_diff -= tanh2_a_;
  b_diff -= tanh2_b_;
  for(int i=1; i<=neurons_; ++i)
    progress = a_diff(i) + b_diff(i);
  progress /= 2*neurons_;

  return progress;
}

template <typename T>
double ESN<T>::adapt(T *inmtx, int inrows, int incols)
  throw(AUExcept)
{
  if( inrows != inputs_ )
    throw AUExcept("adapt: wrong input row size!");

  DEMatrix flin(inrows,incols);

  // copy data to FLENS matrix (column major storage)
  for(int i=0; i<inrows; ++i) {
  for(int j=0; j<incols; ++j) {
    flin(i+1,j+1) = inmtx[i*incols+j];
  } }

  return adapt(flin);
}

template <typename T>
inline void ESN<T>::train(T *inmtx, int inrows, int incols,
                          T *outmtx, int outrows, int outcols, int washout)
  throw(AUExcept)
{
  DEMatrix flin(inrows,incols);
  DEMatrix flout(outrows,outcols);

  // copy data to FLENS matrix (column major storage)
  for(int i=0; i<inrows; ++i) {
  for(int j=0; j<incols; ++j) {
    flin(i+1,j+1) = inmtx[i*incols+j];
  } }
  for(int i=0; i<outrows; ++i) {
  for(int j=0; j<outcols; ++j) {
    flout(i+1,j+1) = outmtx[i*outcols+j];
  } }

//   std::cout << "Flens IN: " << flin << std::endl;
//   std::cout << "Flens OUT: " << flout << std::endl;

  train(flin, flout, washout);
}

template <typename T>
inline void ESN<T>::simulate(T *inmtx, int inrows, int incols,
                             T *outmtx, int outrows, int outcols)
  throw(AUExcept)
{
  if( outcols != incols )
    throw AUExcept("ESN::simulate: output and input must have same nr of columns!");
  if( inrows != inputs_ )
    throw AUExcept("ESN::simulate: wrong input row size!");
  if( outrows != outputs_ )
    throw AUExcept("ESN::simulate: wrong output row size!");

  // check if we already have allocated data in simulation algorithm
  if( sim_->last_out_.numRows() != outputs_ )
    throw AUExcept("ESN::simulate: You need to allocate data for simulation algortihm - e.g. set an Wout matrix or train ESN !");

  DEMatrix flin(inrows,incols);
  DEMatrix flout(outrows,outcols);

  // copy data to FLENS matrix
  for(int i=0; i<inrows; ++i) {
  for(int j=0; j<incols; ++j) {
    flin(i+1,j+1) = inmtx[i*incols+j];
  } }

  simulate(flin, flout);

  // copy data to output
  for(int i=0; i<outrows; ++i) {
  for(int j=0; j<outcols; ++j) {
    outmtx[i*outcols+j] = flout(i+1,j+1);
  } }
}

template <typename T>
inline void ESN<T>::simulateStep(T *invec, int insize, T *outvec, int outsize)
    throw(AUExcept)
{
  if( insize != inputs_ )
    throw AUExcept("ESN::simulate: wrong input row size!");
  if( outsize != outputs_ )
    throw AUExcept("ESN::simulate: wrong output row size!");

  // check if we already have allocated data in simulation algorithm
  if( sim_->last_out_.numRows() != outputs_ )
    throw AUExcept("ESN::simulate: You need to allocate data for simulation algortihm - e.g. set an Wout matrix or train ESN !");

  DEMatrix flin(insize,1);
  DEMatrix flout(outsize,1);

  // copy data to FLENS matrix
  for(int i=0; i<insize; ++i)
    flin(i+1,1) = invec[i];

  simulate(flin, flout);

  // copy data to output
  for(int i=0; i<outsize; ++i)
    outvec[i] = flout(i+1,1);
}

template <typename T>
void ESN<T>::setBPCutoff(const DEVector &f1, const DEVector &f2)
  throw(AUExcept)
{
  if( net_info_[SIMULATE_ALG] != SIM_BP )
    throw AUExcept("ESN::setBPCutoff: you need to set SIM_BP and init the matrix first!");

  sim_->setBPCutoff(f1,f2);
}

template <typename T>
void ESN<T>::setBPCutoff(T *f1vec, int f1size, T *f2vec, int f2size)
    throw(AUExcept)
{
  if( f1size != neurons_ || f2size != neurons_ )
    throw AUExcept("ESN::setBPCutoff: vectors f1, f2 must be same size as neurons in the reservoir !");

  DEVector f1(neurons_);
  DEVector f2(neurons_);

  // copy data to FLENS vectors
  for(int i=0; i<neurons_; ++i)
  {
    f1(i+1) = f1vec[i];
    f2(i+1) = f2vec[i];
  }

  setBPCutoff(f1,f2);
}

template <typename T>
void ESN<T>::setIIRCoeff(const DEMatrix &B, const DEMatrix &A, int series)
  throw(AUExcept)
{
  if( net_info_[SIMULATE_ALG] != SIM_FILTER  &&
      net_info_[SIMULATE_ALG] != SIM_FILTER2 &&
      net_info_[SIMULATE_ALG] != SIM_FILTER_DS &&
      net_info_[SIMULATE_ALG] != SIM_SQUARE )
    throw AUExcept("ESN::setIIRCoeff: you need to set SIM_FILTER, SIM_FILTER2, SIM_FILTER_DS or SIM_SQUARE and init the matrix first!");

  sim_->setIIRCoeff(B,A,series);
}

template <typename T>
void ESN<T>::setIIRCoeff(T *bmtx, int brows, int bcols,
                         T *amtx, int arows, int acols,
                         int series)
  throw(AUExcept)
{
  if( brows != neurons_ || arows != neurons_ )
    throw AUExcept("ESN::setIIRCoeff: A and B must have as many rows as neurons in the ESN !");

  DEMatrix B(brows,bcols);
  DEMatrix A(arows,acols);

  // copy data to FLENS matrix (column major storage)
  for(int i=0; i<arows; ++i)
  {
    for(int j=0; j<acols; ++j)
      A(i+1,j+1) = amtx[i*acols+j];

    for(int j=0; j<bcols; ++j)
      B(i+1,j+1) = bmtx[i*bcols+j];
  }

  sim_->setIIRCoeff(B,A,series);
}

template <typename T>
void ESN<T>::post()
{
  std::cout << "--------------------------------------------\n"
            << "ESN Parameters:\n"
            << "\n"
            << "nr of neurons:\t" << neurons_ << "\n"
            << "reservoir connectivity:\t"
            << init_params_[CONNECTIVITY] << "\n"
            << "spectral radius:\t" << init_params_[ALPHA] << "\n";
  if( net_info_[SIMULATE_ALG] == SIM_LI )
  {
      std::cout << "leaking rate:\t" << init_params_[LEAKING_RATE] << "\n";
  }
  std::cout << "\n"
            << "inputs:\t" << inputs_ << "\n"
            << "input connectivity:\t"
            << init_params_[IN_CONNECTIVITY] << "\n"
            << "input scale:\t" << init_params_[IN_SCALE] << "\n"
            << "input shift:\t" << init_params_[IN_SHIFT] << "\n"
            << "\n"
            << "outputs:\t" << outputs_ << "\n"
            << "feedback connectivity:\t"
            << init_params_[FB_CONNECTIVITY] << "\n"
            << "feedback scale:\t" << init_params_[FB_SCALE] << "\n"
            << "feedback shift:\t" << init_params_[FB_SHIFT] << "\n"
            << "\n"
            << "noise level for simulation/training:\t" << noise_ << "\n"
            << "\n"
            << "reservoir activation fct:\t"
            << getActString( net_info_[RESERVOIR_ACT] ) << "\n"
            << "output activation fct:\t"
            << getActString( net_info_[OUTPUT_ACT] ) << "\n"
            << "\n"
            << "initialization algorithm:\t"
            << getInitString( net_info_[INIT_ALG] ) << "\n"
            << "training algorithm:\t"
            << getTrainString( net_info_[TRAIN_ALG] ) << "\n"
            << "simulation algorithm:\t"
            << getSimString( net_info_[SIMULATE_ALG] ) << "\n"
            << "--------------------------------------------\n";
}

template <typename T>
void ESN<T>::getWin(T **mtx, int *rows, int *cols)
{
  *mtx = Win_.data();
  *rows = Win_.numRows();
  *cols = Win_.numCols();
}

template <typename T>
void ESN<T>::getWback(T **mtx, int *rows, int *cols)
{
  *mtx = Wback_.data();
  *rows = Wback_.numRows();
  *cols = Wback_.numCols();
}

template <typename T>
void ESN<T>::getWout(T **mtx, int *rows, int *cols)
{
//   std::cout << "Wout in C++: " << Wout_ << std::endl;

  *mtx = Wout_.data();
  *rows = Wout_.numRows();
  *cols = Wout_.numCols();
}

template <typename T>
void ESN<T>::getX(T **vec, int *length)
{
  *vec = x_.data();
  *length = x_.length();
}

template <typename T>
void ESN<T>::getW(T *wmtx, int wrows, int wcols)
  throw(AUExcept)
{
  if( wrows != W_.numRows() )
    throw AUExcept("ESN::getW: wrong row size!");
  if( wcols != W_.numCols() )
    throw AUExcept("ESN::getW: wrong column size!");

  // convert to dense matrix for element access
  /// \todo check if this can be avoided
  DEMatrix Wtmp;
  Wtmp = W_;

  for(int i=0; i<wrows; ++i) {
  for(int j=0; j<wcols; ++j) {
    wmtx[i*wcols+j] = Wtmp(i+1,j+1);
  } }
}

template <typename T>
void ESN<T>::getDelays(T *wmtx, int wrows, int wcols)
  throw(AUExcept)
{
  DEMatrix Dtmp = getDelays();

  if( wrows != Dtmp.numRows() )
    throw AUExcept("ESN::getDelays: wrong row size!");
  if( wcols != Dtmp.numCols() )
    throw AUExcept("ESN::getDelays: wrong column size!");

  for(int i=0; i<wrows; ++i) {
  for(int j=0; j<wcols; ++j) {
    wmtx[i*wcols+j] = Dtmp(i+1,j+1);
  } }
}

template <typename T>
void ESN<T>::setInitAlgorithm(InitAlgorithm alg)
  throw(AUExcept)
{
  switch(alg)
  {
    case INIT_STD:
      if(init_) delete init_;
      init_ = new InitStd<T>(this);
      net_info_[INIT_ALG] = INIT_STD;
      break;

    default:
      throw AUExcept("ESN::setInitAlgorithm: no valid Algorithm!");
  }
}

template <typename T>
void ESN<T>::setTrainAlgorithm(TrainAlgorithm alg)
  throw(AUExcept)
{
  switch(alg)
  {
    case TRAIN_PI:
      if(train_) delete train_;
      train_ = new TrainPI<T>(this);
      net_info_[TRAIN_ALG] = TRAIN_PI;
      break;

    case TRAIN_LS:
      if(train_) delete train_;
      train_ = new TrainLS<T>(this);
      net_info_[TRAIN_ALG] = TRAIN_LS;
      break;

    case TRAIN_RIDGEREG:
      if(train_) delete train_;
      train_ = new TrainRidgeReg<T>(this);
      net_info_[TRAIN_ALG] = TRAIN_RIDGEREG;
      break;

    case TRAIN_DS_PI:
      if(train_) delete train_;
      train_ = new TrainDSPI<T>(this);
      net_info_[TRAIN_ALG] = TRAIN_DS_PI;
      break;

    default:
      throw AUExcept("ESN::setTrainAlgorithm: no valid Algorithm!");
  }
}

template <typename T>
void ESN<T>::setSimAlgorithm(SimAlgorithm alg)
  throw(AUExcept)
{
  switch(alg)
  {
    case SIM_STD:
      if(sim_) delete sim_;
      sim_ = new SimStd<T>(this);
      net_info_[SIMULATE_ALG] = SIM_STD;
      break;

    case SIM_SQUARE:
      if(sim_) delete sim_;
      sim_ = new SimSquare<T>(this);
      net_info_[SIMULATE_ALG] = SIM_SQUARE;
      break;

    case SIM_LI:
      if(sim_) delete sim_;
      sim_ = new SimLI<T>(this);
      net_info_[SIMULATE_ALG] = SIM_LI;
      break;

    case SIM_BP:
      if(sim_) delete sim_;
      sim_ = new SimBP<T>(this);
      net_info_[SIMULATE_ALG] = SIM_BP;
      break;

    case SIM_FILTER:
      if(sim_) delete sim_;
      sim_ = new SimFilter<T>(this);
      net_info_[SIMULATE_ALG] = SIM_FILTER;
      break;

    case SIM_FILTER2:
      if(sim_) delete sim_;
      sim_ = new SimFilter2<T>(this);
      net_info_[SIMULATE_ALG] = SIM_FILTER2;
      break;

    case SIM_FILTER_DS:
      if(sim_) delete sim_;
      sim_ = new SimFilterDS<T>(this);
      net_info_[SIMULATE_ALG] = SIM_FILTER_DS;
      break;

    default:
      throw AUExcept("ESN::setSimAlgorithm: no valid Algorithm!");
  }
}

template <typename T>
void ESN<T>::setSize(int neurons)
  throw(AUExcept)
{
  if(neurons<1)
    throw AUExcept("ESN::setReservoirSize: there must be at least one neuron!");

  neurons_=neurons;
}

template <typename T>
void ESN<T>::setInputs(int inputs)
  throw(AUExcept)
{
  if(inputs<1)
    throw AUExcept("ESN::setInputs: there must be at least one input!");

  inputs_=inputs;
}

template <typename T>
void ESN<T>::setOutputs(int outputs)
  throw(AUExcept)
{
  if(outputs<1)
    throw AUExcept("ESN::setOutputs: there must be at least one output!");

  outputs_=outputs;
}

template <typename T>
void ESN<T>::setNoise(double noise)
  throw(AUExcept)
{
  if(noise<0)
    throw AUExcept("ESN::setNoise: Noise level must be a positive number!");

  noise_=noise;
}

template <typename T>
void ESN<T>::setInitParam(InitParameter key, T value)
{
  init_params_[key] = value;
}

template <typename T>
void ESN<T>::setReservoirAct(ActivationFunction f)
  throw(AUExcept)
{
  switch(f)
  {
    case ACT_LINEAR:
      reservoirAct_= act_linear;
      net_info_[RESERVOIR_ACT] = ACT_LINEAR;
      break;

    case ACT_TANH:
      reservoirAct_= act_tanh;
      net_info_[RESERVOIR_ACT] = ACT_TANH;
      break;

    case ACT_TANH2:
      reservoirAct_= act_tanh2;
      net_info_[RESERVOIR_ACT] = ACT_TANH2;
      break;

    case ACT_SIGMOID:
      reservoirAct_= act_sigmoid;
      net_info_[RESERVOIR_ACT] = ACT_SIGMOID;
      break;

    default:
      throw AUExcept("ESN::setReservoirAct: wrong reservoir activation function!");
  }
}

template <typename T>
void ESN<T>::setOutputAct(ActivationFunction f)
  throw(AUExcept)
{
  switch(f)
  {
    case ACT_LINEAR:
      outputAct_    = act_linear;
      outputInvAct_ = act_invlinear;
      net_info_[OUTPUT_ACT] = ACT_LINEAR;
      break;

    case ACT_TANH:
      outputAct_    = act_tanh;
      outputInvAct_ = act_invtanh;
      net_info_[OUTPUT_ACT] = ACT_TANH;
      break;

    case ACT_SIGMOID:
      outputAct_    = act_sigmoid;
      outputInvAct_ = act_invsigmoid;
      net_info_[OUTPUT_ACT] = ACT_SIGMOID;
      break;

    default:
      throw AUExcept("ESN::setOutputAct: wrong output activation function!");
  }
}

template <typename T>
void ESN<T>::setWin(const DEMatrix &Win) throw(AUExcept)
{
  if( Win.numRows() != neurons_ )
      throw AUExcept("ESN::setWin: wrong row size!");
  if( Win.numCols() != inputs_ )
      throw AUExcept("ESN::setWin: wrong column size!");

  Win_ = Win;
}

template <typename T>
void ESN<T>::setW(const DEMatrix &W) throw(AUExcept)
{
  if( W.numRows() != neurons_ )
      throw AUExcept("ESN::setW: wrong row size!");
  if( W.numCols() != neurons_ )
      throw AUExcept("ESN::setW: wrong column size!");

//   W_.initWith(W, 1E-9);
  W_ = W;
}

template <typename T>
void ESN<T>::setWback(const DEMatrix &Wback) throw(AUExcept)
{
  if( Wback.numRows() != neurons_ )
      throw AUExcept("ESN::setWback: wrong row size!");
  if( Wback.numCols() != outputs_ )
      throw AUExcept("ESN::setWback: wrong column size!");

  Wback_ = Wback;
}

template <typename T>
void ESN<T>::setWout(const DEMatrix &Wout) throw(AUExcept)
{
  if( Wout.numRows() != outputs_ )
      throw AUExcept("ESN::setWout: Wout must have output_ rows!");
  if( Wout.numCols() != inputs_+neurons_ && 
      Wout.numCols() != 2*(inputs_+neurons_) )
      throw AUExcept("ESN::setWout: wrong column size!");

  Wout_ = Wout;
}

template <typename T>
void ESN<T>::setX(const DEVector &x) throw(AUExcept)
{
  if( x.length() != neurons_ )
      throw AUExcept("ESN::setX: wrong size!");

  x_ = x;
}

template <typename T>
void ESN<T>::setLastOutput(const DEVector &last)
  throw(AUExcept)
{
  if( last.length() != outputs_ )
      throw AUExcept("setLastOutput: wrong size!");

  sim_->last_out_.resize(last.length(),1);
  sim_->last_out_(_,1) = last;
}

template <typename T>
void ESN<T>::setWin(T *inmtx, int inrows, int incols) throw(AUExcept)
{
  if( inrows != neurons_ )
      throw AUExcept("ESN::setWin: wrong row size!");
  if( incols != inputs_ )
      throw AUExcept("ESN::setWin: wrong column size!");

  Win_.resize(inrows,incols);

  for(int i=0; i<inrows; ++i) {
  for(int j=0; j<incols; ++j) {
    Win_(i+1,j+1) = inmtx[i*incols+j];
  } }
}

template <typename T>
void ESN<T>::setW(T *inmtx, int inrows, int incols) throw(AUExcept)
{
  if( inrows != neurons_ )
      throw AUExcept("ESN::setW: wrong row size!");
  if( incols != neurons_ )
      throw AUExcept("ESN::setW: wrong column size!");

  DEMatrix Wtmp(neurons_,neurons_);

  for(int i=0; i<inrows; ++i) {
  for(int j=0; j<incols; ++j) {
    Wtmp(i+1,j+1) = inmtx[i*incols+j];
  } }

//   W_.initWith(Wtmp, 1E-9);
  W_ = Wtmp;
}

template <typename T>
void ESN<T>::setWback(T *inmtx, int inrows, int incols) throw(AUExcept)
{
  if( inrows != neurons_ )
      throw AUExcept("ESN::setWback: wrong row size!");
  if( incols != outputs_ )
      throw AUExcept("ESN::setWback: wrong column size!");

  Wback_.resize(inrows,incols);

  for(int i=0; i<inrows; ++i) {
  for(int j=0; j<incols; ++j) {
    Wback_(i+1,j+1) = inmtx[i*incols+j];
  } }
}

template <typename T>
void ESN<T>::setWout(T *inmtx, int inrows, int incols) throw(AUExcept)
{
  if( inrows != outputs_ )
      throw AUExcept("ESN::setWout: Wout must have output_ rows!");
  if( incols != inputs_+neurons_ && 
      incols != 2*(inputs_+neurons_) )
      throw AUExcept("ESN::setWout: wrong column size!");

  Wout_.resize(inrows,incols);

  for(int i=0; i<inrows; ++i) {
  for(int j=0; j<incols; ++j) {
    Wout_(i+1,j+1) = inmtx[i*incols+j];
  } }
}

template <typename T>
void ESN<T>::setX(T *invec, int insize) throw(AUExcept)
{
  if( insize != neurons_ )
      throw AUExcept("ESN::setX: wrong size!");

  x_.resize(insize);
  for(int i=0; i<insize; ++i)
    x_(i+1) = invec[i];
}

template <typename T>
void ESN<T>::setLastOutput(T *last, int size)
  throw(AUExcept)
{
  if( size != outputs_ )
      throw AUExcept("setLastOutput: wrong size!");

  sim_->last_out_.resize(size,1);
  for(int i=0; i<size; ++i)
    sim_->last_out_(i+1,1) = last[i];
}

template <typename T>
string ESN<T>::getActString(int act)
{
  switch(act)
  {
    case ACT_LINEAR:
      return "ACT_LINEAR";

    case ACT_TANH:
      return "ACT_TANH";

    case ACT_TANH2:
      return "ACT_TANH2";


    case ACT_SIGMOID:
      return "ACT_SIGMOID";

    default:
      throw AUExcept("ESN::getActString: unknown activation function");
  }
}

template <typename T>
string ESN<T>::getInitString(int alg)
{
  switch(alg)
  {
    case INIT_STD:
      return "INIT_STD";

    default:
      throw AUExcept("ESN::getInitString: unknown init algorithm");
  }
}

template <typename T>
string ESN<T>::getSimString(int alg)
{
  switch(alg)
  {
    case SIM_STD:
      return "SIM_STD";

    case SIM_SQUARE:
      return "SIM_SQUARE";

    case SIM_LI:
      return "SIM_LI";

    case SIM_BP:
      return "SIM_BP";

    case SIM_FILTER:
      return "SIM_FILTER";

    case SIM_FILTER2:
      return "SIM_FILTER2";

    case SIM_FILTER_DS:
      return "SIM_FILTER_DS";

    default:
      throw AUExcept("ESN::getSimString: unknown simulation algorithm");
  }
}

template <typename T>
string ESN<T>::getTrainString(int alg)
{
  switch(alg)
  {
    case TRAIN_PI:
      return "TRAIN_PI";

    case TRAIN_LS:
      return "TRAIN_LS";

    case TRAIN_RIDGEREG:
      return "TRAIN_RIDGEREG";

    case TRAIN_DS_PI:
      return "TRAIN_DS_PI";

    default:
      throw AUExcept("ESN::getTrainString: unknown training algorithm");
  }
}


// template <typename T>
// void ESN<T>::setParameter(string param, string value)
//   throw(AUExcept)
// {
//   // tests machen für diese funktion
// 
//   bool ok = false;
// 
//   // algorithms
// 
//   if( param == "simulation_alg" || param == "sim_alg" ||
//       param == "simulation_algorithm" || param == "sim_algorithm"  )
//   {
//     if( value == "sim_std" || value == "std" )
//     {
//       setSimAlgorithm( SIM_STD );
//       ok = true;
//     }
//   }
//   if( param == "training_alg" || param == "train_alg" ||
//       param == "training_algorithm" || param == "train_algorithm"  )
//   {
//     if( value == "train_leastsquare" || value == "leastsquare" )
//     {
//       setTrainAlgorithm( TRAIN_LS );
//       ok = true;
//     }
//   }
//   if( param == "initialization_alg" || param == "init_alg" ||
//       param == "initialization_algorithm" || param == "init_algorithm"  )
//   {
//     if( value == "init_std" || value == "std" )
//     {
//       setInitAlgorithm( INIT_STD );
//       ok = true;
//     }
//   }
// 
//   // general
// 
//   if( param == "size" || param == "reservoir_size" ||
//       param == "reservoir size" || param == "reservoir" )
//   {
//     setSize( stringToInt(value) );
//     ok = true;
//   }
//   if( param == "inputs" )
//   {
//     setInputs( stringToInt(value) );
//     ok = true;
//   }
//   if( param == "outputs" )
//   {
//     setOutputs( stringToInt(value) );
//     ok = true;
//   }
// 
//   // input parameter map
// 
//   if( param == "connectivity" || param == "conn" )
//   {
//     setInitParam( CONNECTIVITY, stringToDouble(value) );
//     ok = true;
//   }
//   if( param == "alpha" )
//   {
//     setInitParam( ALPHA, stringToDouble(value) );
//     ok = true;
//   }
//   if( param == "in_connectivity" || param == "in_conn" )
//   {
//     setInitParam( IN_CONNECTIVITY, stringToDouble(value) );
//     ok = true;
//   }
//   if( param == "in_scale" )
//   {
//     setInitParam( IN_SCALE, stringToDouble(value) );
//     ok = true;
//   }
//   if( param == "in_shift" )
//   {
//     setInitParam( IN_SHIFT, stringToDouble(value) );
//     ok = true;
//   }
//   if( param == "fb_connectivity" || param == "fb_conn" )
//   {
//     setInitParam( FB_CONNECTIVITY, stringToDouble(value) );
//     ok = true;
//   }
//   if( param == "fb_scale" )
//   {
//     setInitParam( FB_SCALE, stringToDouble(value) );
//     ok = true;
//   }
//   if( param == "fb_shift" )
//   {
//     setInitParam( FB_SHIFT, stringToDouble(value) );
//     ok = true;
//   }
// 
//   // activation functions
// 
//   if( param == "reservoir_act" || param == "res_act" )
//   {
//     if( value == "linear" || value == "lin" )
//     {
//       setReservoirAct( ACT_LINEAR );
//       ok = true;
//     }
//     if( value == "tanh" )
//     {
//       setReservoirAct( ACT_TANH );
//       ok = true;
//     }
//   }
//   if( param == "output_act" || param == "out_act" )
//   {
//     if( value == "linear" || value == "lin" )
//     {
//       setOutputAct( ACT_LINEAR );
//       ok = true;
//     }
//     if( value == "tanh" )
//     {
//       setOutputAct( ACT_TANH );
//       ok = true;
//     }
//   }
// 
//   if( !ok )
//     throw AUExcept("ESN::setParameter: parameter value not valid!");
// }

} // end of namespace aureservoir
