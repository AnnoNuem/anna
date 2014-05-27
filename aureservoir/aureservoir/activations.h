/***************************************************************************/
/*!
 *  \file   activations.h
 *
 *  \brief  file for all kinds of different neuron activation functions
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

#ifndef AURESERVOIR_ACTIVATIONS_H__
#define AURESERVOIR_ACTIVATIONS_H__

#include "utilities.h"
#include <math.h>

namespace aureservoir
{

/*!
 * \enum ActivationFunction
 * all possible activation functions for reservoir and output neurons
 */
enum ActivationFunction
{
  ACT_LINEAR,      //!< linear activation function
  ACT_TANH,        //!< tanh activation function
  ACT_TANH2,       //!< tanh activation function with local slope and bias
  ACT_SIGMOID      //!< sigmoid activation function
};

/// \todo code this activation functions with SSE2 instructions, and/or
///       make faster interpolations

//! @name linear activation functions
//@{

/*!
 * linear activation function, performed on each element
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_linear(T *data, int size)
{ }

/*!
 * inverse linear activation function, performed on each element
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_invlinear(T *data, int size)
{ }

//@}
//! @name tanh activation functions
/// \todo see if there are faster tanh interpolations or SSE2 instructions
//@{

/*!
 * tanh activation function, performed on each element
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_tanh(T *data, int size)
{
  for(int i=0; i<size; ++i)
    data[i] = tanh( data[i] );
}

/*!
 * inverse tanh activation function, performed on each element
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_invtanh(T *data, int size)
{
  for(int i=0; i<size; ++i)
    data[i] = atanh( data[i] );
}

//@}
//! @name tanh2 activation functions
//@{

/// slope vector for tanh2
DEVector<double>::Type tanh2_a_;
/// bias vector for tanh2
DEVector<double>::Type tanh2_b_;

/*!
 * tanh2 activation function with local slope a and bias b
 * this means the following: y(x) = tanh( a*x + b )
 * where a and b are vetors with same size as the data
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_tanh2(T *data, int size)
{
  assert( tanh2_a_.length() == size );
  assert( tanh2_b_.length() == size );

  for(int i=0; i<size; ++i)
    data[i] = tanh( data[i]*tanh2_a_(i+1) + tanh2_b_(i+1) );
}

/*!
 * inverse tanh2 activation function
 * this means the following: y(x) = (atanh(x) - b) / a
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_invtanh2(T *data, int size)
{
  assert( tanh2_a_.length() == size );
  assert( tanh2_b_.length() == size );

  for(int i=0; i<size; ++i)
    data[i] = ( atanh(data[i]) - tanh2_b_(i+1) ) / tanh2_a_(i+1);
}

//@}
//! @name sigmoid activation functions
//@{

/*!
 * sigmoid activation function, performed on each element:
 * y(x) = 1 / (1 + exp(x))
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_sigmoid(T *data, int size)
{
  for(int i=0; i<size; ++i)
    data[i] = 1.0 / (1.0 + exp(data[i]) );
}

/*!
 * inverse sigmoid activation function, performed on each element:
 * y(x) = ln( 1/x - 1 )
 * \todo make checks here if parameters are in correct range ?
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_invsigmoid(T *data, int size)
{
  for(int i=0; i<size; ++i)
    data[i] = log( 1.0/data[i] - 1.0 );
}

//@}

} // end of namespace aureservoir

#endif // AURESERVOIR_ACTIVATIONS_H__
