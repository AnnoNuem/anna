/***************************************************************************/
/*!
 *  \file   oct_esn.h
 *
 *  \brief  octave type for the ESN class
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   Oct 2007
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

#ifndef OCT_ESN_H__
#define OCT_ESN_H__

#include <iostream>
#include <octave/oct.h>
#include <octave/parse.h>
#include <octave/dynamic-ld.h>
#include <octave/oct-map.h>
#include <octave/oct-stream.h>
#include <octave/ov-base-scalar.h>
#include <vector>
#include <string>

#include "aureservoir/aureservoir.h"

using namespace aureservoir;

/*!
 * \class oct_esn
 * octave type for an Echo State Network
 */
class oct_esn: public octave_base_value
{
 public:

  oct_esn() : octave_base_value()
  { }

  /// \todo more constructors ? (e.g. with size)

  ~oct_esn(void)
  { }

  /// get data
  ESN<double> &net() { return esn_; }

  /// prints out typeinformation
  void print (std::ostream& os, bool pr_as_read_syntax = false) const
  { os << "Octavec ESN (Echo State Network) object." << std::endl; }

  bool is_constant (void) const { return true; }
  bool is_defined (void) const { return true; }

 private:

  /// our network data
  ESN<double> esn_;

  DECLARE_OCTAVE_ALLOCATOR
  DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

DEFINE_OCTAVE_ALLOCATOR (oct_esn);
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (oct_esn, "esn", "esn");

#endif // OCT_ESN_
