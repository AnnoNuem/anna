/***************************************************************************/
/*!
 *  \file   oct_esn.h
 *
 *  \brief  octave interface for the ESN class
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

#include "oct_esn.h"

#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>


DEFUN_DLD(esn_new, args, ,
"-*- texinfo -*-\n\
@deftypefn {Function File} @var{esn} = esn_new()\n\
Creates a new Echo State Network.\n\
TODO: include option parameters as in fann_create !\n\n\
@seealso{esn_init}\n\
@end deftypefn\n\
")
{
  // create the network
  oct_esn *esn = new oct_esn();
  return octave_value(esn);
}

DEFUN_DLD(esn_print, args, ,
"-*- texinfo -*-\n\
@deftypefn {Function File} esn_print\n\
Prints the settings of the current Echo State Network.\n\n\
@seealso{esn_set}\n\
@end deftypefn\n\
")
{
  // get the ESN
  const octave_base_value& rep = args(0).get_rep();
  oct_esn& esn = ((oct_esn &)rep);

  esn.net().postParameters();

  return octave_value();
}

DEFUN_DLD(esn_init, args, ,
"-*- texinfo -*-\n\
@deftypefn {Function File} esn_init(@var{esn})\n\
Inits the Echo State Network with the before selected\n\
initialization algorithm.\n\n\
@seealso{esn_set}\n\
@end deftypefn\n\
")
{
  if( args.length() != 1 || args(0).type_name() != "esn")
  {
    error("Argument must be an ESN.\n");
    return octave_value(-1);
  }

  // get the ESN
  const octave_base_value& rep = args(0).get_rep();
  oct_esn& esn = ((oct_esn &)rep);

  try
  {
    esn.net().init();
  }
  catch(AUExcept e)
  {
    error( e.what().c_str() );
    return octave_value(-1);
  }

  return octave_value();
}

DEFUN_DLD(esn_set, args, ,
"-*- texinfo -*-\n\
@deftypefn {Function File} esn_set(@var{parameter},@var{value})\n\
Set parameter of the Echo State Network.\n\n\
@seealso{esn_init}\n\
@end deftypefn\n\
")
{
  if( args.length() != 3 || args(0).type_name() != "esn"
      || !args(1).is_string() )
  {
    error("First argument must the ESN, then specify the parameter \
           (as string), last parameter is its value.\n");
    return octave_value(-1);
  }

  const octave_base_value& rep = args(0).get_rep();
  oct_esn& esn = ((oct_esn &)rep);

  // check parameter value
  if( !args(2).is_real_scalar() && !args(2).is_string() )
  {
    error("Last parameter must be a scalar or string.\n");
    return octave_value(-1);
  }

  // convert parameter value to string
  std::string value;
  std::ostringstream o;
  if( args(2).is_real_scalar() )
  {
    o << args(2).double_value();
    value = o.str();
  }
  else value = args(2).string_value();

  try
  {
    esn.net().setParameter( args(1).string_value(), value );
  }
  catch(AUExcept e)
  {
    error( e.what().c_str() );
    return octave_value(-1);
  }

  return octave_value();
}
