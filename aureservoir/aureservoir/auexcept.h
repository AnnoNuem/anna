/***************************************************************************/
/*!
 *  \file   auexcept.h
 *
 *  \brief  implements exception class for aureservoir library
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

#ifndef AURESERVOIR_EXCEPTION_H__
#define AURESERVOIR_EXCEPTION_H__

#include <string>

namespace aureservoir
{

/*!
 * \class AUExcept
 *
 * Exception Class for aureservoir library
 * The what() method returns the error string.
 */
class AUExcept
{
 protected:
  /// the exception string
  std::string message_;

 public:
  /// construction of the exception
  AUExcept(const std::string &message) { message_ = message; }
  virtual ~AUExcept() { }

  /// returns the error string
  virtual std::string what() { return message_; }
};

} // end of namespace aureservoir

#endif // AURESERVOIR_EXCEPTION_H__
