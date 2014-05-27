/***************************************************************************/
/*!
 *  \file   denormal.h
 *
 *  \brief  functions for denormal handling
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   Sept 2007
 *
 *   Here are some hardware and software based solutions for denormal
 *   handling.
 *   An overview of software based solutions can be found in Laurent de
 *   Soras:
 *   "Denormal numbers in floating point signal processing applications"
 *   \sa http://www.musicdsp.org/files/denormal.pdf
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

#ifndef AURESERVOIR_DENORMAL_H__
#define AURESERVOIR_DENORMAL_H__

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#include <flens/flens.h>
#include "auexcept.h"

namespace aureservoir
{

/// DC offset to prevent denormals for single precision
/// \todo check the exact value
const float SINGLE_DENORMAL_DC = 1.0E-25;

/// DC offset to prevent denormals for double precision
/// \todo check the exact value
const double DOUBLE_DENORMAL_DC = 1.0E-30;

/*!
 * \brief deactivate pentium4 denormals
 *
 * This deactivates denormals on pentium4 processors, which speeds up
 * quite a lot. However, this only works for the SSE unit and is not
 * supported by older models, so all used libraries have to be compiled
 * with -mfpmath=sse -msse flags.
 * To turn denormals off on SSE, we turn on the Denormals Are Zero and
 * Flush to Zero (DAZ and FZ) bits in the MXCSR register.
 *
 * \sa http://developer.apple.com/documentation/Performance/Conceptual/Accelerate_sse_migration/migration_sse_translation/chapter_4_section_2.html
 *
 * \note For some strange reasons this function must be inline, otherwise
 *       I get linker problems !
 */
inline void set_denormal_flags()
  throw(AUExcept)
{
#ifdef __SSE__

  unsigned long cpuflags = 0;

#ifndef USE_X86_64_ASM

  asm volatile (
    "mov $1, %%eax\n"
    "pushl %%ebx\n"
    "cpuid\n"
    "movl %%edx, %0\n"
    "popl %%ebx\n"
    : "=r" (cpuflags)
    :
    : "%eax", "%ecx", "%edx", "memory"
  );

#else

  asm volatile (
    "pushq %%rbx\n"
    "movq $1, %%rax\n"
    "cpuid\n"
    "movq %%rdx, %0\n"
    "popq %%rbx\n"
    : "=r" (cpuflags)
    : 
    : "%rax", "%rcx", "%rdx", "memory"
  );

#endif // USE_X86_64_ASM

  if (! (cpuflags & 1<<25) )
    throw AUExcept("set_denormal_flag: your processor doesn't have SSE support, DAZ and FZ denormal handling not activated !");

  // do we need SSE2 ?
//   if (! (cpuflags & 1<<26) )
//     throw AUExcept("set_denormal_flag: your processor doesn't have SSE2 support, DAZ and FZ denormal handling not activated !");

  // set DAZ and FZ bits
  int oldMXCSR = _mm_getcsr(); //read the old MXCSR setting
  int newMXCSR = oldMXCSR | 0x8040; // set DAZ and FZ bits
  _mm_setcsr( newMXCSR ); //write the new MXCSR setting to the MXCSR

#else

  throw AUExcept("set_denormal_flag: you did not compile with SSE support (-mfpmath=sse -msse), DAZ and FZ denormal handling not activated !");

#endif // __SSE__
}

/*!
 * adds a constant value to prevent going to denormal mode
 * @param data pointer to single precision data
 * @param size of the data
 * \sa http://www.musicdsp.org/files/other001.txt
 */
inline void denormals_add_dc(float *data, int size)
{
  for(int i=0; i<size; ++i)
    data[i] += SINGLE_DENORMAL_DC;
}

/*!
 * adds a constant value to prevent going to denormal mode
 * @param data pointer to double precision data
 * @param size of the data
 * \sa http://www.musicdsp.org/files/other001.txt
 */
inline void denormals_add_dc(double *data, int size)
{
  for(int i=0; i<size; ++i)
    data[i] += DOUBLE_DENORMAL_DC;
}

} // end of namespace aureservoir

#endif // AURESERVOIR_DENORMAL_H__
