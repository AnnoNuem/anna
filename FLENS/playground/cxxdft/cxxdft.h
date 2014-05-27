/*
 *   Copyright (c) 2013, Klaus Pototzky
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PLAYGROUND_CXXDFT_CXXDFT_H
#define PLAYGROUND_CXXDFT_CXXDFT_H 1

// FFTW interface for FFTW
#ifdef WITH_FFTW
#    ifndef HAVE_FFTW
#        define HAVE_FFTW
#    endif
#    ifndef HAVE_FFTW_DOUBLE
#        define HAVE_FFTW_DOUBLE
#    endif
#endif // WITH_FFTW

#ifdef WITH_FFTW_FLOAT
#    ifndef HAVE_FFTW
#        define HAVE_FFTW
#    endif
#    ifndef HAVE_FFTW_FLOAT
#        define HAVE_FFTW_FLOAT
#    endif
#endif // WITH_FFTW_FLOAT

#ifdef WITH_FFTW_DOUBLE
#    ifndef HAVE_FFTW
#        define HAVE_FFTW
#    endif
#    ifndef HAVE_FFTW_FLOAT
#        define HAVE_FFTW_FLOAT
#    endif
#endif // WITH_FFTW_DOUBLE

#ifdef WITH_FFTW_LONGDOUBLE
#    ifndef HAVE_FFTW
#        define HAVE_FFTW
#    endif
#    ifndef HAVE_FFTW_LONGDOUBLE
#        define HAVE_FFTW_LONGDOUBLE
#    endif
#endif // WITH_FFTW_LONGDOUBLE

#ifdef WITH_FFTW_QUAD
#    ifndef HAVE_FFTW
#        define HAVE_FFTW
#    endif
#    ifndef HAVE_FFTW_QUAD
#        define HAVE_FFTW_QUAD
#    endif
#    include <quadmath.h>
#endif // WITH_FFTW_QUAD

#ifdef HAVE_FFTW
#   include "fftw3.h"
#endif

#ifndef FFTW_PLANNER_FLAG
#   define FFTW_PLANNER_FLAG FFTW_ESTIMATE
#endif
#ifndef FFTW_WISDOM_FILENAME
#    define FFTW_WISDOM_FILENAME ""
#endif

#include <playground/cxxdft/direction.h>
#include <playground/cxxdft/single.h>
#include <playground/cxxdft/multiple.h>

#endif // PLAYGROUND_CXXDFT_CXXDFT_H
