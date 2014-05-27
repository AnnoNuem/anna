/*
 *   Copyright (c) 2012, Klaus Pototzky
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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_DOT_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_DOT_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsReal<T>::value &&
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
dotu(IndexType n,
     const T *x, IndexType incX,
     const T *y, IndexType incY,
     T &result)
{
    CXXBLAS_DEBUG_OUT("dotu_intrinsic [real, " INTRINSIC_NAME "]");

    if (incX==1 && incY==1) {

        result = T(0);

        typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL> IntrinsicType;
        const int numElements = IntrinsicType::numElements;

        IndexType i=0;

        IntrinsicType _x, _y;
        IntrinsicType _result;
        _result.setZero();

        for (; i+numElements-1<n; i+=numElements) {
            _x.loadu(x+i);
            _y.loadu(y+i);

            _result = _intrinsic_add(_result, _intrinsic_mul(_x, _y));
        }

        T tmp_result[numElements];
        _result.storeu(tmp_result);

        for (IndexType k=0; k<numElements; ++k) {
            result += tmp_result[k];
        }

        for (;i<n; ++i) {
            result += x[i]*y[i];
        }

    } else {

        cxxblas::dotu<IndexType, T, T ,T>(n, x, incX, y, incY, result);

    }
}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsComplex<T>::value &&
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
dotu(IndexType n,
     const T *x, IndexType incX,
     const T *y, IndexType incY,
     T &result)
{

    CXXBLAS_DEBUG_OUT("dotu_intrinsic [complex, " INTRINSIC_NAME "]");

    if (incX==1 && incY==1) {

        result = T(0);

        typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL>     IntrinsicType;
        typedef typename IntrinsicType::PrimitiveDataType  PT;

        typedef Intrinsics<PT, DEFAULT_INTRINSIC_LEVEL>  IntrinsicPrimitiveType;

        const int numElements = IntrinsicType::numElements;

        IndexType i=0;

        IntrinsicType _x, _y;
        IntrinsicPrimitiveType _real_y, _imag_y;
        IntrinsicType _result;
        _result.setZero();

        for (; i+numElements-1<n; i+=numElements) {

            _x.loadu(x+i);
            _y.loadu(y+i);

            _real_y = _intrinsic_real(_y);
            _imag_y = _intrinsic_imag(_y);

            _result = _intrinsic_add(_result, _intrinsic_mul(_x, _real_y));

            _x = _intrinsic_swap_real_imag(_x);

            _result = _intrinsic_addsub(_result, _intrinsic_mul(_x, _imag_y));
        }

        T tmp_result[numElements];
        _result.storeu(tmp_result);

        for (IndexType k=0; k<numElements; ++k) {
            result += tmp_result[k];
        }

        for (;i<n; ++i) {
            result += x[i]*y[i];
        }

    } else {

        cxxblas::dotu<IndexType, T, T ,T>(n, x, incX, y, incY, result);

    }
}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsReal<T>::value &&
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
dot(IndexType n,
    const T *x, IndexType incX,
    const T *y, IndexType incY,
    T &result)
{
    CXXBLAS_DEBUG_OUT("dot_intrinsic [real, " INTRINSIC_NAME "]");

    cxxblas::dotu(n, x, incX, y, incY, result);

}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsComplex<T>::value &&
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
dot(IndexType n,
    const T *x, IndexType incX,
    const T *y, IndexType incY,
    T &result)
{
    CXXBLAS_DEBUG_OUT("dot_intrinsic [complex, " INTRINSIC_NAME "]");

    using std::conj;

    result = T(0);

    if (incX==1 && incY==1) {

        typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL>     IntrinsicType;
        typedef typename IntrinsicType::PrimitiveDataType  PT;

        typedef Intrinsics<PT, DEFAULT_INTRINSIC_LEVEL>  IntrinsicPrimitiveType;

        const int numElements = IntrinsicType::numElements;

        IndexType i=0;

        IntrinsicType _x, _y;
        IntrinsicPrimitiveType _real_y, _imag_y;
        IntrinsicType _result;
        _result.setZero();

        for (; i+numElements-1<n; i+=numElements) {

            _x.loadu(x+i);
            _y.loadu(y+i);

            _real_y = _intrinsic_real(_y);
            _imag_y = _intrinsic_imag(_y);


            _result = _intrinsic_addsub(_result, _intrinsic_mul(_x, _real_y));

            _x = _intrinsic_swap_real_imag(_x);

            _result = _intrinsic_sub(_result, _intrinsic_mul(_x, _imag_y));
        }

        T tmp_result[numElements];
        _result.storeu(tmp_result);

        for (IndexType k=0; k<numElements; ++k) {
            result -= tmp_result[k];
        }

        for (;i<n; ++i) {
            result += conj(x[i])*y[i];
        }


    } else {
        cxxblas::dot<IndexType, T, T ,T>(n, x, incX, y, incY, result);
    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_DOT_TCC
