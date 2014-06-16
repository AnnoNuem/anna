/*
 *   Copyright (c) 2011, Michael Lehn
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

/*  Based on
 *
      SUBROUTINE DGEQRS( M, N, NRHS, A, LDA, TAU, B, LDB, WORK, LWORK,
     $                   INFO )
      SUBROUTINE ZGEQRS( M, N, NRHS, A, LDA, TAU, B, LDB, WORK, LWORK,
     $                   INFO )
 *
 *  -- LAPACK routine (version 3.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
 *     Courant Institute, Argonne National Lab, and Rice University
 *     February 29, 1992
 */

#ifndef FLENS_LAPACK_GE_QRS_TCC
#define FLENS_LAPACK_GE_QRS_TCC 1

#include <flens/blas/blas.h>
#include <flens/lapack/lapack.h>

namespace flens { namespace lapack {

//-- qrs [real] ----------------------------------------------------------------
template <typename MA, typename VTAU, typename MB, typename VWORK>
typename RestrictTo<IsRealGeMatrix<MA>::value
                 && IsRealDenseVector<VTAU>::value
                 && IsRealGeMatrix<MB>::value
                 && IsRealDenseVector<VWORK>::value,
         void>::Type
qrs(MA &&A, const VTAU &tau, MB &&B, VWORK &&work)
{
    ASSERT(work.length()>=B.numCols());

    typedef typename RemoveRef<MA>::Type   MatrixA;

    typedef typename MatrixA::IndexType    IndexType;
    typedef typename MatrixA::ElementType  T;

    const Underscore<IndexType> _;

    const IndexType m = A.numRows();
    const IndexType n = A.numCols();
    const IndexType nRhs = B.numCols();

//
//  Quick return if possible
//
    if ((n==0) || (nRhs==0) || (m==0)) {
        return;
    }
//
//  B := Q' * B
//
    ormqr(Left, Trans, A, tau, B, work);
//
//  Solve R*X = B(1:n,:)
//
    blas::sm(Left, NoTrans, T(1), A.upper(), B);
}

//-- qrs [complex] -------------------------------------------------------------
template <typename MA, typename VTAU, typename MB, typename VWORK>
typename RestrictTo<IsComplexGeMatrix<MA>::value
                 && IsComplexDenseVector<VTAU>::value
                 && IsComplexGeMatrix<MB>::value
                 && IsComplexDenseVector<VWORK>::value,
         void>::Type
qrs(MA &&A, const VTAU &tau, MB &&B, VWORK &&work)
{
    ASSERT(work.length()>=B.numCols());

    typedef typename RemoveRef<MA>::Type   MatrixA;

    typedef typename MatrixA::IndexType    IndexType;
    typedef typename MatrixA::ElementType  T;

    const Underscore<IndexType> _;

    const IndexType m = A.numRows();
    const IndexType n = A.numCols();
    const IndexType nRhs = B.numCols();

//
//  Quick return if possible
//
    if ((n==0) || (nRhs==0) || (m==0)) {
        return;
    }
//
//  B := Q' * B
//
    unmqr(Left, ConjTrans, A, tau, B, work);
//
//  Solve R*X = B(1:n,:)
//
    blas::sm(Left, NoTrans, T(1), A.upper(), B);
}

} } // namespace lapack, flens

#endif // FLENS_LAPACK_GE_QRS_TCC
