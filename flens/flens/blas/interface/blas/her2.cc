#include <flens/blas/interface/blas/config.h>


using namespace flens;

extern "C" {

void
BLAS(cher2)(const char      *UPLO,
            const INTEGER   *N,
            const cfloat    *ALPHA,
            const cfloat    *X,
            const INTEGER   *INCX,
            const cfloat    *Y,
            const INTEGER   *INCY,
            cfloat          *_A,
            const INTEGER   *LDA)
{
#   ifdef TEST_DIRECT_CBLAS

        char    _UPLO   = toupper(*UPLO);

        StorageUpLo    upLo   = StorageUpLo(_UPLO);

        cblas_cher2(CBLAS_ORDER::CblasColMajor,
                    cxxblas::CBLAS::getCblasType(upLo),
                    *N,
                    reinterpret_cast<const float *>(ALPHA),
                    reinterpret_cast<const float *>(X), *INCX,
                    reinterpret_cast<const float *>(Y), *INCY,
                    reinterpret_cast<float *>(_A), *LDA);

#   else

        using std::abs;
        using std::max;

        char _UPLO = toupper(*UPLO);

#       ifndef NO_INPUT_CHECK
            INTEGER info  = 0;

            if (_UPLO!='U' && _UPLO!='L') {
                info = 1;
            } else if (*N<0) {
                info = 2;
            } else if (*INCX==0) {
                info = 5;
            } else if (*INCY==0) {
                info = 7;
            } else if (*LDA<max(INTEGER(1),*N)) {
                info = 9;
            }
            if (info!=0) {
                BLAS(xerbla)("CHER2 ", &info);
                return;
            }
#       endif

        StorageUpLo  upLo = StorageUpLo(_UPLO);

        CDenseVectorConstView x(CConstArrayView(*N, X, abs(*INCX)), *INCX<0);
        CDenseVectorConstView y(CConstArrayView(*N, Y, abs(*INCY)), *INCY<0);
        CHeMatrixView         A(CFullView(*N, *N, _A, *LDA), upLo);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha  = *ALPHA;
            const auto alpha_ = conj(alpha);

            A += alpha*x*conjTrans(y) + alpha_*y*conjTrans(x);
#       else
            blas::r2(*ALPHA, x, y, A);
#       endif
#   endif
}

void
BLAS(zher2)(const char      *UPLO,
            const INTEGER   *N,
            const cdouble   *ALPHA,
            const cdouble   *X,
            const INTEGER   *INCX,
            const cdouble   *Y,
            const INTEGER   *INCY,
            cdouble         *_A,
            const INTEGER   *LDA)
{
#   ifdef TEST_DIRECT_CBLAS

        char    _UPLO   = toupper(*UPLO);

        StorageUpLo    upLo   = StorageUpLo(_UPLO);

        cblas_zher2(CBLAS_ORDER::CblasColMajor,
                    cxxblas::CBLAS::getCblasType(upLo),
                    *N,
                    reinterpret_cast<const double *>(ALPHA),
                    reinterpret_cast<const double *>(X), *INCX,
                    reinterpret_cast<const double *>(Y), *INCY,
                    reinterpret_cast<double *>(_A), *LDA);

#   else

        using std::abs;
        using std::max;

        char    _UPLO = toupper(*UPLO);

#       ifndef NO_INPUT_CHECK
            INTEGER info  = 0;

            if (_UPLO!='U' && _UPLO!='L') {
                info = 1;
            } else if (*N<0) {
                info = 2;
            } else if (*INCX==0) {
                info = 5;
            } else if (*INCY==0) {
                info = 7;
            } else if (*LDA<max(INTEGER(1),*N)) {
                info = 9;
            }
            if (info!=0) {
                BLAS(xerbla)("ZHER2 ", &info);
                return;
            }
#       endif

        StorageUpLo  upLo = StorageUpLo(_UPLO);

        ZDenseVectorConstView x(ZConstArrayView(*N, X, abs(*INCX)), *INCX<0);
        ZDenseVectorConstView y(ZConstArrayView(*N, Y, abs(*INCY)), *INCY<0);
        ZHeMatrixView         A(ZFullView(*N, *N, _A, *LDA), upLo);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha  = *ALPHA;
            const auto alpha_ = conj(alpha);

            A += alpha*x*conjTrans(y) + alpha_*y*conjTrans(x);
#       else
            blas::r2(*ALPHA, x, y, A);
#       endif
#   endif
}

} // extern "C"
