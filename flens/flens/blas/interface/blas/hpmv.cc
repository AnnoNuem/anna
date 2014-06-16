#include <flens/blas/interface/blas/config.h>


using namespace flens;

extern "C" {

void
BLAS(chpmv)(const char      *UPLO,
            const INTEGER   *N,
            const cfloat    *ALPHA,
            const cfloat    *AP,
            const cfloat    *X,
            const INTEGER   *INCX,
            const cfloat    *BETA,
            cfloat          *Y,
            const INTEGER   *INCY)
{
#   ifdef TEST_DIRECT_CBLAS

    char    _UPLO   = toupper(*UPLO);

    StorageUpLo    upLo   = StorageUpLo(_UPLO);

    cblas_chpmv(CBLAS_ORDER::CblasColMajor,
                cxxblas::CBLAS::getCblasType(upLo),
                *N,
                reinterpret_cast<const float *>(ALPHA),
                reinterpret_cast<const float *>(AP),
                reinterpret_cast<const float *>(X), *INCX,
                reinterpret_cast<const float *>(BETA),
                reinterpret_cast<float *>(Y), *INCY);

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
                info = 6;
            } else if (*INCY==0) {
                info = 9;
            }
            if (info!=0) {
                BLAS(xerbla)("CHPMV ", &info);
                return;
            }
#       endif

        StorageUpLo  upLo = StorageUpLo(_UPLO);

        CHpMatrixConstView    A(CPackedConstView(*N, AP), upLo);
        CDenseVectorConstView x(CConstArrayView(*N, X, abs(*INCX)), *INCX<0);
        CDenseVectorView      y(CArrayView(*N, Y, abs(*INCY)), *INCY<0);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha = *ALPHA;
            const auto beta  = *BETA;

            y = beta*y + alpha*A*x;
#       else
            blas::mv(*ALPHA, A, x, *BETA, y);
#       endif
#   endif
}

void
BLAS(zhpmv)(const char      *UPLO,
            const INTEGER   *N,
            const cdouble   *ALPHA,
            const cdouble   *AP,
            const cdouble   *X,
            const INTEGER   *INCX,
            const cdouble   *BETA,
            cdouble         *Y,
            const INTEGER   *INCY)
{
#   ifdef TEST_DIRECT_CBLAS

        char    _UPLO   = toupper(*UPLO);

        StorageUpLo    upLo   = StorageUpLo(_UPLO);

        cblas_zhpmv(CBLAS_ORDER::CblasColMajor,
                    cxxblas::CBLAS::getCblasType(upLo),
                    *N,
                    reinterpret_cast<const double *>(ALPHA),
                    reinterpret_cast<const double *>(AP),
                    reinterpret_cast<const double *>(X), *INCX,
                    reinterpret_cast<const double *>(BETA),
                    reinterpret_cast<double *>(Y), *INCY);

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
                info = 6;
            } else if (*INCY==0) {
                info = 9;
            }
            if (info!=0) {
                BLAS(xerbla)("ZHPMV ", &info);
                return;
            }
#       endif

        StorageUpLo  upLo = StorageUpLo(_UPLO);

        ZHpMatrixConstView    A(ZPackedConstView(*N, AP), upLo);
        ZDenseVectorConstView x(ZConstArrayView(*N, X, abs(*INCX)), *INCX<0);
        ZDenseVectorView      y(ZArrayView(*N, Y, abs(*INCY)), *INCY<0);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha = *ALPHA;
            const auto beta  = *BETA;

            y = beta*y + alpha*A*x;
#       else
            blas::mv(*ALPHA, A, x, *BETA, y);
#       endif
#   endif
}

} // extern "C"
