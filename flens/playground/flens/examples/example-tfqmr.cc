#include <complex>
#include <iostream>
#define USE_PLAYGROUND
#include <flens/flens.cxx>

using namespace std;
using namespace flens;

typedef complex<double>   T;

int
main()
{
    ///
    /// Define convenient matrix/vector types ...
    ///
    typedef GeMatrix<FullStorage<T> >           Matrix;
    typedef DenseVector<Array<T> >              Vector;
    typedef Matrix::IndexType                   IndexType;
    typedef DenseVector<Array<IndexType> >      IndexVector;

    ///
    /// Then we setup another toy problem
    ///
    const IndexType n = 3;

    Matrix         A(n,n);
    Vector         x(n), b(n);


    A = T(1, 0), T( 1,-1), T(  2,20),
        T(0, 1), T( 1, 2), T(-10, 8),
        T(0,-1), T(-1, 1), T( 40, 6);

    b = T( 1, 0),
        T(-1, 1),
        T(-2,-1);

    cerr << "A.upper().symmetric() = " << A.upper().symmetric() << endl;
    cerr << "b = " << b << endl;

    ///
    /// solve A*x = b using transpose-free quasi minimal residual solver
    ///
    solver::bicgstab(A.upper().symmetric(), x, b);
    cout << "x = " << x << endl;

    ///
    /// Test the result
    ///
    b = A.upper().symmetric()*x;
    cout << "b = " << b << endl;

    return 0;
}
