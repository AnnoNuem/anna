// example of fftw usage with aureservoir

#include "aureservoir/aureservoir.h"

#include <iostream>
#include <complex>
#include <math.h>

using namespace flens;

typedef DenseVector<Array<float> >  DEVector;
typedef DenseVector<Array< complex<float> > >  CDEVector;

int main(int argc, char *argv[])
{
  DEVector x(25), y(25);
  std::fill_n(x.data(), 25, 0);
  std::fill_n(y.data(), 25, 0);
  x(10) = 1;
  y(19) = 1; // delay = 9


  // calc fftsize
  int L = x.length() > y.length() ? x.length() : y.length();
  int fftsize = (int) pow( 2, ceil(log(L)/log(2)) ); // next power of 2

  // calc fft
  CDEVector X,Y;
  aureservoir::rfft(x, X, fftsize);
  aureservoir::rfft(y, Y, fftsize);

  // calc delay with GCC
  int delay = aureservoir::CalcDelay<float>::gcc(X,Y,10,0);
  std::cout << "delay ist: " << delay << "\n";

  return 0;
}
