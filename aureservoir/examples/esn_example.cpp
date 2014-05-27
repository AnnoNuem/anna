/***************************************************************************/
/*!
 *  \file   esn_example.cpp
 *
 *  \brief  example usage of the ESN class
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

#include "aureservoir/aureservoir.h"

#define TYPE double

#include <iostream>
#include <complex>

using namespace aureservoir;
using namespace std;

int main(int argc, char *argv[])
{
  ESN< TYPE > net;

  try
  {
    cout << "## INITIALIZATION ##\n";

    int train_size = 50;
    int ins = 3;
    int outs = 2;

    net.setSize(15);
    net.setInputs(ins);
    net.setOutputs(outs);
    net.setInitParam(CONNECTIVITY, 0.8);
    net.setInitParam(IN_CONNECTIVITY, 0.6);
    net.setInitParam(FB_CONNECTIVITY, 0.4);

    net.init();

    // print current net parameters
    net.post();
    cout << endl << "input weights W_in: " << net.getWin();
    cout << endl << "feedback weights W_back: " << net.getWback();
//     cout << endl << "reservoir weight matrix W: " << net.getW() << endl;


    cout << "\n## TRAINING ##\n";

    ESN<TYPE>::DEMatrix in(ins,train_size), out(outs,train_size);

    for(int i=1; i<=train_size; ++i)
    {
      for(int j=1; j<=ins; ++j)
        in(j,i) = Rand<TYPE>::uniform();

      for(int j=1; j<=outs; ++j)
        out(j,i) = Rand<TYPE>::uniform();
    }

    net.train(in, out, 20);

    cout << "\ntrained output weights W_out: " << net.getWout() << endl;


    cout << "## SIMULATION ##\n";

    int run_size = 10;
    ESN<TYPE>::DEMatrix indata(ins,run_size), result(outs,run_size);

    for(int i=1; i<=run_size; ++i)
    {
      for(int j=1; j<=ins; ++j)
        indata(j,i) = Rand<>::uniform();
    }

    net.simulate( indata, result );

    cout << endl << "simulation results: " << result << endl;
  }
  catch(AUExcept e)
  { cout << "Exception: " << e.what() << endl; }

  return 0;
}
