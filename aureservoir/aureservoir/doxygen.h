/***************************************************************************/
/*!
 *  \file   doxygen.h
 *
 *  \brief  doxygen documentation main page
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   Feb 2008
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

/*!

\mainpage aureservoir

<br>
<b> Efficient C++ library for analog reservoir computing neural networks
(Echo State Networks). </b> <br>
2007-2008, <a href="http://grh.mur.at/">Georg Holzmann</a>


<br>
\section contents_sec Contents
<ul>
  <li> \ref intro_sec </li>
  <li> \ref features_sec </li>
  <li> \ref download_sec </li>
  <li> \ref install_sec </li>
  <li> \ref examples_sec </li>
  <li> \ref tests_sec </li>
  <li> \ref contact_sec </li>
</ul>


<br>
\section intro_sec Introduction

Reservoir computing is a recent kind of recurrent neural network computation,
where only the output weights are trained.
This has the big advantage that training is a simple linear regression task
and one cannot get into a local minimum.
Such a network consists of a randomly created, fixed, sparse recurrent
reservoir and a trainable output layer connected to this reservoir.
Most known types are the "Echo State Network" and the "Liquid State Machine", which achieved very promising results on various machine learning benchmarks.

This library should be an open source (L-GPL) and very efficient
implementation of Echo State Networks
with bindings to scientific computation packages (so far to
<a href="http://www.scipy.org/">python/numpy</a>,
<a href="http://www.puredata.info/">Pure Data</a> and
<a href="http://www.octave.org/">octave</a> are in work, everyone is
invited to make a Matlab binding) 
for offline and realtime simulations.
It can be extended in an easy way with new simulation,
training and adaptation algorithms, which are function objects and
automatically used by the main classes.

For a theoretical overview and some papers about Echo State Networks see:
<a href="http://www.scholarpedia.org/article/Echo_State_Network">
Echo State Networks</a>.


<br>
\section features_sec Features

The library can be used with double or singe precision floating points
and the interface of the main Echo State Network class is documented
at aureservoir::ESN. <br>
All different activation functions, simulation, training and
adaptation algorithms, etc. can be changed at runtime.

Implemented simulation algorithms (see simulate.h):
<ul>
  <li> standard simulation algorithm as in Jaeger's initial paper
       (see aureservoir::SimStd) </li>
  <li> simulation algorithm with leaky integrator neurons
       (see aureservoir::SimLI) </li>
  <li> algorithm with bandpass style neurons as introduced by Wustlich
       and Siewert (see aureservoir::SimBP) </li>
  <li> simulation algorithm with general IIR-Filter neurons
       (see aureservoir::SimFilter) </li>
  <li> algorithm with IIR-Filter before neurons nonlinearity
       (see aureservoir::SimFilter2) </li>
  <li> ESNs with an additional delay&sum readout
       (see aureservoir::SimFilterDS) </li>
  <li> simulation algorithm with additional squared state updates
       (see aureservoir::SimSquare) </li>
</ul>

Implemented training algorithms (see train.h):
<ul>
  <li> offline trainig algorithm using the pseudo inverse
       (see aureservoir::TrainPI) </li>
  <li> training algorithm using the least square solution
       (see aureservoir::TrainLS) </li>
  <li> algorithm with Ridge Regression / Tikhonov Regularization
       (see aureservoir::TrainRidgeReg) </li>
  <li> offline algorithm for delay&sum readout with pseudo inverse
       (see aureservoir::TrainDSPI) </li>
</ul>

Implemented reservoir adaptation algorithms:
<ul>
  <li> Gaussian-IP reservoir adaptation method for tanh neurons
       by David Verstraeten, Benjamin Schrauwen and Dirk Stroobandt
       (see aureservoir::ESN::adapt) </li>
</ul>

New algorithms can be added by just deriving from the appropriate
base class and overloading the relevant methods.


<br>
\section download_sec Download

Aureservoir is free software. You can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation. Either
version 2.1 of the License, or (at your option) any later version.

Releases of aureservoir can be found at
<a href="http://sourceforge.net/projects/aureservoir/">
http://sourceforge.net/projects/aureservoir/</a>.

To get the latest source of aureservoir one can check out the
SVN repository with the following instruction set: <br>
\verbatim
svn co https://aureservoir.svn.sourceforge.net/svnroot/aureservoir aureservoir
\endverbatim
If you don't know how to use SVN checkout the description at
<a href="http://sourceforge.net/svn/?group_id=209855">
aureservoir subversion page </a>.

The sourcefore project page of aureservoir is
<a href="http://sourceforge.net/projects/aureservoir/">
http://sourceforge.net/projects/aureservoir/</a>,
there you can also browse the code online at
<a href="http://aureservoir.svn.sourceforge.net/viewvc/aureservoir/">
SVN browse</a>.


<br>
\section install_sec Installation and Compilation

To get aureservoir finally working one needs to manage two steps:
first the installation of all dependencies, second the compilation
of aureservoir itself. <br>
Here I will describe only the procedure of compiling the aureservoir
library and its python bindings.

<b>Installation of all dependencies:</b>
<ul>

<li>
Aureservoir uses the C++ library
<a href="http://flens.sourceforge.net/">FLENS</a>, which is very fast
and elegant to use (see my benchmarks at
<a href="http://grh.mur.at/misc/sparselib_benchmark/">
 Sparse Matrix Library Benchmark</a>). <br>
Therefore you have to install the CVS version of FLENS first.
Detailed instructions on how to do that can be found at
<a href="http://flens.sourceforge.net/obtain.html">
Obtain and Install FLENS </a>.
</li>

<li>
<a href="http://www.python.org/">Python</a> and
<a href="http://numpy.scipy.org/">Numpy</a> must be installed.
</li>

<li>
Additionally <a href="http://www.fftw.org/">FFTW3</a>
is needed for FFT calculation.
</li>

<li>
Aureservoir uses <a href="http://www.scons.org/">SCons</a>
as build tool, you need that too.
</li>

</ul>

<b>Compilation of aureservoir:</b>

<ul>

<li>
The C++ aureservoir library consists of header files with template classes
only, therefore you don't have to compile the library itself.
</li>

<li>
Small intro how to use SCons: <br>
- In general SCons checks for all dependencies and the compilation
  can be started with the command <b><i>scons</i></b>.
- To install the compiled binary use <b><i>scons install</i></b>
  with root privileges.
- Additionally you can set an alternative path for installed libraries
  and some optimization flags, the available options can be seen with
  <b><i>scons -h</i></b>.
- Then you can set these option in example with
  <b><i>scons arch=pentium4</i></b>. The options will be stored so that you only
  have to set them once.
</li>

<li>
<b>Important:</b> You should first try to compile the C++ example
in aureservoir/examples, because it gives more debug information in
case something is going wrong ! <br>
To do so go into the directory aureservoir/examples and compile it with
the command <b><i>scons</i></b> or use <b><i>scons -h</i></b> to see
which additional options can be set.
</li>

<li>
Now change to aureservoir/python and try to compile the python bindings. <br>
Again type <b><i>scons</i></b> or <b><i>scons -h</i></b> to see
additional options and compile the library. <br>
Afterward type <b><i>scons install</i></b> with root privileges if you
want to install it system wide.
</li>

<li>
That's it, you should be able to use it now, so have a look at
the \ref examples_sec .<br>
You can also try if all the included tests are successful, change to
aureservoir/python/tests and type <b><i>python run_all_tests.py</i></b>.
</li>

</ul>

Let me know if you have any problems, I am sure we will find a solution !


<br>
\section examples_sec Examples

Most of the examples are in python and located in
aureservoir/python/examples. <br>
However, as the aureservoir python syntax is the same as in C++
one can use these examples analogous in C++.

Following python examples are included ATM:
<ul>
  <li> \ref slow_sine.py <br>
       slow sine genration task with standard and bandpass ESN </li>

  <li> \ref narma10.py <br>
      a 10th order NARMA system identification task
      with additional squared state updates</li>

  <li> \ref singleneuron_sinosc.py <br>
      with a delay&sum readout it's possible to train
      a sine oscillator with just one neuron</li>

  <li> \ref multiple_sines.py <br>
      trains an ESN to be a generator for a sum of 10 sines
      or a product of 5 sines</li>

  <li> \ref sparse_nonlin_system_identification.py <br>
      sparse nonlinear system identification with long-term
      dependencies of 2 example systems</li>
</ul>

Also one very basic C++ example is included:
<ul>
  <li> \ref esn_example.cpp <br>
       shows basic ESN usage in C++ </li>
</ul>

See also <a href="examples.html">examples</a>.


<br>
\section tests_sec Unit Tests

The algorithms are extensively tested.
All are recalculated in python and tested against the C++ implementation.

The python tests are in the directory aureservoir/python/tests.
However, there are still some (old) C++ tests in aureservoir/tests
using <a href="http://sourceforge.net/projects/cppunit">cppunit</a>.

To run all the python unit tests change to aureservoir/python/tests
and type <b><i>python run_all_tests.py</i></b>.
This should give you an <i>OK</i> at the bottom of the output if
everything works.


<br>
\section contact_sec Feedback and Contact

Please don't hesitate to report problems, requests or any other
feedback to grh _at_ mur _dot_ at.

Feel free to also use the bug and feature request tracker
at the sourceforge page
<a href="http://sourceforge.net/tracker/?group_id=209855">
aureservoir tracker</a>.
*/
