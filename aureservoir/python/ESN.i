
// File: classaureservoir_1_1ESN.xml
%feature("docstring") ESN "

class for a basic Echo State Network

This class implements a basic Echo State Network as described in
articles by Herbert Jaeger on the following page: See:
http://www.scholarpedia.org/article/Echo_State_Network  The template
argument T can be float or double. Single Precision (float) saves
quite some computation time.

The \"echo state\" approach looks at RNNs from a new angle. Large RNNs
are interpreted as \"reservoirs\" of complex, excitable dynamics.
Output units \"tap\" from this reservoir by linearly combining the
desired output signal from the rich variety of excited reservoir
signals. This idea leads to training algorithms where only the
network-to-output connection weights have to be trained. This can be
done with known, highly efficient linear regression algorithms. from
See:  http://www.faculty.iu-bremen.de/hjaeger/esn_research.html

C++ includes: esn.h ";

/*  algorithms are friends  */

/*  Algorithm interface  */

%feature("docstring")  ESN::init "throw ( AUExcept)
Initialization Algorithm for an Echo State Network See:  class
InitBase ";

%feature("docstring")  ESN::adapt "throw ( AUExcept)
Reservoir Adaptation Algorithm Interface At the moment this is only
the Gaussian-IP reservoir adaptation method for tanh neurons. See:
\"Adapting reservoirs to get Gaussian distributions\" by David
Verstraeten, Benjamin Schrauwen and Dirk Stroobandt

Parameters:
-----------

in:  matrix of input values (inputs x timesteps), the reservoir will
be adapted by this number of timesteps.

mean value of differences between all parameters before and after
adaptation, can be used to see if learning still makes an progress. ";

%feature("docstring")  ESN::train "throw ( AUExcept)
Training Algorithm Interface See:  class TrainBase

Parameters:
-----------

in:  matrix of input values (inputs x timesteps)

out:  matrix of desired output values (outputs x timesteps) for
teacher forcing

washout:  washout time in samples, used to get rid of the transient
dynamics of the network starting state ";

%feature("docstring")  ESN::simulate "

Simulation Algorithm Interface See:  class SimBase

Parameters:
-----------

in:  matrix of input values (inputs x timesteps)

out:  matrix for output values (outputs x timesteps) ";

%feature("docstring")  ESN::resetState "

resets the internal state vector x of the reservoir to zero ";

/*  C-style Algorithm interface  */

%feature("docstring")  ESN::adapt "throw ( AUExcept)
C-style Reservoir Adaptation Algorithm Interface (data will be copied
into a FLENS matrix) At the moment this is only the Gaussian-IP
reservoir adaptation method for tanh neurons. See:  \"Adapting
reservoirs to get Gaussian distributions\" by David Verstraeten,
Benjamin Schrauwen and Dirk Stroobandt

Parameters:
-----------

inmtx:  matrix of input values (inputs x timesteps), the reservoir
will be adapted by this number of timesteps.

mean value of differences between all parameters before and after
adaptation, can be used to see if learning still makes an progress. ";

%feature("docstring")  ESN::train "throw ( AUExcept)
C-style Training Algorithm Interface (data will be copied into a FLENS
matrix) See:  class TrainBase

Parameters:
-----------

inmtx:  input matrix in row major storage (usual C array) (inputs x
timesteps)

outmtx:  output matrix in row major storage (outputs x timesteps) for
teacher forcing

washout:  washout time in samples, used to get rid of the transient
dynamics of the network starting state ";

%feature("docstring")  ESN::simulate "throw ( AUExcept)
C-style Simulation Algorithm Interface with some additional error
checking. (data will be copied into a FLENS matrix) See:  class
SimBase

Parameters:
-----------

inmtx:  input matrix in row major storage (usual C array) (inputs x
timesteps)

outmtx:  output matrix in row major storage (outputs x timesteps),

Data must be already allocated! ";

%feature("docstring")  ESN::simulateStep "throw (
AUExcept) C-style Simulation Algorithm Interface, for single step
simulation See:  class SimBase ";

/*  Additional Interface for Bandpass and IIR-Filter Neurons  */

/* */

%feature("docstring")  ESN::setBPCutoff "throw (
AUExcept) Set lowpass/highpass cutoff frequencies for bandpass style
neurons. \" See:  class SimBP

Parameters:
-----------

f1:  vector with lowpass cutoff for all neurons (size = neurons)

f2:  vector with highpass cutoffs (size = neurons) ";

%feature("docstring")  ESN::setBPCutoff "throw (
AUExcept) Set lowpass/highpass cutoff frequencies for bandpass style
neurons \" (C-style Interface).

Parameters:
-----------

f1:  vector with lowpass cutoff for all neurons (size = neurons)

f2:  vector with highpass cutoffs (size = neurons) ";

%feature("docstring")  ESN::setIIRCoeff "throw (
AUExcept) sets the IIR-Filter coefficients, like Matlabs filter
object.

Parameters:
-----------

B:  matrix with numerator coefficient vectors (m x nb) m ... nr of
parallel filters (neurons) nb ... nr of filter coefficients

A:  matrix with denominator coefficient vectors (m x na) m ... nr of
parallel filters (neurons) na ... nr of filter coefficients

seris:  nr of serial IIR filters, e.g. if series=2 the coefficients B
and A will be divided in its half and calculated with 2 serial IIR
filters ";

%feature("docstring")  ESN::setIIRCoeff "throw (
AUExcept) sets the IIR-Filter coefficients, like Matlabs filter
object.

Parameters:
-----------

B:  matrix with numerator coefficient vectors (m x nb) m ... nr of
parallel filters (neurons) nb ... nr of filter coefficients

A:  matrix with denominator coefficient vectors (m x na) m ... nr of
parallel filters (neurons) na ... nr of filter coefficients

seris:  nr of serial IIR filters, e.g. if series=2 the coefficients B
and A will be divided in its half and calculated with 2 serial IIR
filters ";

/*  GET parameters  */

%feature("docstring")  ESN::post "

posts current parameters to stdout ";

%feature("docstring")  ESN::getSize "

reservoir size (nr of neurons) ";

%feature("docstring")  ESN::getInputs "

nr of inputs to the reservoir ";

%feature("docstring")  ESN::getOutputs "

nr of outputs from the reservoir ";

%feature("docstring")  ESN::getNoise "

current noise level ";

%feature("docstring")  ESN::getInitParam "

returns an initialization parametern from the parameter map

Parameters:
-----------

key:  the requested parameter

the value of the parameter ";

%feature("docstring")  ESN::getInitAlgorithm "

initialization algorithm ";

%feature("docstring")  ESN::getTrainAlgorithm "

training algorithm ";

%feature("docstring")  ESN::getSimAlgorithm "

simulation algorithm ";

%feature("docstring")  ESN::getReservoirAct "

reservoir activation function ";

%feature("docstring")  ESN::getOutputAct "

output activation function ";

/*  GET internal data  */

%feature("docstring")  ESN::getWin "

input weight matrix (neurons x inputs) ";

%feature("docstring")  ESN::getW "

reservoir weight matrix (neurons x neurons) ";

%feature("docstring")  ESN::getWback "

feedback (output to reservoir) weight matrix (neurons x outputs) ";

%feature("docstring")  ESN::getWout "

output weight matrix (outputs x neurons+inputs) ";

%feature("docstring")  ESN::getX "

internal state vector x (size = neurons) ";

%feature("docstring")  ESN::getDelays "throw (
AUExcept) query the trained delays in delay&sum readout See:  class
SimFilterDS

matrix with delay form neurons+inputs to all outputs size = (output x
neurons+inputs) ";

/*  GET internal data C-style interface  */

%feature("docstring")  ESN::getWin "

get pointer to input weight matrix data and dimensions (neurons x
inputs) WARNING:  This data is in fortran style column major storage !
";

%feature("docstring")  ESN::getWback "

get pointer to feedback weight matrix data and dimensions (neurons x
outputs) WARNING:  This data is in fortran style column major storage
! ";

%feature("docstring")  ESN::getWout "

get pointer to output weight matrix data and dimensions (outputs x
neurons+inputs) WARNING:  This data is in fortran style column major
storage ! ";

%feature("docstring")  ESN::getX "

get pointer to internal state vector x data and length ";

%feature("docstring")  ESN::getW "throw ( AUExcept)
Copies data of the sparse reservoir weight matrix into a dense C-style
matrix. Memory of the C array must be allocated before!

Parameters:
-----------

wmtx:  pointer to matrix of size (neurons_ x neurons_) ";

%feature("docstring")  ESN::getDelays "throw (
AUExcept) query the trained delays in delay&sum readout See:  class
SimFilterDS and copies the data into a C-style matrix

Memory of the C array must be allocated before!

Parameters:
-----------

wmtx:  matrix with delay form neurons+inputs to all outputs size =
(output x neurons+inputs) ";

/*  SET methods  */

%feature("docstring")  ESN::setInitAlgorithm "throw (
AUExcept) set initialization algorithm ";

%feature("docstring")  ESN::setTrainAlgorithm "throw (
AUExcept) set training algorithm ";

%feature("docstring")  ESN::setSimAlgorithm "throw (
AUExcept) set simulation algorithm ";

%feature("docstring")  ESN::setSize "throw ( AUExcept)
set reservoir size (nr of neurons) ";

%feature("docstring")  ESN::setInputs "throw (
AUExcept) set nr of inputs to the reservoir ";

%feature("docstring")  ESN::setOutputs "throw (
AUExcept) set nr of outputs from the reservoir ";

%feature("docstring")  ESN::setNoise "throw ( AUExcept)
set noise level for training/simulation algorithm

Parameters:
-----------

noise:  with uniform distribution within [-noise|+noise] ";

%feature("docstring")  ESN::setInitParam "

set initialization parameter ";

%feature("docstring")  ESN::setReservoirAct "throw (
AUExcept) set reservoir activation function ";

%feature("docstring")  ESN::setOutputAct "throw (
AUExcept) set output activation function ";

/*  SET internal data  */

/* Additional method to set all parameters with string key-value
pairs, which can be used for bindings from other languages

Parameters:
-----------

param:  the parameter to set

value:  the value of that parameter

*/

%feature("docstring")  ESN::setWin "throw ( AUExcept)
set input weight matrix (neurons x inputs) ";

%feature("docstring")  ESN::setW "throw ( AUExcept) set
reservoir weight matrix (neurons x neurons) ";

%feature("docstring")  ESN::setWback "throw ( AUExcept)
set feedback weight matrix (neurons x outputs) ";

%feature("docstring")  ESN::setWout "throw ( AUExcept)
set output weight matrix (outputs x neurons+inputs) ";

%feature("docstring")  ESN::setX "throw ( AUExcept) set
internal state vector (size = neurons) ";

%feature("docstring")  ESN::setLastOutput "throw (
AUExcept) set last output, stored by the simulation algorithm needed
in singleStep simulation with feedback

Parameters:
-----------

last:  vector with length = outputs ";

/*  SET internal data C-style interface  */

%feature("docstring")  ESN::setWin "throw ( AUExcept)
set input weight matrix C-style interface (neurons x inputs) (data
will be copied into a FLENS matrix)

Parameters:
-----------

inmtx:  pointer to win matrix in row major storage ";

%feature("docstring")  ESN::setW "throw ( AUExcept) set
reservoir weight matrix C-style interface (neurons x neurons) (data
will be copied into a FLENS matrix)

Parameters:
-----------

inmtx:  pointer to a dense reservoir matrix in row major storage ";

%feature("docstring")  ESN::setWback "throw ( AUExcept)
set feedback weight matrix C-style interface (neurons x outputs) (data
will be copied into a FLENS matrix)

Parameters:
-----------

inmtx:  pointer to wback matrix in row major storage ";

%feature("docstring")  ESN::setWout "throw ( AUExcept)
set output weight matrix C-style interface (outputs x neurons+inputs)
(data will be copied into a FLENS matrix)

Parameters:
-----------

inmtx:  pointer to wout matrix in row major storage ";

%feature("docstring")  ESN::setX "throw ( AUExcept) set
internal state vector C-style interface (size = neurons) (data will be
copied into a FLENS matrix)

Parameters:
-----------

invec:  pointer to state vector ";

%feature("docstring")  ESN::setLastOutput "throw (
AUExcept) set last output, stored by the simulation algorithm needed
in singleStep simulation with feedback

Parameters:
-----------

last:  vector with size = outputs ";

%feature("docstring")  ESN::ESN "

Constructor. ";

%feature("docstring")  ESN::ESN "

Copy Constructor. ";

%feature("docstring")  ESN::~ESN "

Destructor. ";

