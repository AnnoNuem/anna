
:::_aureservoir_:::
  python examples


The main python examples are located in this directory.
As the aureservoir python syntax is the same as in C++
one can use these examples analogous in C++.


Following files are included ATM:

- slow_sine.py:
  slow sine genration task with standard and bandpass ESN

- narma10.py:
  a 10th order NARMA system identification task
  with additional squared state updates

- singleneuron_sinosc.py:
  with a delay&sum readout it's possible to train
  a sine osc with just one neuron

- multiple_sines.py:
  trains an ESN to be a generator for a sum of 10 sines
  or a product of 5 sines

- sparse_nonlin_system_identification.py:
  sparse nonlinear system identification with long-term
  dependencies of 2 example systems


Additionally some utility files are included:
- errorcalc.py: mean square error calculation utilities
- filtering.py: tools to calculate filter parameters
- filteresn.py: helper class for Filter and Bandpass ESN


2008,
Georg Holzmann
