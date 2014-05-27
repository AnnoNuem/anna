###########################################################
# with a delay&sum readout it's possible to train
# a sine osc with just one neuron
#
# see " Echo State Networks with Filter Neurons and a
# Delay&Sum Readout"
# http://grh.mur.at/misc/ESNsWithFilterNeuronsAndDSReadout.pdf
#
# 2008, Georg Holzmann
###########################################################

import numpy as N
import pylab as P
import sys, errorcalc
sys.path.append("../")
from aureservoir import *

###########################################################
# MAIN

# setup net
net = DoubleESN()
net.setSize(1)
net.setInputs(1)
net.setOutputs(1)
net.setInitParam(CONNECTIVITY, 1.)
net.setInitParam(ALPHA, 0.)
net.setInitParam(IN_CONNECTIVITY, 0.)
net.setInitParam(IN_SCALE, 0.)
net.setInitParam(FB_CONNECTIVITY, 1.)
net.setInitParam(FB_SCALE, 1.)
net.setInitParam(FB_SHIFT, 0.)
net.setReservoirAct(ACT_LINEAR)
net.setOutputAct(ACT_LINEAR)
net.setInitAlgorithm(INIT_STD)
net.setSimAlgorithm(SIM_FILTER_DS)
net.setTrainAlgorithm(TRAIN_DS_PI)

# you need crosscorrelation method here, GCC won't work:
net.setInitParam(DS_USE_CROSSCORR)
net.setInitParam(DS_MAXDELAY, 1000)
net.init()

trainsize = 1000
washout = 500
testsize = 300

# make sine
n = N.arange(float(trainsize+testsize))
signal = N.sin(2*N.pi*n* 0.05)

# generate train/test data
trainout = signal[0:trainsize]
trainout.shape = 1, -1
trainin = N.zeros(trainout.shape)
testout = signal[trainsize:trainsize+testsize]
testout.shape = 1, -1
testin = N.zeros(testout.shape)

# ESN training
net.train(trainin, trainout, washout)
delays = N.zeros((1,2))
net.getDelays(delays)
print "output weights:", net.getWout()
print "calculated delays:", delays

# ESN simulation
esnout = N.empty(testout.shape)
net.simulate(testin, esnout)
print "\nNRMSE: ", errorcalc.nrmse(esnout, testout, 50)

# some plotting
P.plot(testout.flatten())
P.plot(esnout.flatten(), 'r')
P.show()
