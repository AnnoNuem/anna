###########################################################
# slow sine genration task with standard and BP ESN
#
# see Jaeger H. (2002), "Tutorial on training recurrent
# neural networks, covering BPPT, RTRL, EKF and the
# echo state network approach"
#
# 2007, Georg Holzmann
###########################################################

import numpy as N
import pylab as P
from aureservoir import *
import sys
sys.path.append("../")
from aureservoir import *
import errorcalc, filteresn


###########################################################
# FUNCTIONS

def setup_STD_ESN():
	""" configuration of a standard ESN """
	net = DoubleESN()
	net.setSize(20)
	net.setInputs(1)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.2 )
	net.setInitParam( ALPHA, 0.45 )
	net.setInitParam( IN_CONNECTIVITY, 0. )
	net.setInitParam( IN_SCALE, 0. )
	net.setInitParam( FB_CONNECTIVITY, 1. )
	net.setInitParam( FB_SCALE, 1. )
	net.setReservoirAct( ACT_TANH )
	net.setOutputAct( ACT_LINEAR )
	net.setSimAlgorithm( SIM_STD )
	net.setTrainAlgorithm( TRAIN_PI )
	trainnoise = 1e-6
	testnoise = 0.
	return net, trainnoise, testnoise

def setup_ESN_BP():
	""" configuration of a bandpass ESN with cutoff frequency
	at the sine frequency """
	net = filteresn.BPESN()
	net.setSize(20)
	net.setInputs(1)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.2 )
	net.setInitParam( ALPHA, 0.3 )
	net.setInitParam( IN_CONNECTIVITY, 0. )
	net.setInitParam( IN_SCALE, 0. )
	net.setInitParam( FB_CONNECTIVITY, 1. )
	net.setInitParam( FB_SCALE, 1. )
	net.setReservoirAct( ACT_TANH )
	net.setOutputAct( ACT_LINEAR )
	net.setSimAlgorithm( SIM_BP )
	net.setTrainAlgorithm( TRAIN_PI )
	net.setConstCutoffs(0.01, 0.01) # sine frequency
	trainnoise = 0.
	testnoise = 0.
	return net, trainnoise, testnoise

def generate_slow_sine(size,ampl=1):
	""" generates a slow sinewave:
	    y[n] = ampl * sin(n/100)
	    omega = 2*pi*f -> f = 0.0015915494309189536
	    -> Periode = 628.318 """
	x = N.arange( float(size) )
	y = ampl*N.sin( x/100 )
	return y

def get_esn_data(signal,trainsize,testsize):
	""" returns trainin, trainout, testin, testout """
	
	trainout = signal[0:trainsize]
	trainout.shape = 1,-1
	trainin = N.zeros(trainout.shape)
	
	testout = signal[trainsize:trainsize+testsize]
	testout.shape = 1,-1
	testin = N.zeros(testout.shape)
	
	return trainin, trainout, testin, testout

def plot(esnout,testout):
	""" plotting """
	P.title('Original=blue, ESNout=red')
	P.plot(testout,'b',esnout,'r')
	P.show()


###########################################################
# MAIN

trainsize = 4000
washout = 2000
testsize = 8000

# choose ESN: compare STD-ESN with BP-ESN
net, trainnoise, testnoise = setup_STD_ESN()
#net, trainnoise, testnoise = setup_ESN_BP()
net.init()

# generate signals
slsine = generate_slow_sine(trainsize+testsize)
trainin, trainout, testin, testout = get_esn_data(slsine,trainsize,testsize)

# ESN training
net.setNoise(trainnoise)
net.train(trainin,trainout,washout)
print "output weights:"
print "\tmean: ", net.getWout().mean(), "\tmax: ", abs(net.getWout()).max()

# ESN simulation
esnout = N.empty(testout.shape)
net.setNoise(testnoise)
net.simulate(testin,esnout)
print "\nNRMSE: ", errorcalc.nrmse( esnout, testout, 50 )
plot(esnout,testout)
