###########################################################
# train ESN to be a generator for a sum of 10 sines or
# a product of 5 sines
#
# see " Echo State Networks with Filter Neurons and a
# Delay&Sum Readout"
# http://grh.mur.at/misc/ESNsWithFilterNeuronsAndDSReadout.pdf
#
# 2008, Georg Holzmann
###########################################################

import numpy as N
import pylab as P
import sys, errorcalc, filteresn
sys.path.append("../")
from aureservoir import *


###########################################################
# FUNCTIONS

def setup_FILTER_ESN():
	""" ESN with IIR filter neurons
	"""
	net = filteresn.IIRESN()
	net.setSize(100)
	net.setInputs(1)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.2 )
	net.setInitParam( ALPHA, 1. )
	net.setInitParam( IN_CONNECTIVITY, 0. )
	net.setInitParam( IN_SCALE, 0. )
	net.setInitParam( FB_CONNECTIVITY, 1. )
	net.setInitParam( FB_SCALE, 1. )
	net.setReservoirAct( ACT_LINEAR )
	net.setOutputAct( ACT_LINEAR )
	net.setInitAlgorithm( INIT_STD )
	net.setSimAlgorithm( SIM_FILTER )
	net.setTrainAlgorithm( TRAIN_PI )
	net.ds = 0
	return net

def setup_DS_ESN():
	""" ESN with IIR filter neurons and a delay&sum readout
	"""
	net = filteresn.IIRESN()
	net.setSize(100)
	net.setInputs(1)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.2 )
	net.setInitParam( ALPHA, 1. )
	net.setInitParam( IN_CONNECTIVITY, 0. )
	net.setInitParam( IN_SCALE, 0. )
	net.setInitParam( FB_CONNECTIVITY, 1. )
	net.setInitParam( FB_SCALE, 1. )
	net.setReservoirAct( ACT_LINEAR )
	net.setOutputAct( ACT_LINEAR )
	net.setInitAlgorithm( INIT_STD )
	net.setSimAlgorithm( SIM_FILTER_DS )
	net.setTrainAlgorithm( TRAIN_DS_PI )
	net.setInitParam(DS_USE_CROSSCORR) # use crosscorr here !
	net.setInitParam(DS_MAXDELAY, 3000)
	net.ds = 1
	return net

def generate_sig(size,freq,amp=1.,operation="add"):
	""" genereates as many sines as in the freq array and
	adds (add) or modulates (mod) them
	"""
	x = N.arange( float(size) )
	y = N.empty((len(freq),size))
	for n in range(len(freq)):
		y[n] = amp*N.sin( 2*N.pi*x* freq[n] )
	
	if operation == "add":
		sig = N.zeros(size)
		for n in range(len(freq)):
			sig += y[n]
	elif operation == "mod":
		sig = N.ones(size)
		for n in range(len(freq)):
			sig *= y[n]
	else:
		raise ValueError, "wrong operation"
	return sig

def get_esn_data(signal,trainsize,testsize):
	""" returns trainin, trainout, testin, testout
	"""
	trainout = signal[0:trainsize]
	trainout.shape = 1,-1
	trainin = N.zeros(trainout.shape)
	testout = signal[trainsize:trainsize+testsize]
	testout.shape = 1,-1
	testin = N.zeros(testout.shape)
	return trainin, trainout, testin, testout

def plot(esnout,testout):
	""" plotting """
	from matplotlib import font_manager
	P.subplot(121)
	P.title('sum/product of sines generator')
	P.plot(testout,'b')
	P.plot(esnout,'r')
	P.subplot(122)
	P.title('zoomed to 500 samples')
	P.plot(testout[:500],'b')
	P.plot(esnout[:500],'r')
	P.legend( ('target', 'ESN output'), loc="upper left", \
	            prop=font_manager.FontProperties(size='smaller') )
	P.show()


###########################################################
# MAIN

trainsize = 7500
washout = 2000
testsize = 4000

trainnoise = 0.
testnoise = 0.

# choose ESN: compear with and wihtout delay&sum readout
#net = setup_FILTER_ESN()
net = setup_DS_ESN()

# choose test signal: use product or sum of sines
nr = trainsize+testsize
freq5 = [0.00098, 0.005073, 0.0105, 0.050985, 0.1063]
freq10 = [0.0014, 0.0021, 0.004, 0.00639, 0.00803, 0.0107, 0.021, 0.0309, \
          0.05, 0.107]
signal = generate_sig(nr,freq5,1.,"mod") # product of 5 sines
#signal = generate_sig(nr,freq10,0.33,"add") # sum of 10 sines
trainin, trainout, testin, testout = get_esn_data(signal,trainsize,testsize)

# ESN neuron filters:
#net.setStdESN()
net.setLogBPCutoffs(0.001, 0.48, 2)
net.init()

# DEBUG; plot filter frequency responses
#net.plot_filters(9)
#exit(0)

# ESN training
net.setNoise(trainnoise)
net.train(trainin,trainout,washout)
if (net.ds == 1):
	delays = N.zeros((1,101))
	net.getDelays(delays)
	print "trained delays:"
	print delays
print "output weights:"
print "\tmean: ", net.getWout().mean(), "\tmax: ", abs(net.getWout()).max()

# ESN simulation
esnout = N.empty(testout.shape)
net.setNoise(testnoise)
net.simulate(testin,esnout)
print "\nNRMSE: ", errorcalc.nrmse( esnout, testout, 50 )

# final plotting
plot(esnout,testout)
