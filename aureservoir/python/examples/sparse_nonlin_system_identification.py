###########################################################
# sparse nonlinear system identification with long-term
# dependencies of 2 example systems
#
# see " Echo State Networks with Filter Neurons and a
# Delay&Sum Readout"
# http://grh.mur.at/misc/ESNsWithFilterNeuronsAndDSReadout.pdf
#
# 2008, Georg Holzmann
###########################################################

from numpy import *
import pylab as P
import sys, errorcalc
sys.path.append("../")
from aureservoir import *


###########################################################
# FUNCTIONS

def setup_SQUARE_ESN():
	""" ESN with squared state updates
	"""
	net = DoubleESN()
	net.setSize(100)
	net.setInputs(2)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.05 )
	net.setInitParam( ALPHA, 0.8 )
	net.setInitParam( IN_CONNECTIVITY, 1. )
	net.setInitParam( IN_SCALE, 0.1 )
	net.setInitParam( FB_CONNECTIVITY, 0. )
	net.setInitParam( FB_SCALE, 0. )
	net.setReservoirAct( ACT_TANH )
	net.setOutputAct( ACT_LINEAR )
	net.setSimAlgorithm( SIM_SQUARE )
	net.setTrainAlgorithm( TRAIN_PI )
	net.trainnoise = 0.0001
	net.testnoise = 0.
	net.ds = 0
	net.init()
	return net

def setup_DS_ESN():
	""" ESN with squared state updates and a delay&sum readout
	"""
	net = DoubleESN()
	net.setSize(100)
	net.setInputs(2)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.05 )
	net.setInitParam( ALPHA, 0.8 )
	net.setInitParam( IN_CONNECTIVITY, 1. )
	net.setInitParam( IN_SCALE, 0.1 )
	net.setInitParam( FB_CONNECTIVITY, 0. )
	net.setInitParam( FB_SCALE, 0. )
	net.setReservoirAct( ACT_TANH )
	net.setOutputAct( ACT_LINEAR )
	net.setSimAlgorithm( SIM_SQUARE )
	net.setTrainAlgorithm( TRAIN_DS_PI )
	net.setInitParam(DS_USE_GCC) # use GCC here !
	net.setInitParam(DS_MAXDELAY, 3000)
	net.trainnoise = 0.0001
	net.testnoise = 0.
	net.ds = 1
	net.init()
	return net

def narma10sparse(x,d=10):
	""" same tenth-order NARMA system with sparse
	x and y, d is the stepsize
	"""
	size = len(x)
	y = zeros(x.shape)
	for n in range(10*d,size):
		y[n] = 0.3*y[n-1*d] + 0.05*y[n-1*d]*(y[n-1*d]+y[n-2*d]+y[n-3*d] \
		       +y[n-4*d]+y[n-5*d]+y[n-6*d]+y[n-7*d]+y[n-8*d]+y[n-9*d] \
		       +y[n-10*d]) + 1.5*x[n-10*d]*x[n-1*d] + 0.1
	return y

def sparseSystem2(x,step=10):
	""" system suggested from stefan
	"""
	size = len(x)
	y = zeros(x.shape)
	for n in range(2*step+2,size):
		y[n] = (x[n]+x[n-1]+x[n-2]+x[n-3]) * \
		       (x[n-1*step]+x[n-1*step-1]+x[n-1*step-2]) * \
		       (x[n-2*step]+x[n-2*step-1]+x[n-2*step-2])
	return y

def get_esn_data(x,y,trainsize,testsize):
	""" returns trainin, trainout, testin, testout
	"""
	skip = 500 # NARMA initialization
	trainin = x[skip:skip+trainsize]
	trainin.shape = 1,-1
	trainout = y[skip:skip+trainsize]
	trainout.shape = 1,-1
	testin = x[skip+trainsize:skip+trainsize+testsize]
	testin.shape = 1,-1
	testout = y[skip+trainsize:skip+trainsize+testsize]
	testout.shape = 1,-1
	# for 2. input
	trainin1 = ones((2,trainin.shape[1]))
	testin1 = ones((2,testin.shape[1]))
	trainin1[0] = trainin
	testin1[0] = testin
	return trainin1, trainout, testin1, testout

def plot(esnout,testout):
	""" plotting """
	from matplotlib import font_manager
	P.subplot(121)
	P.title('Sparse Nonlinear System Identification')
	P.plot(testout,'b')
	P.plot(esnout,'r')
	P.subplot(122)
	P.title('zoomed to first 100 samples')
	P.plot(testout[:100],'b')
	P.plot(esnout[:100],'r')
	P.legend( ('target', 'ESN output'), loc="upper right", \
            prop=font_manager.FontProperties(size='smaller') )
	P.show()


###########################################################
# MAIN

trainsize = 3200
washout = 400
testsize = 2200

# choose ESN: compare standard with D&S ESN
#net = setup_SQUARE_ESN()
net = setup_DS_ESN()

# generate train/test signals
size = trainsize+testsize+500
x = random.rand(size)*0.5

# choose system: for very large stepsize increase trainsize !!
#y = narma10sparse(x,10)
y = sparseSystem2(x,step=50)

# create in/out data
trainin, trainout, testin, testout = get_esn_data(x,y,trainsize,testsize)

# ESN training
net.setNoise(net.trainnoise)
net.train(trainin,trainout,washout)
if (net.ds == 1):
	delays = zeros((1,102))
	net.getDelays(delays)
	print "trained delays:"
	print delays
print "output weights:"
print "\tmean: ", net.getWout().mean(), "\tmax: ", abs(net.getWout()).max()

# ESN simulation
esnout = empty(testout.shape)
net.setNoise(net.testnoise)
net.simulate(testin,esnout)
nrmse = errorcalc.nrmse( esnout, testout, washout )
print "\nNRMSE: ", nrmse
print "\nNMSE: ", errorcalc.nmse( esnout, testout, washout )

# final plotting
plot(esnout,testout)
