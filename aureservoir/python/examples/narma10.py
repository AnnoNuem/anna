###########################################################
# a 10th order NARMA system identification task
# with additional squared state updates
#
# see Jaeger H. (2003), "Adaptive nonlinear system
# identification with echo state networks."
#
# 2007, Georg Holzmann
###########################################################

from numpy import *
import pylab as P
import sys
sys.path.append("../")
from aureservoir import *
import errorcalc


###########################################################
# FUNCTIONS

def setup_STD_ESN():
	""" setup ESN like in Jaegers paper,
	without squared state updates
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
	net.setSimAlgorithm( SIM_STD )
	net.setTrainAlgorithm( TRAIN_PI )
	trainnoise = 0.0001
	testnoise = 0.
	net.init()
	return net, trainnoise, testnoise

def setup_SQUARE_ESN():
	""" setup ESN like in Jaegers paper with squared state updates
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
	trainnoise = 0.0001
	testnoise = 0.
	net.init()
	return net, trainnoise, testnoise

def narma10(x):
	""" tenth-order NARMA system applied to the input signal
	"""
	size = len(x)
	y = zeros(x.shape)
	for n in range(10,size):
		y[n] = 0.3*y[n-1] + 0.05*y[n-1]*(y[n-1]+y[n-2]+y[n-3] \
		       +y[n-4]+y[n-5]+y[n-6]+y[n-7]+y[n-8]+y[n-9]+y[n-10]) \
		       + 1.5*x[n-10]*x[n-1] + 0.1
	return y

def get_esn_data(x,y,trainsize,testsize,inscale=1.,inshift=0.):
	""" returns trainin, trainout, testin, testout
	"""
	skip = 50 # NARMA initialization
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
	P.title('NARMA System Identification')
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
washout = 200
testsize = 2200

# choose ESN: compare performance between square and STD-ESN
#net, trainnoise, testnoise = setup_STD_ESN()
net, trainnoise, testnoise = setup_SQUARE_ESN()

# generate train/test signals
size = trainsize+testsize
x = random.rand(size)*0.5
y = narma10(x)

# create in/outs with bias input
trainin, trainout, testin, testout = get_esn_data(x,y,trainsize,testsize)

# ESN training
net.setNoise(trainnoise)
print "training ..."
net.train(trainin,trainout,washout)
print "output weights:"
print "\tmean: ", net.getWout().mean(), "\tmax: ", abs(net.getWout()).max()

# ESN simulation
esnout = empty(testout.shape)
net.setNoise(testnoise)
net.simulate(testin,esnout)
nrmse = errorcalc.nrmse( esnout, testout, washout )
print "\nNRMSE: ", nrmse
print "\nNMSE: ", errorcalc.nmse( esnout, testout, washout )
plot(esnout,testout)
