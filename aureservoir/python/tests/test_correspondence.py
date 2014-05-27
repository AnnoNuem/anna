import sys
from numpy.testing import *
import numpy as N
import random, scipy.signal

# TODO: right module and path handling
sys.path.append("../")
from aureservoir import *


class test_correspondence(NumpyTestCase):

    def setUp(self):
	
	# parameters
	self.size = random.randint(10,15)
	self.ins = random.randint(1,5)
	self.outs = random.randint(1,5)
	self.conn = random.uniform(0.9,0.99)
	self.train_size = 25
	self.sim_size = 10
	self.dtype = 'float64'
	
	# construct network
	if self.dtype is 'float32':
		self.net = SingleESN()
	else:
		self.net = DoubleESN()
	
	# set parameters
	self.net.setReservoirAct(ACT_TANH)
	self.net.setOutputAct(ACT_TANH)
	self.net.setSize( self.size )
	self.net.setInputs( self.ins )
	self.net.setOutputs( self.outs )
	self.net.setInitParam(CONNECTIVITY, self.conn)
	self.net.setInitParam(FB_CONNECTIVITY, 0.5)
	self.net.setSimAlgorithm(SIM_STD)
	self.net.setTrainAlgorithm(TRAIN_PI)


    def testCopyConstructor(self, level=1):
	""" test if a copied net generates the same result """
        
	self.net.init()
	
	# set output weight matrix
	trainin = N.random.rand(self.ins,self.train_size) * 2 - 1
	trainout = N.random.rand(self.outs,self.train_size) * 2 - 1
	trainin = N.asfarray(trainin, self.dtype)
	trainout = N.asfarray(trainout, self.dtype)
	self.net.train(trainin,trainout,1)
	
	# copy network
	# ATTENTION: operator= is shallow copy !
	if self.dtype is 'float32':
		netA = SingleESN(self.net)
	else:
		netA = DoubleESN(self.net)
	
	# test matrices
	W = N.empty((self.size,self.size),self.dtype)
	self.net.getW( W )
	WA = N.empty((self.size,self.size),self.dtype)
	netA.getW( WA )
	assert_array_almost_equal(W,WA)
	assert_array_almost_equal(self.net.getWback(),netA.getWback())
	assert_array_almost_equal(self.net.getWout(),netA.getWout())
	assert_array_almost_equal(self.net.getWin(),netA.getWin())
	assert_array_almost_equal(self.net.getX(),netA.getX())
	
	# simulate both networks separate and test result
	indata = N.random.rand(self.ins,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	outdata = N.empty((self.outs,self.sim_size),self.dtype)
	outdataA = N.empty((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	netA.simulate( indata, outdataA )
	assert_array_almost_equal(outdata,outdataA)


    def testCopyConstructorBP(self, level=1):
	""" test if a copied bandpass ESN generates the same result """
        
	# set bandpass parameters
	self.net.setSimAlgorithm(SIM_BP)
	f1 = N.linspace(0.1, 1., self.net.getSize())
	f2 = N.linspace(0.0001, 0.5, self.net.getSize())
	self.net.init()
	self.net.setBPCutoff(f1,f2)
	
	# set output weight matrix
	trainin = N.random.rand(self.ins,self.train_size) * 2 - 1
	trainout = N.random.rand(self.outs,self.train_size) * 2 - 1
	trainin = N.asfarray(trainin, self.dtype)
	trainout = N.asfarray(trainout, self.dtype)
	self.net.train(trainin,trainout,1)
	
	# copy network
	# ATTENTION: operator= is shallow copy !
	if self.dtype is 'float32':
		netA = SingleESN(self.net)
	else:
		netA = DoubleESN(self.net)
	
	# simulate both networks separate and test result
	indata = N.random.rand(self.ins,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	outdata = N.empty((self.outs,self.sim_size),self.dtype)
	outdataA = N.empty((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	netA.simulate( indata, outdataA )
	assert_array_almost_equal(outdata,outdataA)



    def testSetInternalData(self, level=1):
	""" test if manually setting the weigth matrices generates the
	same result """
	
	self.net.init()
	
	# train first ESN
	trainin = N.random.rand(self.ins,self.train_size) * 2 - 1
	trainout = N.random.rand(self.outs,self.train_size) * 2 - 1
	trainin = N.asfarray(trainin, self.dtype)
	trainout = N.asfarray(trainout, self.dtype)
	self.net.train(trainin,trainout,1)
	self.net.resetState()
	
	# create second ESN
	if self.dtype is 'float32':
		netA = SingleESN()
	else:
		netA = DoubleESN()
	netA.setReservoirAct(ACT_TANH)
	netA.setOutputAct(ACT_TANH)
	netA.setSize( self.size )
	netA.setInputs( self.ins )
	netA.setOutputs( self.outs )
	netA.setSimAlgorithm(SIM_STD)
	netA.setTrainAlgorithm(TRAIN_PI)
	
	# set internal data in second ESN
	netA.setWin( self.net.getWin().copy() )
	netA.setWout( self.net.getWout().copy() )
	netA.setWback( self.net.getWback().copy() )
	W = N.empty((self.size,self.size),self.dtype)
	self.net.getW( W )
	netA.setW( W )
	netA.setX( self.net.getX().copy() )
		
	# simulate both networks separate and test result
	indata = N.random.rand(self.ins,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	outdata = N.empty((self.outs,self.sim_size),self.dtype)
	outdataA = N.empty((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	netA.simulate( indata, outdataA )
	assert_array_almost_equal(outdata,outdataA)


    def testIIRFilters(self, level=1):
	""" test correspondence of SIM_FILTER and pythons lfilter """
        
	# set parameters for non-interacting neurons:
	# new state only depends on the input
	self.net.setSimAlgorithm(SIM_FILTER)
	self.ins = 1
	self.outs = 1
	self.net.setReservoirAct(ACT_LINEAR)
	self.net.setOutputAct(ACT_LINEAR)
	self.net.setInputs( self.ins )
	self.net.setOutputs( self.outs )
	self.net.setInitParam(ALPHA, 0.)
	self.net.setInitParam(FB_CONNECTIVITY, 0.)
	self.net.setInitParam(IN_CONNECTIVITY, 1.)
	self.net.setInitParam(IN_SCALE, 0.)
	self.net.setInitParam(IN_SHIFT, 1.)
	self.net.init()
	
	# set paramas for a biquad bandpass filter:
	B = N.empty((self.size,3))
	A = N.empty((self.size,3))
	B[:,0] = 0.29573818
	B[:,1] = 0.
	B[:,2] = -0.29573818
	A[:,0] = 1.29573818
	A[:,1] = -1.84775907
	A[:,2] = 0.70426182
	self.net.setIIRCoeff(B,A)
	
	# simulate network step by step:
	# the states x are now only the filtered input signal, which
	# is the same for each neuron ! (because ALPHA = 0)
	indata = N.random.rand(1,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	filterout = N.zeros((1,self.sim_size),self.dtype)
	outtmp = N.zeros((self.outs),self.dtype)
	for n in range(self.sim_size):
		intmp = indata[:,n].copy()
		self.net.simulateStep( intmp, outtmp )
		filterout[0,n] = self.net.getX()[0]
	
	# now calculate the same with scipy.signal.lfilter
	filterout2 = scipy.signal.lfilter(B[0], A[0], indata.flatten())
	
	assert_array_almost_equal(filterout.flatten(),filterout2)


    def testSerialIIRFilters(self, level=1):
	""" test correspondence of a cascaded IIR filter """
        
	# set parameters for non-interacting neurons:
	# new state only depends on the input
	self.net.setSimAlgorithm(SIM_FILTER)
	self.ins = 1
	self.outs = 1
	self.net.setReservoirAct(ACT_LINEAR)
	self.net.setOutputAct(ACT_LINEAR)
	self.net.setInputs( self.ins )
	self.net.setOutputs( self.outs )
	self.net.setInitParam(ALPHA, 0.)
	self.net.setInitParam(FB_CONNECTIVITY, 0.)
	self.net.setInitParam(IN_CONNECTIVITY, 1.)
	self.net.setInitParam(IN_SCALE, 0.)
	self.net.setInitParam(IN_SHIFT, 1.)
	self.net.init()
	
	# set paramas for a gammatone filter
	# (cascade of 4 biquads)
	Btmp,Atmp = self._gammatone_biquad(44100,self.size,150)
	B = N.empty((self.size,12))
	A = N.empty((self.size,12))
	serial = 4
	B[:,0:3] = Btmp[:,0,:]
	B[:,3:6] = Btmp[:,1,:]
	B[:,6:9] = Btmp[:,2,:]
	B[:,9:12] = Btmp[:,3,:]
	A[:,0:3] = Atmp[:,0,:]
	A[:,3:6] = Atmp[:,1,:]
	A[:,6:9] = Atmp[:,2,:]
	A[:,9:12] = Atmp[:,3,:]
	self.net.setIIRCoeff(B,A,serial)
	
	# simulate network step by step:
	# the states x are now only the filtered input signal, which
	# is the same for each neuron ! (because ALPHA = 0)
	indata = N.random.rand(1,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	filterout = N.zeros((1,self.sim_size),self.dtype)
	outtmp = N.zeros((self.outs),self.dtype)
	for n in range(self.sim_size):
		intmp = indata[:,n].copy()
		self.net.simulateStep( intmp, outtmp )
		filterout[0,n] = self.net.getX()[0]
	
	# now calculate the same with scipy.signal.lfilter
	filterout2 = self._gammatonefilter( indata.flatten() )
	
	assert_array_almost_equal(filterout.flatten(),filterout2)


    def testIIRvsSTDESN(self, level=1):
	""" test if IIR-ESN with b=1 and a=1 gives same result as
	standard ESN """
	
	self.net.init()
	
	# train first ESN
	trainin = N.random.rand(self.ins,self.train_size) * 2 - 1
	trainout = N.random.rand(self.outs,self.train_size) * 2 - 1
	trainin = N.asfarray(trainin, self.dtype)
	trainout = N.asfarray(trainout, self.dtype)
	self.net.train(trainin,trainout,1)
	self.net.resetState()
	
	# create second ESN
	if self.dtype is 'float32':
		netA = SingleESN()
	else:
		netA = DoubleESN()
	netA.setReservoirAct(ACT_TANH)
	netA.setOutputAct(ACT_TANH)
	netA.setSize( self.size )
	netA.setInputs( self.ins )
	netA.setOutputs( self.outs )
	netA.setSimAlgorithm(SIM_FILTER)
	netA.setTrainAlgorithm(TRAIN_PI)
	B = N.zeros((self.size,2))
	A = N.zeros((self.size,2))
	B[:,0] = 1.
	A[:,0] = 1.
	netA.init()
	netA.setIIRCoeff(B,A)
	
	# set internal data in second ESN
	netA.setWin( self.net.getWin().copy() )
	netA.setWout( self.net.getWout().copy() )
	netA.setWback( self.net.getWback().copy() )
	W = N.empty((self.size,self.size),self.dtype)
	self.net.getW( W )
	netA.setW( W )
	netA.setX( self.net.getX().copy() )
	
	# simulate both networks separate and test result
	indata = N.random.rand(self.ins,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	outdata = N.empty((self.outs,self.sim_size),self.dtype)
	outdataA = N.empty((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	netA.simulate( indata, outdataA )
	assert_array_almost_equal(outdata,outdataA)
	
	
    def _gammatone_biquad(self,fs,numChannels,lowFreq):
	""" Computes the filter coefficients for a bank of 
	Gammatone filters.  These filters were defined by Patterson and 
	Holdworth for simulating the cochlea.
	
	This implements the same as makeGammatoneFilter, but has to be used
	with a cascade of 4 biquad and therefore avoids the numerical stability
	problem with high sampling rates and low frequencies.
	
	return     B, A    (3d array of coeffs for each channel for each biquad)
	
	B: the forward part of the filter,
	   z.B. B(channel, biquad_nr, biquad_coeff)
	   index1 = channel of the gammatone filter
	   index2 = biquad nr (0-3)
	   index3 = biquad coefficient (0-2)
	A: the recursive part of the filter (same structure as B)
	
	2007,
	Georg Holzmann
	"""
	
	T = 1./fs
	
	# Change the followFreqing three parameters if you wish to use a
	# different ERB scale.
	EarQ = 9.26449            # Glasberg and Moore Parameters
	minBW = 24.7
	order = 1
	
	# All of the following expressions are derived in Apple TR #35, "An
	# Efficient Implementation of the Patterson-Holdsworth Cochlear
	# Filter Bank."
	cf = N.arange(numChannels) + 1
	cf = -(EarQ*minBW) + N.exp( cf * (-N.log(fs/2. + EarQ*minBW) + \
	     N.log(lowFreq + EarQ*minBW) ) / numChannels ) \
	     *(fs/2. + EarQ*minBW)
	
	ERB = ((cf/EarQ)**order + minBW**order)**(1/order)
	B = 1.019*2*N.pi*ERB
	
	# calculate gain factor
	gain = abs( (-2*N.exp(4*1j*cf*N.pi*T)*T + \
	       2*N.exp(-(B*T) + 2*1j*cf*N.pi*T) * T * \
	       (N.cos(2*cf*N.pi*T) - N.sqrt(3. - 2**(3./2.)) * \
	       N.sin(2*cf*N.pi*T))) * (-2*N.exp(4*1j*cf*N.pi*T)*T + \
	       2*N.exp(-(B*T) + 2*1j*cf*N.pi*T) * T * \
	       (N.cos(2*cf*N.pi*T) + N.sqrt(3. - 2**(3./2.)) * \
	       N.sin(2*cf*N.pi*T))) * (-2*N.exp(4*1j*cf*N.pi*T)*T + \
	       2*N.exp(-(B*T) + 2*1j*cf*N.pi*T) * T * (N.cos(2*cf*N.pi*T) - \
	       N.sqrt(3. + 2**(3./2.)) * N.sin(2*cf*N.pi*T))) * \
	       (-2*N.exp(4*1j*cf*N.pi*T)*T + 2*N.exp(-(B*T) + \
	       2*1j*cf*N.pi*T) * T * (N.cos(2*cf*N.pi*T) + \
	       N.sqrt(3. + 2**(3./2.)) * N.sin(2*cf*N.pi*T))) / \
	       (-2. / N.exp(2*B*T) - 2*N.exp(4*1j*cf*N.pi*T) + \
	       2.*(1. + N.exp(4*1j*cf*N.pi*T)) / N.exp(B*T)) ** 4)
	
	# implementation with 4 biquads:
	# z.B. Afilt(channel, biquad_nr, biquad_coeff)
	Afilt = N.zeros((len(cf),4,3))  # feedback path
	Bfilt = N.zeros((len(cf),4,3))  # forward path
	
	# init all 4 biquads
	for n in range(4):
		Afilt[:,n,0] = 1.
		Afilt[:,n,1] = -2 * N.cos(2*cf*N.pi*T) / N.exp(B*T)
		Afilt[:,n,2] = N.exp(-2*B*T)
		Bfilt[:,n,0] = T
		Bfilt[:,n,2] = 0.
	
	# init the rest
	tmp = 2*T*N.cos(2*cf*N.pi*T) / N.exp(B*T)
	Bfilt[:,0,1] = -(tmp+2*N.sqrt(3.+2**1.5)*T*N.sin(2*cf*N.pi*T)/N.exp(B*T))/2.
	Bfilt[:,1,1] = -(tmp-2*N.sqrt(3.+2**1.5)*T*N.sin(2*cf*N.pi*T)/N.exp(B*T))/2.
	Bfilt[:,2,1] = -(tmp+2*N.sqrt(3.-2**1.5)*T*N.sin(2*cf*N.pi*T)/N.exp(B*T))/2.
	Bfilt[:,3,1] = -(tmp-2*N.sqrt(3.-2**1.5)*T*N.sin(2*cf*N.pi*T)/N.exp(B*T))/2.
	
	# normalize first biquad to gain
	for n in range(3):
		Bfilt[:,0,n] = Bfilt[:,0,n] / gain
	
	return Bfilt, Afilt
	
	
    def _gammatonefilter(self, signal, chan=0):
	""" a gammatone filter bank """
	B,A = self._gammatone_biquad(44100,self.size,150)
	
	# run the 4 biquads
	y = scipy.signal.lfilter(B[chan,0,:],A[chan,0,:],signal)
	y = scipy.signal.lfilter(B[chan,1,:],A[chan,1,:],y)
	y = scipy.signal.lfilter(B[chan,2,:],A[chan,2,:],y)
	y = scipy.signal.lfilter(B[chan,3,:],A[chan,3,:],y)
	return y


if __name__ == "__main__":
    NumpyTest().run()
