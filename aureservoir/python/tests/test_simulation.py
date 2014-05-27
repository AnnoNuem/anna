import sys
from numpy.testing import *
import numpy as N
import random, scipy.signal

# TODO: right module and path handling
sys.path.append("../")
from aureservoir import *


class test_simulation(NumpyTestCase):

    def setUp(self):
	
	# parameters
	self.size = random.randint(10,15)
	self.ins = random.randint(1,5)
	self.outs = random.randint(1,5)
	self.conn = random.uniform(0.9,0.99)
	self.sim_size = 10
	self.dtype = 'float64'
	
	# construct network
	if self.dtype is 'float32':
		self.net = SingleESN()
	else:
		self.net = DoubleESN()
	
	# set parameters
	self.net.setReservoirAct(ACT_LINEAR)
	self.net.setOutputAct(ACT_LINEAR)
	self.net.setSize( self.size )
	self.net.setInputs( self.ins )
	self.net.setOutputs( self.outs )
	self.net.setInitParam(CONNECTIVITY, self.conn)
	self.net.setInitParam(FB_CONNECTIVITY, 0.5)


    def testStd(self, level=1):
	""" test SIM_STD with linear activation functions """
        
	# setup net
	self.net.setSimAlgorithm(SIM_STD)
	self.net.init()
	
	# set output weight matrix
	wout = N.random.rand(self.outs,self.size+self.ins) * 2 - 1
	wout = N.asfarray(wout, self.dtype)
	self.net.setWout( wout )
	
	# simulate network
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size),self.dtype)*2-1
	outdata = N.zeros((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	
	# get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	Wout = self.net.getWout()
	Wback = self.net.getWback()
	x = N.zeros((self.size))
	outtest = N.zeros((self.outs,self.sim_size),self.dtype)
	
	# recalc algorithm in python
	for n in range(self.sim_size):
		# calc new network activation
		x = N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		if n > 0:
			x += N.dot( Wback, outtest[:,n-1] )
		# output = Wout * [x; in]
		outtest[:,n] = N.dot( Wout, N.r_[x,indata[:,n]] )
	
	assert_array_almost_equal(outdata,outtest,3)


    def testStdStepTANH(self, level=1):
	""" test SIM_STD in single step mode with tanh activations """
        
	# setup net
	self.net.setReservoirAct(ACT_TANH)
	self.net.setOutputAct(ACT_TANH)
	self.net.setSimAlgorithm(SIM_STD)
	self.net.init()
	
	# set output weight matrix
	wout = N.random.rand(self.outs,self.size+self.ins) * 2 - 1
	wout = N.asfarray(wout, self.dtype)
	self.net.setWout( wout )
	
	# simulate network step by step
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size), \
	                    self.dtype) * 2 - 1
	outdata = N.zeros((self.outs,self.sim_size),self.dtype)
	outtmp = N.zeros((self.outs),self.dtype)
	for n in range(self.sim_size):
		intmp = indata[:,n].copy()
		self.net.simulateStep( intmp, outtmp )
		outdata[:,n] = outtmp.copy()
	
	# get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	Wout = self.net.getWout()
	Wback = self.net.getWback()
	x = N.zeros((self.size))
	outtest = N.zeros((self.outs,self.sim_size),self.dtype)
	
	# recalc algorithm in python
	for n in range(self.sim_size):
		# calc new network activation
		x = N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		if n > 0:
			x += N.dot( Wback, outtest[:,n-1] )
		# reservoir activation function
		x = N.tanh( x )
		# output = Wout * [x; in]
		outtest[:,n] = N.tanh(N.dot( Wout, N.r_[x,indata[:,n]] ))
	
	assert_array_almost_equal(outdata,outtest,3)


    def testSquare(self, level=1):
	""" test SIM_SQUARE """
        
	# setup net
	self.net.setSimAlgorithm(SIM_SQUARE)
	self.net.init()
	
	# set output weight matrix
	wout = N.random.rand(self.outs,self.size+self.ins) * 2 - 1
	wout = N.asfarray(wout, self.dtype)
	self.net.setWout( wout )
	
	# simulate network
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size),self.dtype)*2-1
	outdata = N.zeros((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	
	## get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	Wout = self.net.getWout()
	Wback = self.net.getWback()
	x = N.zeros((self.size))
	outtest = N.zeros((self.outs,self.sim_size),self.dtype)
	
	# recalc algorithm in python
	for n in range(self.sim_size):
		# calc new network activation
		x = N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		if n > 0:
			x += N.dot( Wback, outtest[:,n-1] )
		# output = Wout * [x; in; x^2; in^2]
		sqstates = N.r_[x, indata[:,n], x**2, indata[:,n]**2]
		outtest[:,n] = N.dot( Wout, sqstates )
	
	assert_array_almost_equal(outdata,outtest,3)


    def testLI(self, level=1):
	""" test simulation with leaky integrating neurons """
        
	# setup net
	self.lr = 0.2
	self.net.setSimAlgorithm(SIM_LI)
	self.net.setInitParam(LEAKING_RATE, self.lr)
	self.net.setInitParam(ALPHA, self.lr)
	self.net.init()
	
	# set output weight matrix
	wout = N.random.rand(self.outs,self.size+self.ins) * 2 - 1
	wout = N.asfarray(wout, self.dtype)
	self.net.setWout( wout )
	
	## simulate network
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size),self.dtype)*2-1
	outdata = N.zeros((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	
	### get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	Wout = self.net.getWout()
	Wback = self.net.getWback()
	x = N.zeros((self.size))
	outtest = N.zeros((self.outs,self.sim_size),self.dtype)
	
	## recalc algorithm in python
	for n in range(self.sim_size):
		# calc new network activation
		x = (1-self.lr)*x + N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		if n > 0:
			x += N.dot( Wback, outtest[:,n-1] )
		# output = Wout * [x; in]
		outtest[:,n] = N.dot( Wout, N.r_[x, indata[:,n]] )
	
	assert_array_almost_equal(outdata,outtest,3)


    def testBP(self, level=1):
	""" test bandpass style neurons simulation
	(with random cutoff frequencies) """
        
	# setup net
	self.net.setInitAlgorithm(INIT_STD)
	self.net.setSimAlgorithm(SIM_BP)
	self.net.init()
	
	# set cutoff frequencies
	f1 = N.random.rand(self.size) * 0.8 + 0.1
	f2 = N.random.rand(self.size) * 0.8 + 0.1
	self.net.setBPCutoff(f1,f2)
	
	# set output weight matrix
	wout = N.random.rand(self.outs,self.size+self.ins) * 2 - 1
	wout = N.asfarray(wout, self.dtype)
	self.net.setWout( wout )
	
	# simulate network
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size),self.dtype)*2-1
	outdata = N.zeros((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	
	# get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	Wout = self.net.getWout()
	Wback = self.net.getWback()
	x = N.zeros((self.size))
	outtest = N.zeros((self.outs,self.sim_size),self.dtype)
	
	# parameters for bandpass filtering
	ema1 = N.zeros(x.shape)
	ema2 = N.zeros(x.shape)
	scale = f2 / f1 + 1.
	
	# recalc algorithm in python
	for n in range(self.sim_size):
		# calc new network activation
		x = N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		if n > 0:
			x += N.dot( Wback, outtest[:,n-1] )
		# bandpass style filtering
		ema1 = ema1 + f1 * (x-ema1)
		ema2 = ema2 + f2 * (ema1-ema2)
		x = (ema1 - ema2) * scale
		# output = Wout * [x; in]
		outtest[:,n] = N.dot( Wout, N.r_[x,indata[:,n]] )
	
	assert_array_almost_equal(outdata,outtest,3)


    def testBPConst(self, level=1):
	""" test bandpass style neurons simulation
	(with constant cutoff frequencies) """
        
	# setup net
	f1 = N.ones(self.size) * random.uniform(0.001,1.)
	f2 = N.ones(self.size) * random.uniform(0.,1.)
	self.net.setSimAlgorithm(SIM_BP)
	self.net.init()
	self.net.setBPCutoff(f1,f2)
	
	# set output weight matrix
	wout = N.random.rand(self.outs,self.size+self.ins) * 2 - 1
	wout = N.asfarray(wout, self.dtype)
	self.net.setWout( wout )
	
	# simulate network
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size),self.dtype)*2-1
	outdata = N.zeros((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	
	# get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	Wout = self.net.getWout()
	Wback = self.net.getWback()
	x = N.zeros((self.size))
	outtest = N.zeros((self.outs,self.sim_size),self.dtype)
	
	# parameters for bandpass filtering
	ema1 = N.zeros(x.shape)
	ema2 = N.zeros(x.shape)
	scale = 1 + f2/f1
	
	# recalc algorithm in python
	for n in range(self.sim_size):
		# calc new network activation
		x = N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		if n > 0:
			x += N.dot( Wback, outtest[:,n-1] )
		# bandpass style filtering
		ema1 = ema1 + f1 * (x-ema1)
		ema2 = ema2 + f2 * (ema1-ema2)
		x = (ema1 - ema2) * scale
		# output = Wout * [x; in]
		outtest[:,n] = N.dot( Wout, N.r_[x,indata[:,n]] )
	
	assert_array_almost_equal(outdata,outtest,3)


    def testIIRFilter(self, level=1):
	""" test IIR-Filter neurons simulation
	"""
	# setup net
	self.net.setInitAlgorithm(INIT_STD)
	self.net.setSimAlgorithm(SIM_FILTER)
	self.net.init()
	
	# set IIR coeffs (biquad bandpass filter)
	b = N.array(([0.5,0.,-0.5])) / 1.5
	a = N.array(([1.5,0.,0.5])) / 1.5
	B = N.ones((self.size,3)) * b
	A = N.ones((self.size,3)) * a
	self.net.setIIRCoeff(B,A)
	
	## set output weight matrix
	wout = N.random.rand(self.outs,self.size+self.ins) * 2 - 1
	wout = N.asfarray(wout, self.dtype)
	self.net.setWout( wout )
	
	## simulate network
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size),self.dtype)*2-1
	outdata = N.zeros((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	
	# get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	Wout = self.net.getWout()
	Wback = self.net.getWback()
	x = N.zeros((self.size))
	outtest = N.zeros((self.outs,self.sim_size),self.dtype)
	
	# initial conditions for filter
	zinit = N.zeros((self.size,2))
	
	# recalc algorithm in python
	for n in range(self.sim_size):
		#calc new network activation
		x = N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		if n > 0:
			x += N.dot( Wback, outtest[:,n-1] )
		# calc IIR filter:
		for i in range(self.size):
			insig = N.array(([x[i],])) # hack for lfilter
			x[i],zinit[i] = scipy.signal.lfilter(B[i,:], A[i,:], \
			                insig, zi=zinit[i])
		#output = Wout * [x; in]
		outtest[:,n] = N.dot( Wout, N.r_[x,indata[:,n]] )
	
	assert_array_almost_equal(outdata,outtest)


    def testSimFilter2(self, level=1):
	""" test SimFilter2 simulation
	"""
	# setup net
	self.net.setInitAlgorithm(INIT_STD)
	self.net.setSimAlgorithm(SIM_FILTER2)
	self.net.init()
	
	# set IIR coeffs (biquad bandpass filter)
	b = N.array(([0.5,0.,-0.5])) / 1.5
	a = N.array(([1.5,0.,0.5])) / 1.5
	B = N.ones((self.size,3)) * b
	A = N.ones((self.size,3)) * a
	self.net.setIIRCoeff(B,A)
	
	# set output weight matrix
	wout = N.random.rand(self.outs,self.size+self.ins) * 2 - 1
	wout = N.asfarray(wout, self.dtype)
	self.net.setWout( wout )
	
	# simulate network
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size),self.dtype)*2-1
	outdata = N.zeros((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	
	# get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	Wout = self.net.getWout()
	Wback = self.net.getWback()
	x = N.zeros((self.size))
	outtest = N.zeros((self.outs,self.sim_size),self.dtype)
	
	# initial conditions for filters
	fin = N.zeros((self.size,2))
	ffb = N.zeros((self.size,2))
	fx = N.zeros((self.size,2))
	states = N.zeros(self.size)
	insig = N.zeros(self.size)
	fbsig = N.zeros(self.size)
	
	# recalc algorithm in python
	for n in range(self.sim_size):
		# filter calculation
		for i in range(self.size):
			# filter reservoir states
			tmp = N.array(([x[i],]))
			states[i],fx[i] = scipy.signal.lfilter(B[i,:], A[i,:], \
			                tmp, zi=fx[i])
			# filter inputs
			tmp = N.array(([ N.dot(Win,indata[:,n])[i] ,]))
			insig[i],fin[i] = scipy.signal.lfilter(B[i,:], A[i,:], \
			                tmp, zi=fin[i])
			# filter outputs
			if n > 0:
				tmp = N.array(([N.dot(Wback,outtest[:,n-1]) \
				[i],]))
			else:
				tmp = N.array(([0.,]))
			fbsig[i],ffb[i] = scipy.signal.lfilter(B[i,:], A[i,:], \
			                tmp, zi=ffb[i])
		#calc new network activation
		x = N.dot( W, states ) + insig + fbsig
		#output = Wout * [x; in]
		outtest[:,n] = N.dot( Wout, N.r_[x,indata[:,n]] )
	
	assert_array_almost_equal(outdata,outtest)


    def testSerialIIRFilter(self, level=1):
	""" test seris of IIR filters in neurons
	"""
	# setup net
	self.net.setInitAlgorithm(INIT_STD)
	self.net.setSimAlgorithm(SIM_FILTER)
	self.net.init()
	
	# set IIR coeffs (biquad bandpass filter)
	b = N.array(([0.5,0.,-0.5])) / 1.5
	a = N.array(([1.5,0.,0.5])) / 1.5
	B = N.ones((self.size,6))
	A = N.ones((self.size,6))
	B[:,0:3] = B[:,0:3]*b
	A[:,0:3] = A[:,0:3]*a-0.01
	B[:,3:6] = B[:,3:6]*b
	A[:,3:6] = A[:,3:6]*a-0.01
	self.net.setIIRCoeff(B,A,2)
	
	# set output weight matrix
	wout = N.random.rand(self.outs,self.size+self.ins) * 2 - 1
	wout = N.asfarray(wout, self.dtype)
	self.net.setWout( wout )
	
	## simulate network
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size),self.dtype)*2-1
	outdata = N.zeros((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	
	# get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	Wout = self.net.getWout()
	Wback = self.net.getWback()
	x = N.zeros((self.size))
	outtest = N.zeros((self.outs,self.sim_size),self.dtype)
	
	# initial conditions for filters
	zinit1 = N.zeros((self.size,2))
	zinit2 = N.zeros((self.size,2))
	
	# recalc algorithm in python
	for n in range(self.sim_size):
		#calc new network activation
		x = N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		if n > 0:
			x += N.dot( Wback, outtest[:,n-1] )
		# calc IIR filter:
		for i in range(self.size):
			insig = N.array(([x[i],])) # hack for lfilter
			insig,zinit1[i] = scipy.signal.lfilter(B[i,0:3], \
			                 A[i,0:3], insig, zi=zinit1[i])
			x[i],zinit2[i] = scipy.signal.lfilter(B[i,3:6], \
			                 A[i,3:6], insig, zi=zinit2[i])
		#output = Wout * [x; in]
		outtest[:,n] = N.dot( Wout, N.r_[x,indata[:,n]] )
	
	assert_array_almost_equal(outdata,outtest)


if __name__ == "__main__":
    NumpyTest().run()
