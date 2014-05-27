import sys
from numpy.testing import *
import numpy as N
import random, scipy.signal

# TODO: right module and path handling
sys.path.append("../")
from aureservoir import *


class test_adaptation(NumpyTestCase):

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
	self.net.setReservoirAct(ACT_TANH2)
	self.net.setOutputAct(ACT_LINEAR)
	self.net.setSize( self.size )
	self.net.setInputs( self.ins )
	self.net.setOutputs( self.outs )
	self.net.setInitParam(CONNECTIVITY, self.conn)
	self.net.setInitParam(IN_CONNECTIVITY, 1.)
	self.net.setInitParam(FB_CONNECTIVITY, 0.)


    def testGaussianIP(self, level=1):
	""" test Gaussian IP reservoir adaptation """
        
	# setup net
	lr = random.uniform(0.1,0.0001)
	mean = random.uniform(-0.1,0.1)
	var = random.uniform(0.001,0.3)
	self.net.setInitParam(IP_LEARNRATE, lr)
	self.net.setInitParam(IP_MEAN, mean)
	self.net.setInitParam(IP_VAR, var)
	self.net.init()
	
	# indata
	indata = N.asfarray(N.random.rand(self.ins,self.sim_size), \
	                    self.dtype) * 2 - 1
	
	# adapt reservoir
	self.net.adapt(indata)
	
	# simulate network and collect states
	states = N.zeros((self.size,self.sim_size),self.dtype)
	outtmp = N.zeros((self.outs),self.dtype)
	for n in range(self.sim_size):
		intmp = indata[:,n].copy()
		self.net.simulateStep( intmp, outtmp )
		states[:,n] = self.net.getX().copy()
		
	# get data to python
	W = N.zeros((self.size,self.size),self.dtype)
	self.net.getW( W )
	Win = self.net.getWin()
	x = N.zeros((self.size))
	
	# recalc adaptation algorithm
	a = N.ones(x.shape)
	b = N.zeros(x.shape)
	for n in range(self.sim_size):
		x = N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		y = N.tanh( a*x + b )
		db = -lr*(-mean/var+(y/var)*(2*var+1-y**2+mean*y))
		b += db
		a = a + lr/a + db*x
	
	# recalc simulation and collect states
	states2 = N.zeros((self.size,self.sim_size),self.dtype)
	for n in range(self.sim_size):
		x = N.dot( W, x )
		x += N.dot( Win, indata[:,n] )
		x = N.tanh( a*x + b )
		states2[:,n] = x
	
	assert_array_almost_equal(states,states2)


if __name__ == "__main__":
    NumpyTest().run()
