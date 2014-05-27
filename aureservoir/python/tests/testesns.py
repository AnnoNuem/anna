import numpy as N
from aureservoir import *
import filtering, util
import pylab as P
from scipy import signal
from scipy.linalg import pinv, inv
import scipy.lib.lapack.clapack
#from scipy.fftpack import hilbert


class DelLine():
	""" helper class for a delay line """
	
	def __init__(self, delay, initvalues):
		""" init storage and make some checks
		delay         the delay in samples
		initvalues    the initial values in the delayline
		"""
		if( len(initvalues) != delay ):
			raise ValueError, "wrong size of initvalues !"
		self.delay = delay
		self.ringbuf = initvalues.copy()
		self.readpt = 0
		
	def tic(self, insample):
		""" one step - put in insample and returns output with
		the desired delaytime.
		"""
		if( self.delay == 0 ):
			return insample
		else:
			outsample = self.ringbuf[self.readpt]
			self.ringbuf[self.readpt] = insample
			self.readpt = (self.readpt+1) % self.delay
			return outsample


class IIRESN(DoubleESN):
	""" ESN with general IIR Filter neurons.
	2007, Georg Holzmann
	"""
	
	B = N.empty((1,1))  # feedforward path
	A = N.empty((1,1))  # feedback path
	serial = 1          # serial IIR filters
	fc = N.empty((1,1)) # center frequencies (of e.g. bandpass filters)
	
	def setStdESN(self):
		""" set filter coefficients so that we get a standard
		(unfiltered) ESN
		"""
		size = self.getSize()
		self.B = N.zeros((size,2))
		self.A = N.zeros((size,2))
		self.B[:,0] = 1.
		self.A[:,0] = 1.
	
	def setGammatoneFilters(self,fs,lowFreq):
		""" Set to each neuron one gammatone filter so that the filter
		frequencies will reach from lowFreq to fs/2 (each neuron will
		have a different gammatone filter !)
		
		fs         sample rate
		lowFreq    lowest filter frequency
		"""
		size = self.getSize()
		B, A = filtering.gammatone_biquad(fs,size,lowFreq)
		
		# 4 biquads
		self.B = N.empty((size,12))
		self.A = N.empty((size,12))
		self.serial = 4
		self.B[:,0:3] = B[:,0,:]
		self.B[:,3:6] = B[:,1,:]
		self.B[:,6:9] = B[:,2,:]
		self.B[:,9:12] = B[:,3,:]
		self.A[:,0:3] = A[:,0,:]
		self.A[:,3:6] = A[:,1,:]
		self.A[:,6:9] = A[:,2,:]
		self.A[:,9:12] = A[:,3,:]
		
	
	def setERB(self,fs,lowFreq):
		""" Set to each neuron to an Equivalent Rectangular Bandwidth
		Bandpass Filter.
		
		fs         sample rate
		lowFreq    lowest filter frequency
		"""
		size = self.getSize()
		self.B, self.A = filtering.ERB(fs,size,lowFreq)
	
	def setConstBPCutoffs(self, f=[0.1,0.3], bw=0.2, fs=1.):
		""" f is a list with bandpass centerfrequencies, bw the
		bandwidth in octaves
		"""
		size = self.getSize()
		nr = len(f)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		self.fc = N.zeros(size)
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,f[n%nr],'BPF',BW=bw)
			self.fc[n] = f[n%nr] / fs
	
	def setConstParametric(self, f=[0.1,0.3], bw=1., stopGain=-20, fs=1.):
		""" f is a list with bandpass centerfrequencies, bw the
		bandwidth in octaves, stopGain the gain in the stop bands
		"""
		size = self.getSize()
		nr = len(f)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		self.fc = N.zeros(size)
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,f[n%nr],'peakingEQ',BW=bw, \
			                 dBgain=-stopGain)
			self.fc[n] = f[n%nr] / fs
		
		# normalize parametric EQ to 0 dB
		g = 10**(-stopGain/20.)
		self.B = self.B / g
		
	def setLogBPCutoffs(self, f_start=0.001, f_stop=0.5, bw=0.2, fs=1.):
		""" set log spaced bandpass filters from f_start to f_stop
		with a bandwidth of bw octaves
		"""
		size = self.getSize()
		self.fc = N.logspace(N.log10(f_start), N.log10(f_stop), \
		               size, True)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'BPF',BW=bw)
		self.fc[n] /= fs
	
	def setLogParametric(self, f_start=0.001, f_stop=0.5, bw=0.2, \
		             stopGain=-20, fs=1.):
		""" set log spaced parametric filters from f_start to f_stop
		with a bandwidth of bw octaves and stopGain in the stop bands
		"""
		size = self.getSize()
		self.fc = N.logspace(N.log10(f_start), N.log10(f_stop), \
		               size, True)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'BPF',BW=bw)
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'peakingEQ',BW=bw, \
			                 dBgain=-stopGain)
		self.fc[n] /= fs
		
		# normalize parametric EQ to 0 dB
		g = 10**(-stopGain/20.)
		self.B = self.B / g
	
	def setLinBPCutoffs(self, f_start=0.001, f_stop=0.5, bw=0.2, fs=1.):
		""" set linear spaced bandpass filters from f_start to f_stop
		with a bandwidth of bw octaves
		"""
		size = self.getSize()
		self.fc = N.linspace(f_start, f_stop, size)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'BPF',BW=bw)
		self.fc[n] /= fs
	
	def setLinParametric(self, f_start=0.001, f_stop=0.5, bw=0.2, \
		             stopGain=-20, fs=1.):
		""" set linear spaced parametric filters from f_start to f_stop
		with a bandwidth of bw octaves and stopGain in the stop bands
		"""
		size = self.getSize()
		self.fc = N.linspace(f_start, f_stop, size)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'BPF',BW=bw)
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'peakingEQ',BW=bw, \
			                 dBgain=-stopGain)
		self.fc[n] /= fs
		
		# normalize parametric EQ to 0 dB
		g = 10**(-stopGain/20.)
		self.B = self.B / g
	
	def setGroupBP(self, f_start=0.001, f_stop=0.5, bw=0.2, \
		       groups=10, spacing="log", fs=1.):
		""" groups neurons in groups with same filters, the spacing can
		be log or lin
		"""
		size = self.getSize()
		if spacing == "lin":
			f = N.linspace(f_start, f_stop, groups)
		elif spacing == "log":
			f = N.logspace(N.log10(f_start), N.log10(f_stop), \
		               groups, True)
		else:
			raise ValueError, "unknown spacing"
		self.setConstBPCutoffs(f, bw, fs)
	
	def setGroupParametric(self, f_start=0.001, f_stop=0.5, bw=0.2, \
		       stopGain=-20, groups=10, spacing="log", fs=1.):
		""" groups neurons in groups with same filters, the spacing can
		be log or lin
		"""
		size = self.getSize()
		if spacing == "lin":
			f = N.linspace(f_start, f_stop, groups)
		elif spacing == "log":
			f = N.logspace(N.log10(f_start), N.log10(f_stop), \
		               groups, True)
		else:
			raise ValueError, "unknown spacing"
		self.setConstParametric(f, bw, stopGain, fs)
	
	def setRandomFIR(self, order=2):
		""" generates minimum-phase FIR filters with random zeros
		for each neuron
		order = nr of zeros in one filter
		"""
		size = self.getSize()
		zeros = (N.random.rand(size,order)*2-1) + \
		        (N.random.rand(size,order)*2-1)*1j
		k = 1.
		self.B = N.zeros((size,order+1))
		self.A = N.zeros((size,order+1))
		for n in range(size):
			self.B[n],self.A[n] = \
			signal.zpk2tf(zeros[n], [], k)
	
	def setRandomIIR(self, order=2):
		""" generates minimum-phase IIR filters with random zeros
		and poles for each neuron
		order = nr of zeros/poles in one filter
		"""
		size = self.getSize()
		zeros = (N.random.rand(size,order)*2-1) + \
		        (N.random.rand(size,order)*2-1)*1j
		poles = (N.random.rand(size,order)*2-1) + \
		        (N.random.rand(size,order)*2-1)*1j
		k = 1.
		self.B = N.zeros((size,order+1))
		self.A = N.zeros((size,order+1))
		for n in range(size):
			self.B[n],self.A[n] = \
			signal.zpk2tf(zeros[n], poles[n], k)
		
	def init(self):
		""" overrides the ESN init() method to additionaly set the
		bandpass frequencies """
		
		if( self.B.size == 1 or self.A.size == 1 ):
			print "IIRESN filter parameters not initialized !"
		
		size = self.getSize()
		
		DoubleESN.init(self)
		self.setIIRCoeff(self.B,self.A,self.serial)
	
	def plot_filters(self,every=1,fftsize=16384):
		""" plots the currently set filters
		"""
		neurons = self.getSize()
		impulse = N.zeros(fftsize)
		impulse[0] = 1
		response = N.empty((neurons,fftsize/2+1))
		
		# get frequency response of neurons
		if self.serial == 1:
			for n in range(0,neurons,every):
				tmp = signal.lfilter(self.B[n],self.A[n], impulse)
				response[n] = N.fft.rfft( tmp )
		else:
			fstep = self.B.shape[1] / self.serial
			for n in range(0,neurons,every):
				tmp = signal.lfilter(self.B[n,0:fstep], \
				                     self.A[n,0:fstep], \
				                     impulse)
				for i in range(1,self.serial):
					beg = i*fstep
					end = (i+1)*fstep
					#print i, beg, end
					tmp = signal.lfilter(self.B[n,beg:end], \
					                     self.A[n,beg:end], \
							     tmp )
				response[n] = N.fft.rfft( tmp )
		
		# color map
		col = ['b','g','r','c','m','y','k']
		
		# plot amplitude response
		le = len(abs(response[0]))
		freqscale = N.arange(le) / (float(le)/(20000))
		denormal = 1e-20
		for n in range(0,neurons,every):
			tmp = 20.*N.log10(abs(response[n])+denormal)
			#tmp = abs(response[n])
			#P.title("log spaced bandpass filters, BW = 1.5 octaves")
			P.title("parametric EQs, BW = 2 octaves, stop = -20dB")
			P.semilogx(freqscale,tmp,col[n%len(col)])
			#P.axis([20, 20000, -50, 5])
			P.axis([100, 20000, -30, 5])
			P.xlabel("frequency (Hz)")
			P.ylabel("amplitude (dB)")
		P.show()


class DSESN(IIRESN):
	""" ESN class for some experiments.
	
	2007, Georg Holzmann
	"""
	
	# maximum delay of the delayline
	maxdelay = 100
	
	# prefiltering of the cross correlation
	gcctype = 'unfiltered'
	
	# additional squared state updates
	squareupdate = 0
	
	def train(self, indata, outdata, washout):
		""" set to desired train method here """
		#DoubleESN.train(self,indata,outdata,washout)
		self.trainDelaySum(indata,outdata,washout)
	
	
	def simulate(self, indata, outdata):
		""" set to desired simulate method here """
		self.simulateDelaySum(indata,outdata)
		#DoubleESN.simulate(self,indata,outdata)
	
	
	def trainDelaySum(self, indata, outdata, washout):
		""" Calculates the optimal delay for each readout neuron with
		the crosscorrelation.
		"""
		steps = indata.shape[1]
		size = self.getSize()
		maxdelay = self.maxdelay
		insig = indata.copy()
		
		# teacher forcing
		X = self._teacherForcing(insig,outdata)
		#print "indata[0]",indata.shape,indata[0]
		#print "X[0]",X.shape,X[0]
		
		self.delays = N.zeros(size+self.getInputs())
		for n in range(size):
			xcorr = abs( util.GCC(X[n],outdata[0], \
			             self.gcctype, 0) )
			self.delays[n] = xcorr[0:maxdelay+1].argmax()
		for n in range(self.getInputs()):
			xcorr = abs( util.GCC(insig[n],outdata[0], \
			             self.gcctype, 0) )
			self.delays[n+size] = xcorr[0:maxdelay+1].argmax()
		del xcorr
		
		# setup the "delay lines" for each neuron
		self.dellines = []
		
		# now delay neuron signals as calculated:
		# add zeros to the beginning and store the rest into the delaylines
		for n in range(size):
			d = self.delays[n]
			rest = X[n,steps-d:steps].copy()
			X[n,:] = N.r_[ N.zeros(d), X[n,0:steps-d] ]
			self.dellines.append( DelLine(d,rest) )
		for n in range(self.getInputs()):
			d = self.delays[n+size]
			rest = insig[n,steps-d:steps].copy()
			insig[n,:] = N.r_[ N.zeros(d), insig[n,0:steps-d] ]
			self.dellines.append( DelLine(d,rest) )
		del rest
		
		# some debugging
		#print "delays:", self.delays
		
		# calc new washout beacuse of the delay
		# (is this good ? - should be better given from outside)
		#washout += int(self.delays.max())
		#print "effective trainsize:",steps-washout
		
		
		if (self.squareupdate == 0):
			# restructure data
			M = N.r_[X,insig]
			M = M[:,washout:steps].T
			T = outdata[:,washout:steps].T
			
			# calc pseudo inverse: wout = pinv(M) * T
			# or with least square (much faster):
			v,wout,s,rank,info = scipy.lib.lapack.clapack.dgelss( M, T )
			self.wout = wout[0:size+self.getInputs()].T
			
		else:
			# restructure data
			M = N.r_[X,insig,X**2,insig**2]
			M = M[:,washout:steps].T
			T = outdata[:,washout:steps].T
			
			# calc pseudo inverse: wout = pinv(M) * T
			# or with least square (much faster):
			v,wout,s,rank,info = scipy.lib.lapack.clapack.dgelss( M, T )
			self.wout = wout[0:2*(size+self.getInputs())].T
		
		# set the readout to the radius
		self.setWout(self.wout.copy())
	
	
	def simulateDelaySum(self, indata, outdata):
		""" Simulation algorithm for a DelayAndSum readout.
		"""
		#print "TestESN: Simulate DelaySum ..."
		
		steps = indata.shape[1]
		outs = self.getOutputs()
		outtmp = N.empty((outs,))
		
		for n in range(steps):
			self.simulateDelaySumStep(indata[:,n].flatten(),outtmp)
			outdata[:,n] = outtmp
	
	
	def simulateDelaySumStep(self, indata, outdata):
		""" One-Step Simulation algorithm for a DelayAndSum readout.
		"""
		neurons = self.getSize()
		outtmp = N.empty((self.getOutputs(),))
		
		# simulate one step
		self.simulateStep(indata, outtmp)
		x = self.getX().copy()
		
		# get delayed value for each neuron and input
		for i in range(neurons):
			x[i] = self.dellines[i].tic(x[i])
		for i in range(self.getInputs()):
			indata[i] = self.dellines[neurons+i].tic(indata[i])
		
		if self.squareupdate==0:
			state = N.r_[x,indata]
		else:
			state = N.r_[x,indata,x**2,indata**2]
		outdata[:] = N.dot( self.wout, state ).copy()
		self.setLastOutput(outdata)
	
	
	def _teacherForcing(self, indata, outdata):
		""" teacher forcing and collect internal states
		"""
		#print "TestESN: teacher forcing ..."
		
		steps = indata.shape[1]
		neurons = self.getSize()
		X = N.empty((neurons,steps))
		outtmp = N.empty((self.getOutputs(),))
		ins = self.getInputs()
		outs = self.getOutputs()
		
		# step by step ESN simulation
		for n in range(steps):
			self.simulateStep(indata[:,n].flatten(), outtmp)
			self.setLastOutput(outdata[:,n].flatten())
			X[:,n] = self.getX().copy()
		
		return X
