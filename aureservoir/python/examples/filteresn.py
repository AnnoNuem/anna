###########################################################
# Helper class for Filter and Bandpass ESN
#
# 2007, Georg Holzmann
###########################################################

import numpy as N
import sys
sys.path.append("../")
from aureservoir import *
import filtering
import pylab as P
from scipy import signal


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


class BPESN(DoubleESN):
	""" A bandpass style ESN with different cutoff frequencies
	combinations.
	
	2007, Georg Holzmann
	"""
	
	f1 = N.empty(1)
	f2 = N.empty(1)
	
	def setOctaveCutoffs(self,groups=10,start_f1=0.2,start_f2=0.1):
		""" set bandpass cutoffs: always go down one octave
		
		    groups     divide reservoir in that many groups
		    start_f1   starting f1 frequency, each next group
		               will have the half of this value
		    start_f2   starting f2 frequency
		"""
		size = self.getSize()
		self.f1 = N.empty(size)
		self.f2 = N.empty(size)
		subsize = size / groups
		
		for n in range(groups):
			self.f1[n*subsize:(n+1)*subsize] = start_f1 * 2.**(-n)
			self.f2[n*subsize:(n+1)*subsize] = start_f2 * 2.**(-n)

	def setLinCutoffs(self, start_f1=0.0001, stop_f1=1., \
		              start_f2=0.0001, stop_f2=1.):
		""" set linear spaced bandpass cutoffs between
		start_f1, stop_f1 and start_f2, stop_f2
		"""
		size = self.getSize()
		self.f1 = N.linspace(start_f1,stop_f1,size)
		self.f2 = N.linspace(start_f2,stop_f2,size)

	def setLogCutoffs(self, start_f1=0.001, stop_f1=1., \
		              start_f2=0.001, stop_f2=1.):
		""" set log spaced bandpass cutoffs between
		start_f1, stop_f1 and start_f2, stop_f2
		"""
		size = self.getSize()
		self.f1 = N.logspace(N.log10(start_f1), N.log10(stop_f1), \
		                     size, True)
		self.f2 = N.logspace(N.log10(start_f2), N.log10(stop_f2), \
		                     size, True)

	def setConstCutoffs(self, f1=1., f2=0.):
		""" set const cutoffs to f1 and f2
		"""
		size = self.getSize()
		self.f1 = N.ones(size) * f1
		self.f2 = N.ones(size) * f2

	def init(self):
		""" overrides the ESN init() method to additionaly set the
		bandpass frequencies """
		
		if( self.f1.size == 1 or self.f2.size == 1 ):
			raise ValueError, "BPESN cutoff not initialized !"
		
		self.setSimAlgorithm(SIM_BP)
		size = self.getSize()
		
		DoubleESN.init(self)
		self.setBPCutoff(self.f1,self.f2)

