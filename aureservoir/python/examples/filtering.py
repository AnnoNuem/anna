###########################################################
# Tools to calculate filter parameters
#
# 2007, Georg Holzmann
###########################################################

import numpy as N
import scipy.signal

__all__ = [
    'fftfilt'
    'tominphase',
    'biquad',
    'gammatone',
    'gammatone_biquad',
    'ERB'
    ]

def fftfilt(signal, b, a):
	""" Approximation of an IIR-filter (given b and a) in frequency domain.
	This calculates the frequency response of the IIR filter and then
	approximates it with fftsize same as the length of the input signal.
	"""
	fftsize = len(signal)
	impulse = N.zeros(fftsize)
	impulse[0] = 1
	h = scipy.signal.lfilter(b,a,impulse)
	
	# go in frequency domain
	H = N.fft.rfft(h, fftsize)
	X = N.fft.rfft(signal, fftsize)
	Y = H * X
	y = N.fft.irfft(Y)
	return y


def tominphase(b, a, verbose=0):
	""" Converts a linear IIR filter, represented with numerator b and denominator a,
	to a minimum phase filter with the same amplitude response.
	Returns new (b,a) pair
	"""
	# get poles and zeros
	z,p,k = scipy.signal.tf2zpk(b,a)
	
	if( verbose != 0 ):
		print "tominphase, zeros:"
		print "\tabs before minphase: ", abs(z)
	
	# if abs(zero)>1, then reflect zero inside the unit circle to 1/zero
	for n in range(len(z)):
		if( abs(z[n])>1 ):
			z[n] = 1/z[n]
		if( abs(z[n])==1 ):
			print "tominphase warning: zero at the unit circle ! scaling ..."
			z[n] -= 10**(-12)*(1+1j)
	
	if( verbose != 0 ):
		print "\tabs after minphase: ", abs(z)
	
	return scipy.signal.zpk2tf(z,p,k)


def biquad(Fs, f0, ftype='LPF', Q=1., BW=None, dBgain=0.):
	""" Generates some common filter types for a biquad IIR filter.
	
	Implemented after
	"Cookbook formulae for audio EQ biquad filter coefficients"
	by Robert Bristow-Johnson
	http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
	
	Fs       the sampling frequency
	f0       "wherever it's happenin', man."  Center Frequency or
	         Corner Frequency, or shelf midpoint frequency, depending
	         on which filter type.  The "significant frequency".
	ftype    filter type, must be a string out of: "LPF", "HPF", "BPF",
	         "notch", "APF", "peakingEQ", "lowShelf", "highShelf"
	Q        the EE kind of definition, except for peakingEQ in which A*Q is
	         the classic EE Q.  That adjustment in definition was made so that
	         a boost of N dB followed by a cut of N dB for identical Q and
	         f0/Fs results in a precisely flat unity gain filter or "wire".
	BW       can be used _instead_ of Q to set the bandwidth in octaves
	         (between -3 dB frequencies for BPF and notch or between midpoint
	         (dBgain/2) gain frequencies for peaking EQ)
	dBgain   used for peaking and shelving filters to set the gain in dB
	
	returns  B,A for scipy.signal.lfilter
	
	All filter transfer functions were derived from analog prototypes
	and had been digitized using the Bilinear Transform.
	
	2007,
	Georg Holzmann
	"""
	
	#some checks
	filtertypes = ["LPF", "HPF", "BPF", "notch", "APF", "peakingEQ",
	               "lowShelf", "highShelf"]
	if( ftype not in filtertypes ):
		raise ValueError, "Wrong filter type !"
	
	# some intermediate variables
	A = 10**(dBgain/40.)
	w0 = 2 * N.pi * f0 / Fs
	if( BW != None ):
		#print BW
		alpha = N.sin(w0)*N.sinh( N.log(2)/2 * BW * w0/N.sin(w0) )
		#Q = ( 2*N.sinh(N.log(2)/2*BW*w0/N.sin(w0)) )**(-1)
		#print Q
	else:
		# calc with Q
		alpha = N.sin(w0)/(2.*Q)
	
	# parameter arrays
	Bfilt = N.zeros(3)    # forward path
	Afilt = N.zeros(3)    # feedback path
	
	if( ftype=='LPF' ):
		Bfilt[0] = (1 - N.cos(w0)) / 2.
		Bfilt[1] = 1 - N.cos(w0)
		Bfilt[2] = (1 - N.cos(w0)) / 2.
		Afilt[0] = 1 + alpha
		Afilt[1] = -2*N.cos(w0)
		Afilt[2] = 1 - alpha
	elif( ftype=='HPF' ):
		Bfilt[0] = (1 + N.cos(w0))/2.
		Bfilt[1] = -(1 + N.cos(w0))
		Bfilt[2] = (1 + N.cos(w0))/2.
		Afilt[0] = 1 + alpha
		Afilt[1] = -2*N.cos(w0)
		Afilt[2] = 1 - alpha
	elif( ftype=='BPF' ):
		# constant 0dB peak gain
		Bfilt[0] = alpha
		Bfilt[1] = 0
		Bfilt[2] = -alpha
		Afilt[0] = 1 + alpha
		Afilt[1] = -2*N.cos(w0)
		Afilt[2] = 1 - alpha
	elif( ftype=='notch' ):
		Bfilt[0] = 1.
		Bfilt[1] = -2*N.cos(w0)
		Bfilt[2] = 1.
		Afilt[0] = 1 + alpha
		Afilt[1] = -2*N.cos(w0)
		Afilt[2] = 1 - alpha
	elif( ftype=='APF' ):
		Bfilt[0] = 1 - alpha
		Bfilt[1] = -2*N.cos(w0)
		Bfilt[2] = 1 + alpha
		Afilt[0] = 1 + alpha
		Afilt[1] = -2*N.cos(w0)
		Afilt[2] = 1 - alpha
	elif( ftype=='peakingEQ' ):
		Bfilt[0] = 1 + alpha*A
		Bfilt[1] = -2*N.cos(w0)
		Bfilt[2] = 1 - alpha*A
		Afilt[0] = 1 + alpha/A
		Afilt[1] = -2*N.cos(w0)
		Afilt[2] = 1 - alpha/A
	elif( ftype=='lowShelf' ):
		Bfilt[0] = A*((A+1)-(A-1)*N.cos(w0) + 2*N.sqrt(A)*alpha)
		Bfilt[1] = 2*A*( (A-1) - (A+1)*N.cos(w0) )
		Bfilt[2] = A*((A+1)-(A-1)*N.cos(w0)-2*N.sqrt(A)*alpha)
		Afilt[0] = (A+1)+(A-1)*N.cos(w0)+2*N.sqrt(A)*alpha
		Afilt[1] = -2*( (A-1) + (A+1)*N.cos(w0))
		Afilt[2] = (A+1) + (A-1)*N.cos(w0)-2*N.sqrt(A)*alpha
	elif( ftype=='highShelf' ):
		Bfilt[0] = A*((A+1)+(A-1)*N.cos(w0)+2*N.sqrt(A)*alpha)
		Bfilt[1] = -2*A*( (A-1) + (A+1)*N.cos(w0) )
		Bfilt[2] = A*( (A+1) + (A-1)*N.cos(w0)-2*N.sqrt(A)*alpha )
		Afilt[0] = (A+1) - (A-1)*N.cos(w0) + 2*N.sqrt(A)*alpha
		Afilt[1] = 2*( (A-1) - (A+1)*N.cos(w0) )
		Afilt[2] = (A+1) - (A-1)*N.cos(w0) - 2*N.sqrt(A)*alpha
	else:
		raise ValueError, "Wrong filter type !"
	
	return Bfilt, Afilt


def gammatone(fs,numChannels,lowFreq):
	""" Computes the filter coefficients for a bank of 
	Gammatone filters.  These filters were defined by Patterson and 
	Holdworth for simulating the cochlea.
	
	(implemented after "An Efficient Implementation of the 
	Patterson-Holdsworth Auditory Filter Bank" by Malcolm Slaney)
	
	returns     B, A
	
	Each row of the returned filter arrays (B=forward, A=feedback)
	can be passed to the scipy.signal.lfilter function (same as matlabs
	filter function).

	The filter bank contains "numChannels" channels that extend from
	half the sampling rate (fs) to "lowFreq".
	
	Note: there can be numerical problems with very small cfs (100Hz) and
	large sample rates (44kHz), when a number of poles are combined.
	These small errors lead to poles outside the unit circle and instability.
	Therefore an implementation with 4 biquads should be used
	(see makeGammatoneFilterBiquad).
	
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
	
	# calculate filter coefficients
	Afilt = N.zeros((len(cf),9))  # feedback path
	Bfilt = N.zeros((len(cf),5))  # forward path
	Bfilt[:,0] = T**4 / gain
	Bfilt[:,1] = -4*T**4*N.cos(2*cf*N.pi*T) / N.exp(B*T) / gain
	Bfilt[:,2] = 6*T**4*N.cos(4*cf*N.pi*T) / N.exp(2*B*T) / gain
	Bfilt[:,3] = -4*T**4*N.cos(6*cf*N.pi*T) / N.exp(3*B*T) / gain
	Bfilt[:,4] =  T**4*N.cos(8*cf*N.pi*T) / N.exp(4*B*T) / gain
	Afilt[:,0] = 1.
	Afilt[:,1] = -8*N.cos(2*cf*N.pi*T) / N.exp(B*T)
	Afilt[:,2] = 4*(4 + 3*N.cos(4*cf*N.pi*T)) / N.exp(2*B*T)
	Afilt[:,3] = -8*(6*N.cos(2*cf*N.pi*T) + N.cos(6*cf*N.pi*T)) / N.exp(3*B*T)
	Afilt[:,4] = 2*(18 + 16*N.cos(4*cf*N.pi*T)+N.cos(8*cf*N.pi*T)) / N.exp(4*B*T)
	Afilt[:,5] = -8*(6*N.cos(2*cf*N.pi*T) + N.cos(6*cf*N.pi*T)) / N.exp(5*B*T)
	Afilt[:,6] = 4*(4 + 3*N.cos(4*cf*N.pi*T)) / N.exp(6*B*T)
	Afilt[:,7] = -8*N.cos(2*cf*N.pi*T) / N.exp(7*B*T)
	Afilt[:,8] = N.exp(-8*B*T)
	
	return Bfilt,Afilt


def gammatone_biquad(fs,numChannels,lowFreq):
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


def ERB(fs,numChannels,lowFreq):
	""" Equivalent Rectangular Bandwidth Filter Coefficients for
	biquad IIR Filter.
	
	(implemented after "An Efficient Implementation of the 
	Patterson-Holdsworth Auditory Filter Bank" by Malcolm Slaney)
	
	returns     B, A
	
	Each row of the returned filter arrays (B=forward, A=feedback)
	can be passed to the scipy.signal.lfilter function (same as matlabs
	filter function).

	The filter bank contains "numChannels" channels that extend from
	half the sampling rate (fs) to "lowFreq".
	
	2007,
	Georg Holzmann
	"""
	print "ERB WARNUNG: check this again - it seems that there is a \
	       problem here somewhere !"
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
	
	# bandwidth = f0 / Q -> Q = f0 / bandwidth
	filterQ = cf / ERB
	
	# calc biquad bandpass filters
	Bfilt = N.zeros((len(cf),3))
	Afilt = N.zeros((len(cf),3))
	for n in range(len(cf)):
		Bfilt[n],Afilt[n] = biquad(fs,cf[n],'BPF',Q=filterQ[n])
	
	return Bfilt, Afilt

