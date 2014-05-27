from numpy import *
from scipy import fftpack, signal


def GCC(x,y,filt="unfiltered",fftshift=1, b=None, a=None):
	""" Generalized Cross-Correlation of _real_ signals x and y with
	specified pre-whitening filter.
	
	The GCC is computed with a pre-whitening filter onto the
	cross-power spectrum in order to weight the magnitude value
	against its SNR. The weighted CPS is used to obtain the
	cross-correlation in the time domain with an inverse FFT.
	The result is _not_ normalized.
	
	See "The Generalized Correlation Method for Estimation of Time Delay"
	by Charles Knapp and Clifford Carter, programmed with looking at the
	matlab GCC implementation by Davide Renzi.
	
	x, y      input signals on which gcc is calculated
	filt      the pre-whitening filter type (explanation see below)
	fftshift  if not zero the final ifft will be shifted
	returns   the gcc in time-domain
	
	descibtion of pre-whitening filters:
	
	'unfiltered':
	performs simply a crosscorrelation
	
	'roth':
	this processor suppress frequency regions where the noise is
	large than signals.
	
	'scot': Smoothed Coherence Transform (SCOT)
	this processor exhibits the same spreading as the Roth processor.
	
	'phat': Phase Transform (PHAT)
	ad hoc technique devoleped to assign a specified weight according
	to the SNR.
	
	'cps-m': SCOT filter modified
        this processor computes the Cross Power Spectrum Density and
        apply the SCOT filter with a power at the denominator to avoid
        ambient reverberations that causes false peak detection.
	
	'ht': Hannah and Thomson filter (HT)
        HT processor computes a PHAT transform weighting the phase
        according to the strength of the coherence.
	
	'prefilter': general IIR prefilter
	with b and a a transfer function of an IIR filter can be given, which
	will be used as a prefilter (can be a "non-whitening" filter)
	
	2007, Georg Holzmann
	"""
	L = max( len(x), len(y) )
	fftsize = int(2**ceil(log(L)/log(2))) # next power2
	X = fft.rfft(x, fftsize)
	Y = fft.rfft(y, fftsize)
	
	# calc crosscorrelation
	Gxy = X.conj() * Y
	
	# calc the filters
	
	if( filt == "unfiltered" ):
		Rxy = Gxy
		
	elif( filt == "roth" ):
		Gxx = X.conj() * X
		W = ones(Gxx.shape,Gxx.dtype)
		W[Gxx!=0] = 1. / Gxx[Gxx!=0]
		Rxy = Gxy * W
		
	elif( filt == 'scot' ):
		Gxx = X.conj() * X
		Gyy = Y.conj() * Y
		W = ones(Gxx.shape,Gxx.dtype)
		tmp = sqrt(Gxx * Gyy)
		W[tmp!=0] = 1. / tmp[tmp!=0]
		Rxy = Gxy * W
	
	elif( filt == 'phat' ):
		W = ones(Gxy.shape,Gxy.dtype)
		tmp = abs(Gxy)
		W[tmp!=0] = 1. / tmp[tmp!=0]
		Rxy = Gxy * W
	
	elif( filt == 'cps-m' ):
		Gxx = X.conj() * X
		Gyy = Y.conj() * Y
		W = ones(Gxx.shape,Gxx.dtype)
		factor = 0.75 # common value between .5 and 1
		tmp = (Gxx * Gyy)**factor
		W[tmp!=0] = 1. / tmp[tmp!=0]
		Rxy = Gxy * W
	
	elif( filt == 'ht' ):
		Gxx = X.conj() * X
		Gyy = Y.conj() * Y
		W = ones(Gxy.shape,Gxy.dtype)
		gamma = W.copy()
		# coherence function evaluated along the frame
		tmp = sqrt(Gxx * Gyy)
		gamma[tmp!=0] = abs(Gxy[tmp!=0]) / tmp[tmp!=0]
		# HT filter
		tmp = abs(Gxy) * (1-gamma)
		W[tmp!=0] = gamma[tmp!=0] / tmp[tmp!=0]
		Rxy = Gxy * W
	
	elif( filt == 'prefilter' ):
		# calc frequency response of b,a filter
		impulse = zeros(fftsize)
		impulse[0] = 1
		h = signal.lfilter(b,a,impulse)
		H = fft.rfft(h, fftsize)
		Rxy = H * Gxy  # hm ... conj da irgendwo ?
		
	else:
		raise ValueError, "wrong pre-whitening filter !"
	
	# inverse transform with optional fftshift
	if( fftshift!=0 ):
		gcc = fftpack.fftshift( fft.irfft( Rxy ) )
	else:
		gcc = fft.irfft( Rxy )
	return gcc


def nextpower2(x):
	""" Computes the next power of two value of x.
	E.g x=3 -> 4, x=5 -> 8
	"""
	exponent = ceil(log(x)/log(2))
	return 2**exponent


#L=max(length(x),length(y))
#NFFT=2*L
#xx=fft(x,NFFT)
#yy=fft(y,NFFT)
#Sxy=xx.*conj(yy)
#rxy=fftshift(real(ifft(Sxy)));
#[k,J]=max(rxy)
#k=51
#J=14