###########################################################
# Some mean square error calculation utilities
#
# 2007, Georg Holzmann
###########################################################

import numpy as N

__all__ = [
    'nrmse',
    'nmse'
    ]

def nrmse( insignal, targetsignal, discard=0 ):
	"""
	Calculates the NRMSE (normalized root mean square error)
	of the input signal compared to the target signal.
	Initial values can be discarded with discard=n.
	
	2007, Georg Holzmann
	"""
	# TODO: make for matrix in and target
	
	# reshape values
	insignal.shape = -1,
	targetsignal.shape = -1,
	
	if( targetsignal.size > insignal.size ):
		maxsize = insignal.size
	else:
		maxsize = targetsignal.size
	
	origsig = targetsignal[discard:maxsize]
	testsig = insignal[discard:maxsize]
	
	error = (origsig - testsig)**2
	nrmse = N.sqrt( error.mean() / (origsig.std()**2) )
	
	return nrmse

def nmse( insignal, targetsignal, discard=0 ):
	"""
	Calculates the NMSE (normalized mean square error)
	of the input signal compared to the target signal.
	Initial values can be discarded with discard=n.
	
	2007, Georg Holzmann
	"""
	# TODO: make for matrix in and target
	
	# reshape values
	insignal.shape = -1,
	targetsignal.shape = -1,
	
	if( targetsignal.size > insignal.size ):
		maxsize = insignal.size
	else:
		maxsize = targetsignal.size
	
	origsig = targetsignal[discard:maxsize]
	testsig = insignal[discard:maxsize]
	
	error = (origsig - testsig)**2
	nmse = error.mean() / (origsig.std()**2)
	
	return nmse

