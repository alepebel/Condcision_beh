from scipy.stats import circmean,circvar
from cmath import phase
from  numpy import array
from scipy.stats import circmean,circvar,circstd
from numpy import *
from cmath import phase
from matplotlib.pylab import *


def len2(x):
	if type(x) is not type([]):
		if type(x) is not type(array([])):
			return -1
	return len(x)

def phase2(x):
	if not isnan(x):
		return phase(x)
	return nan

def circdist(angles1,angles2):
	import numpy as np
	return np.angle(np.exp(1j*np.asarray(angles1))/np.exp(1j*np.asarray(angles2)))

def circ_mean(x):
	from scipy.stats import circmean
	from numpy import pi
	return circmean(x,low=-pi,high=pi)
