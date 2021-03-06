## Spectro
## =======
## This program analyzes UV-VIS absorption spectra from aqueous surfactant suspended dispersions of carbon nanotubes.
## It does so by fitting absorption profile models from the literature with a linear regression at each step of a non-linear regression fitting of the background (amorphous carbon and pi plasmon resonances) model.
##
## This program is written using python 2.7 with a Qt4 Gui and has the following dependencies:
##
## pythonxy:
## http://www.mirrorservice.org/sites/pythonxy.com/Python(x,y)-2.7.6.1.exe
##
## lmfit:
## https://pypi.python.org/packages/2.7/l/lmfit/lmfit-0.7.4.win32-py2.7.exe
##
## This program was written by Chase Brown and is under the GNU General Public License.

from PyQt4 import QtCore, QtGui
import sys
from math import factorial
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from scipy import *
import scipy
import lmfit
import multiprocessing as mp
import time
import datetime
import xlsxwriter
import operator

##
## CONSTANTS
##

# A value of 1 for APP_SCREEN_RATIO will open the program in a 'maximized' type of position.
APP_SCREEN_RATIO = 0.7
# Due to the Global Interepreter Lock (GIL) in python, we have to switch the main thread from updating the GUI front end to doing calculations in the back end 
UPDATE_TIME_IN_MILLISECONDS = 300

# E=hc/lambda , and hc=1240 eV*nm
WAVELENGTH_TO_ENERGY_CONVERSION = 1240.0 # eV*nm

# Using Dresselhaus' convention for carbon nanotubes (CNTs), (n,m), we can create all CNTs within two (n,m) values
lowestNM = 5
highestNM = 12

## Pi Plasmon ##
# Starting pi plasmon amplitude as a ratio of the absorbance at the peak center
AMP_RATIO_PI = 0.6
# Starting peak center for the pi plasmon
PI_PLASMON_CENTER = 5.6
# Allowable variance in pi plasmon center
PI_PLASMON_CENTER_VAR = 0.6
PI_PLASMON_CENTER_MAX = (PI_PLASMON_CENTER + PI_PLASMON_CENTER_VAR)
PI_PLASMON_CENTER_MIN = (PI_PLASMON_CENTER - PI_PLASMON_CENTER_VAR)
# Full width at half maximum for pi plasmon
PI_PLASMON_FWHM = 0.6
PI_PLASMON_FWHM_MAX = 5.0
PI_PLASMON_FWHM_MIN = 0.1

## Graphite Lorentzian ##
# Starting graphite amplitude as a ratio of the absorbance at the peak center
AMP_RATIO_GRAPHITE = 0.5
# Starting peak center for the graphite
GRAPHITE_CENTER = 4.1
# Allowable variance in graphite
GRAPHITE_CENTER_VAR = 0.4
GRAPHITE_CENTER_MAX = (GRAPHITE_CENTER + GRAPHITE_CENTER_VAR)
GRAPHITE_CENTER_MIN = (GRAPHITE_CENTER - GRAPHITE_CENTER_VAR)
# Full width at half maximum for graphite
GRAPHITE_FWHM = 0.6
GRAPHITE_FWHM_MAX = 5.0
GRAPHITE_FWHM_MIN = 0.5

# Extinction coefficients for types of amorphous carbon
# Source: "Analyzing Absorption Backgrounds in Single-Walled Carbon Nanotube Spectra" (Anton V. Naumov, Saunab Ghosh, Dmitri A. Tsyboulski, Sergei M. Bachilo, and R. Bruce Weisman)
alpha_N134 = 0.155 # L/mg
alpha_aCB = 0.082 # L/mg
b_N134 = 0.0030 # nm^-1
b_aCB = 0.00155 # nm^-1

# Metallic Background coefficients
alpha_METAL = 0.048 # L/mg
b_METAL = 0.00155 # nm^-1

# Energy Transitions given by Reference 2: "An atlas of carbon nanotube optical transitions"
# Equation used for energy transitions can be found in the supplementary information
# Below is the list of anisotropy prefactors and fermi velocity renormalization constants used in the equation

# Supporting Info:
beta = -0.620 # eV * nm
alpha = [1.532, 1.474, 1.504, 1.556, 1.560, 1.576, 1.588, 1.593, 1.596, 1.608] # eV * nm
eta   = [0.148, 0.097, 0.068, 0.058, 0.058, 0.061, 0.050, 0.052, 0.058, 0.058] # eV * nm^2
gamma = [-0.056,-0.009,-0.002,0.014, 0.016, 0.009, 0.000, 0.000, 0.011, 0.004] # eV * nm^2
# Main Paper:
beta_mainpaper = -0.173 # eV * nm^2
eta_mainpaper = [0.142, 0.097, 0.068, 0.058, 0.058, 0.058, 0.047, 0.052, 0.047, 0.054] # eV * nm^2
vF_mainpaper =  [1.229, 1.152, 1.176, 1.221, 1.226, 1.236, 1.241, 1.244, 1.248, 1.256] # 10^6 m s^-1


## Useful functions for getting data from arbitrary data files
## Since most data files contain contiguous sets of data in columns, these functions extract the data and disregard the other information
def retrieve_XY(file_path):
	# XY data is read in from a file in text format
	file_data = open(file_path).readlines()
	
	# The list of strings (lines in the file) is made into a list of lists while splitting by whitespace and removing commas
	file_data = map(lambda line: line.rstrip('\n').replace(',',' ').split(), file_data)

	# Remove empty lists, make into numpy array
	xy_array = np.array(filter(None, file_data))

	# Each line is searched to make sure that all items in the line are a number
	where_num = np.array(map(is_number, xy_array))
	
	# The data is searched for the longest contiguous chain of numbers
	contig = contiguous_regions(where_num)
	try:
		# Data lengths
		data_lengths = contig[:,1] - contig[:,0]
		# All maximums in contiguous data
		maxs = np.amax(data_lengths)
		longest_contig_idx = np.where(data_lengths == maxs)
	except ValueError:
		print 'Problem finding contiguous data'
		return np.array([])
	# Starting and stopping indices of the contiguous data are stored
	ss = contig[longest_contig_idx]

	# The file data with this longest contiguous chain of numbers
	# Float must be cast to each value in the lists of the contiguous data and cast to a numpy array 
	longest_data_chains = np.array([[map(float, n) for n in xy_array[i[0]:i[1]]] for i in ss])

	# If there are multiple sets of data of the same length, they are added in columns
	column_stacked_data_chain = np.hstack(longest_data_chains)

	return column_stacked_data_chain

#http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
def contiguous_regions(condition):
	"""Finds contiguous True regions of the boolean array "condition". Returns
	a 2D array where the first column is the start index of the region and the
	second column is the end index."""

	# Find the indicies of changes in "condition"
	d = np.diff(condition)
	idx, = d.nonzero() 

	# We need to start things after the change in "condition". Therefore, 
	# we'll shift the index by 1 to the right.
	idx += 1

	if condition[0]:
		# If the start of condition is True prepend a 0
		idx = np.r_[0, idx]

	if condition[-1]:
		# If the end of condition is True, append the length of the array
		idx = np.r_[idx, condition.size] # Edit

	# Reshape the result into two columns
	idx.shape = (-1,2)
	return idx

def is_number(s):
	try:
		np.float64(s)
		return True
	except ValueError:
		return False


## Useful functions for spectral processing

heaviside = lambda x: 0.5 * (np.sign(x) + 1)

# https://gist.github.com/RyanHope/2321077
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
	The Savitzky-Golay filter removes high frequency noise from data.
	It has the advantage of preserving the original shape and
	features of the signal better than other types of filtering
	approaches, such as moving averages techniques.
	Parameters
	----------
	y : array_like, shape (N,)
		the values of the time history of the signal.
	window_size : int
		the length of the window. Must be an odd integer number.
	order : int
		the order of the polynomial used in the filtering.
		Must be less then `window_size` - 1.
	deriv: int
		the order of the derivative to compute (default = 0 means only smoothing)
	Returns
	-------
	ys : ndarray, shape (N)
		the smoothed signal (or it's n-th derivative).
	Notes
	-----
	The Savitzky-Golay is a type of low-pass filter, particularly
	suited for smoothing noisy data. The main idea behind this
	approach is to make for each point a least-square fit with a
	polynomial of high order over a odd-sized window centered at
	the point.
	Examples
	--------
	t = np.linspace(-4, 4, 500)
	y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
	ysg = savitzky_golay(y, window_size=31, order=4)
	import matplotlib.pyplot as plt
	plt.plot(t, y, label='Noisy signal')
	plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
	plt.plot(t, ysg, 'r', label='Filtered signal')
	plt.legend()
	plt.show()
	References
	----------
	.. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
	   Data by Simplified Least Squares Procedures. Analytical
	   Chemistry, 1964, 36 (8), pp 1627-1639.
	.. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
	   W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
	   Cambridge University Press ISBN-13: 9780521880688
	"""
	
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError, msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')


# Optical transition from i (valence) to j (conduction)
# Metallics may have a splitting between high and low energies (trigonal warping effect)
class Transition:
	def __init__(self, swcnt, i, j=None, k = 0):
		self.swcnt = swcnt
		self.i = i
		self.k = k
		if j is not None:
			self.j = j
		else:
			self.j = i

		# Solvents affect optical transitions (redshifting or blueshifting) due to differences in dielectric constants and refractive indexes, etc.
		# Silvera-Batista, Carlos A., Randy K. Wang, Philip Weinberg, and Kirk J. Ziegler.
		# "Solvatochromic shifts of single-walled carbon nanotubes in nonpolar microenvironments."
		# Physical Chemistry Chemical Physics 12, no. 26 (2010): 6990-6998.
		self.epsilon = self.swcnt.spectra.epsilon
		self.eta = self.swcnt.spectra.eta
		self.Dswcntsolvent = self.swcnt.spectra.Dswcntsolvent

		# Calculate the properties of the transition and store them
		self.p = self.p()
		self.k_p = self.k_p()
		self.theta_p = self.theta_p()
		self.Eii = self.Eii()
		self.fs = self.fs()
		self.FWHM = self.FWHM()
		self.a = self.a()
		self.b = self.b()
		self.delta = self.delta()
		
		self.shape = 1.0
		self.amp = self.fs
		if self.swcnt.spectra.X is not None:
			E = self.swcnt.spectra.X
		else:
			E = np.array([0])

		# Absorption profile from:
		# Liu, Kaihui, Xiaoping Hong, Sangkook Choi, Chenhao Jin, Rodrigo B. Capaz, Jihoon Kim, Shaul Aloni et al. 
		# "Systematic Determination of Absolute Absorption Cross-section of Individual Carbon Nanotubes."
		# arXiv preprint arXiv:1311.3328 (2013).
		self.line = self.fs/np.pi * \
					self.FWHM/((E-self.Eii)**2+self.FWHM**2) + \
					self.fs/(self.a*np.pi) * \
					np.convolve( (self.b*self.FWHM)/(E**2+(self.b*self.FWHM)**2), \
								 (heaviside(E-(self.Eii + self.delta))/np.sqrt(abs(E-(self.Eii + self.delta))))
								 , mode='same')
	  

		
	# Energy Transition Functions from Reference 2:
	# Optical transition index 'p'
	def p(self):
		if(self.swcnt.mod_type==0.): p = 3.*self.i
		elif(self.swcnt.mod_type==1. or self.swcnt.mod_type==2.):
			if(self.i%2.==0.): p = self.i+(self.i/2.)-1. # If i is even
			elif(self.i%2.==1.): p = self.i+int(self.i/2.) # If i is odd
		else: print("Error in electronic type")
		p = int(p)
		return p
	
	# Length of polar coordinates vector from the K point (Reference 2):
	def k_p(self):
		k_p = 2.*self.p/(3.*self.swcnt.dt)
		return k_p

	# Angle of wave vector around K point in radians
	def theta_p(self):
		theta_p = []
		if(self.swcnt.mod_type==0.):
			if(self.k==0):
				# lower energy sub band
				theta_p = self.swcnt.theta + np.pi
			if(self.k==1):
				# higher energy sub band
				theta_p = self.swcnt.theta
		elif(self.swcnt.mod_type==1.):
			theta_p = self.swcnt.theta + self.i*np.pi
		elif(self.swcnt.mod_type==2.):
			theta_p = self.swcnt.theta + (self.i+1.)*np.pi
		return theta_p
	
	# Energy Optical Transitions from Valence Band (i) to Conduction Band (i) given by Reference 2: "An atlas of carbon nanotube optical transitions"
	# Equation used for energy transitions can be found in the supplementary information of Reference 2
	def Eii_vacc(self):
		theta_p = self.theta_p
		k_p = self.k_p
		p = self.p

		# Supporting info algorithm  
		Eii = alpha[p-1]*k_p + beta*k_p*np.log10(1.5*k_p) + (k_p**2.)*(eta[p-1]+gamma[p-1]*np.cos(theta_p*3.))*np.cos(theta_p * 3.)
		return Eii

	def Eii(self):
		Eii_vacc = self.Eii_vacc()
		# Silvera-Batista, Carlos A., Randy K. Wang, Philip Weinberg, and Kirk J. Ziegler.
		# "Solvatochromic shifts of single-walled carbon nanotubes in nonpolar microenvironments."
		# Physical Chemistry Chemical Physics 12, no. 26 (2010): 6990-6998.
		Eii_shift = -self.Dswcntsolvent*self.Onsager()/(Eii_vacc**3.*self.swcnt.dt**5.)
		Eii = Eii_vacc + Eii_shift
		return Eii
		
	def Onsager(self):
		fe = 2.*(self.epsilon-1.)/(2*self.epsilon+1)
		fn = 2.*(self.eta**2-1.)/(2*self.eta**2+1)
		return fe - fn


	# Optical osciallator strength per atom for semiconducting tubes
	# Liu, Kaihui, Xiaoping Hong, Sangkook Choi, Chenhao Jin, Rodrigo B. Capaz, Jihoon Kim, Shaul Aloni et al.
	# "Systematic Determination of Absolute Absorption Cross-section of Individual Carbon Nanotubes."
	# arXiv preprint arXiv:1311.3328 (2013).
	def fs(self):
		return 45.9/((self.p + 7.5)*self.swcnt.dt)

	# Full width at half maximum
	def FWHM(self):   
		if(self.swcnt.electronic_type!=0):
			FWHM = 0.0194 * self.Eii
		else:
			FWHM = 0.0214 * self.Eii
		return FWHM
	
	def a(self):
		if(self.swcnt.electronic_type!=0):
			a = 4.673 - 0.747 * self.swcnt.dt
		else:
			a = 0.976 + 0.186 * self.swcnt.dt
		return a

	def b(self):
		if(self.swcnt.electronic_type!=0):
			b = 0.97 + 0.256 * self.swcnt.dt
		else:
			b = 3.065 - 0.257 * self.swcnt.dt
		return b

	def delta(self):
		if(self.swcnt.electronic_type!=0):
			delta = 0.273 - 0.041 * self.swcnt.dt
		else:
			delta = 0.175 - 0.0147 * self.swcnt.dt
		return delta
	
	# Returns a string which gives information about the transition
	def transition_string(self):
		NM_Eii_string = "SWCNT" + self.swcnt.NM() + '_E' + str(self.i) + str(self.j)
		if(self.swcnt.electronic_type!=0):
			transition_string = NM_Eii_string
		if(self.swcnt.electronic_type==0):
			if(self.k==0):
				transition_string = NM_Eii_string + "_low"
			if(self.k==1):
				transition_string = NM_Eii_string + "_high"
		else:
			transition_string = NM_Eii_string
		return transition_string

# Single-walled carbon nanotube object
# each SWCNT is defined by it's characteristic (n,m) chirality
class SWCNT:
	def __init__(self, n, m, spectra):
		n = float(n)
		m = float(m)
		self.n = n
		self.m = m
		if spectra is not None:
			self.spectra = spectra
		else:
			self.spectra = Spectra('default', np.zeros(0), np.zeros(0))
		# Electronic type test:
		# 0 - metallic
		# 1 - semiconducting type 1 (S1)
		# 2 - semiconducting type 2 (S2)
		self.electronic_type = np.mod(2.*n + m,3.)
		
		# Alternative nomenclature:
		# 0 - metallic
		# 1 - mod type 1 <=> semiconducting type 2 (MOD1 <=> S2)
		# 2 - mod type 2 <=> semiconducting type 1 (MOD2 <=> S1)
		self.mod_type = np.mod(n - m,3.)
		
		# Basic Nanotube Properties from Reference 1:
		# Chiral vector length in angstroms (1.421 is the C-C distance in angstroms)
		self.Ch = np.sqrt(3.0)*1.421*np.sqrt(n**2. + m**2. + n*m)
		# CNT diameter in angstroms  (/10. --> nanometers)
		self.dt = self.Ch/np.pi/10.
		# Chiral angle in radians
		self.theta = np.arctan(np.sqrt(3.)*m/(m + 2.*n))
		# Consider S11, S22, S33, S44, S55, S66, S77 and M11 M22, M33, with high and low tranisitions for metals
		upper_ij_metal = 3
		upper_ij_sc = 7
		if(self.electronic_type==0):
			if(self.n==self.m):
				self.allowed_transitions = [Transition(self, i) for i in range(1,upper_ij_metal+1)]
			else:
				self.allowed_transitions = [Transition(self, i, i, k) for i in range(1,upper_ij_metal+1) for k in range(0,2)]
		else:
			self.allowed_transitions = [Transition(self, i) for i in range(1,upper_ij_sc+1)]

	# A model line for each nanotube is the sum of each of the transitions that can occur for the nanotube
	@property
	def line(self):
		arrays = np.array([transition.line for transition in self.allowed_transitions])
		return np.sum(arrays, axis=0)
		
	# Other useful functions for debugging and documentation
	def print_electronic_type(self):
		if(self.mod_type==0): return "Metallic"
		elif(self.mod_type==1 or self.mod_type==2): return "Semiconducting"
		else: return "Error in n,m indices"
		
	# Returns the Dresselhaus nomenclature "(n,m)" for each nanotube
	def strNM(self):
		string_tube = "(" + str(np.int(self.n)) + "," +  str(np.int(self.m)) + ")"
		return string_tube
	
	# For paramters, we cannot store "(,)" symbols
	# So this is just a string of "NM" - such as "66" for "(6,6)"
	def NM(self):
		NM = str(np.int(self.n)).rstrip(".") + str(np.int(self.m)).rstrip(".")
		return NM

# Psuedo Voigts can create lorentz and Guassian functions or a convolution of both
# It is useful for background creation
def pseudoVoigt(x, amp, center, width, shapeFactor):
	LorentzPortion = (width**2./((x-center)**2.+width**2.))
	GaussianPortion = np.e**(-((x-center)**2./(2.*width**2.)))
	try:
		Voigt = amp*(shapeFactor*LorentzPortion+(1.-shapeFactor)*GaussianPortion)
	except ZeroDivisionError:
		width += 0.001
		pseudoVoigt(x, amp, center, width, shapeFactor)
	return Voigt

# Function to create all of the tubes and store them in a list
def initialize_SWCNTs(lowestNM, highestNM, spectra):
	SWCNTs=[]
	# Create a list of all tube species we are interested in
	for n in range(0,highestNM+1):
		for m in range(0,highestNM+1):
			if(n<lowestNM and m<lowestNM): break
			elif(n<m): break
			else: SWCNTs.append(SWCNT(n, m, spectra))
	return SWCNTs
	
# Spectra object which holds the X and Y data and the sample name
class Spectra(QtCore.QObject):
	
	# Signal/slot setup uses signals to tell the program when to update the GUI 
	update_signal = QtCore.pyqtSignal(QtCore.QObject)
	done_signal = QtCore.pyqtSignal()
	
	def __init__(self, spectra_name, X, Y):
		QtCore.QObject.__init__(self)
		self.spectra_name = spectra_name
		self.X = X
		# The given spectra is smoothed using the savitsky golay smoothing algorithm
		self.Y = savitzky_golay(y=Y,window_size=5, order=2) 
		self.model = Y*0
		self.background_model = Y*0
		self.model_without_background = Y*0
		self.step = 0

		# Solvents affect optical transitions (redshifting or blueshifting) due to differences in dielectric constants and refractive indexes, etc.
		# Silvera-Batista, Carlos A., Randy K. Wang, Philip Weinberg, and Kirk J. Ziegler.
		# "Solvatochromic shifts of single-walled carbon nanotubes in nonpolar microenvironments."
		# Physical Chemistry Chemical Physics 12, no. 26 (2010): 6990-6998.
		self.epsilon = 2.3
		self.eta = 1.33
		self.Dswcntsolvent = 0.09 # eV^4*nm^5
		
		# All the single-walled carbon nanotubes to be used in the deconvolution process
		# The list will be an array of SWCNTs from (n,m)=(lowestNM, 0) to (n,m)=(highestNM,highestNM)
		self.SWCNT_list = initialize_SWCNTs(lowestNM, highestNM, self)
		self.transition_list = [transition for swcnt in self.SWCNT_list for transition in swcnt.allowed_transitions] #if(self.in_spectrum(transition)==True)]
		# First, create our SWCNT profile matrix
		self.SWCNT_matrix = np.matrix(np.column_stack([swcnt.line for swcnt in self.SWCNT_list]))
		self.swcnts_soln = np.ones(self.SWCNT_matrix.shape[1], dtype=np.float64)
		
		self.params = lmfit.Parameters()
		
		self.species_percentage_dictionary = {}
		self.species_percentage_error_dictionary = {}
		self.metallic_percentage = 0.0
		self.mean_diameter = 0.0
		graphite_amp = AMP_RATIO_GRAPHITE * np.interp(GRAPHITE_CENTER, self.X, self.Y)
		PP_amp = AMP_RATIO_PI * np.interp(PI_PLASMON_CENTER, self.X, self.Y)

		self.bkg_soln = np.array([0.0, 0.0, 0.0, 0.0, \
								  graphite_amp, GRAPHITE_CENTER, GRAPHITE_FWHM, \
								  PP_amp, PI_PLASMON_CENTER, PI_PLASMON_FWHM])
		self.bkg_soln_bounds = np.array([(0.0,None), (0.0, None), (0.0, None), (0.0, None), \
										 (0.0, None), (GRAPHITE_CENTER_MIN, GRAPHITE_CENTER_MAX), (GRAPHITE_FWHM_MIN, GRAPHITE_FWHM_MAX), \
										 (0.0, None), (PI_PLASMON_CENTER_MIN, PI_PLASMON_CENTER_MAX), (PI_PLASMON_FWHM_MIN, PI_PLASMON_FWHM_MAX)])		


		self.sample_params = np.array([self.epsilon, self.eta, self.Dswcntsolvent])
		self.sample_params_bounds = np.array([(1.,5.),(1.,5.),(0.0,0.1)])
		
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.send_update_GUI)
		self.timer.start(UPDATE_TIME_IN_MILLISECONDS)

	def calc_species_norm_amps_dictionary(self):
		species_norm_amp_dict = {}
		for i, swcnt in enumerate(self.SWCNT_list):
			species_norm_amp_dict[swcnt] = self.swcnts_soln[i]/swcnt.allowed_transitions[0].fs
		return species_norm_amp_dict

	def calc_species_norm_amps_error_dictionary(self):
		species_norm_amp_error_dict = {}
		for swcnt in self.SWCNT_list:
			for transition in swcnt.allowed_transitions:
				try:
					Eii_amp_error_value = self.params[transition.transition_string() + '_amp'].stderr
				except KeyError:
					Eii_amp_error_value = -1.0
				try:
					species_norm_amp_error_dict[swcnt] = Eii_amp_error_value/transition.fs
				except TypeError:
					species_norm_amp_error_dict[swcnt] = -1.0
				break
		return species_norm_amp_error_dict
	
	def calc_species_percentage_dictionary(self):
		species_percentage_dict = {}
		species_norm_dict = self.calc_species_norm_amps_dictionary()
		# First get the sum of all of the amplitudes, while normalizing them using the optical oscillator strength
		sum_Eiis_norm_by_fs = sum(species_norm_dict.values())
		for swcnt in self.SWCNT_list:
			try:
				species_percentage_dict[swcnt] = 100.*species_norm_dict[swcnt] / sum_Eiis_norm_by_fs
			except (ZeroDivisionError, KeyError):
				species_percentage_dict[swcnt] = 0.0
		return species_percentage_dict

	def calc_species_percentage_error_dictionary(self):
		species_percentage_error_dict = {}
		species_norm_error_dict = self.calc_species_norm_amps_error_dictionary()
		# First get the sum of all of the amplitudes, while normalizing them using the optical oscillator strength
		sum_Eiis_norm_by_fs = sum(species_norm_error_dict.values())
		for swcnt in self.SWCNT_list:
			try:
				species_percentage_error_dict[swcnt] = 100.*species_norm_error_dict[swcnt] / sum_Eiis_norm_by_fs
			except (ZeroDivisionError, KeyError, TypeError):
				species_percentage_error_dict[swcnt] = -1.0
		return species_percentage_error_dict

	def calc_metallic_percentage(self):
		metallic_percentage = 0.0
		for swcnt in self.SWCNT_list:
			if(swcnt.electronic_type==0.0):
				metallic_percentage += self.species_percentage_dictionary[swcnt]
		return metallic_percentage
	
	def calc_mean_diameter(self):
		mean_diameter = 0.0
		for swcnt in self.SWCNT_list:
			mean_diameter += self.species_percentage_dictionary[swcnt]/100. * swcnt.dt
		return mean_diameter
	

	@QtCore.pyqtSlot() 
	def deconvolute(self):
		self.state = 0
		x, f, d = scipy.optimize.fmin_l_bfgs_b(func = self.residual, x0=self.bkg_soln, bounds=self.bkg_soln_bounds, approx_grad = True, factr = 10)
		
		self.done_signal.emit()
		
	def residual(self, bkg_params):
		self.step += 1
		# Initialize
		residual_array = np.zeros(len(self.X))
		temp_background_model = np.zeros(len(self.X))
		temp_model = np.zeros(len(self.X))
		temp_model_without_background = np.zeros(len(self.X))

		# Calculate the background first and add SWCNT voigt profiles on later
		self.bkg_soln = bkg_params
		
		aCBConc = bkg_params[0]
		N134Conc = bkg_params[1]
		aCBy0 = bkg_params[2]
		N134y0 = bkg_params[3]
		GLamp = bkg_params[4]
		GLcenter = bkg_params[5]
		GLFWHM = bkg_params[6]
		PPamp = bkg_params[7]
		PPcenter = bkg_params[8]
		PPFWHM = bkg_params[9]

		aCB = aCBConc * alpha_aCB * (aCBy0 + np.exp(-b_aCB * (WAVELENGTH_TO_ENERGY_CONVERSION/self.X)))
		N134 = N134Conc * alpha_N134 * (N134y0 + np.exp(-b_N134 * (WAVELENGTH_TO_ENERGY_CONVERSION/self.X)))
		GL = pseudoVoigt(self.X, GLamp, GLcenter, GLFWHM, 1)
		PP = pseudoVoigt(self.X, PPamp, PPcenter, PPFWHM, 1)
		
		temp_background_model = aCB + N134 + GL + PP

		# In the first state, solve the fitting for just the background parameters and nanotube amplitudes simultaneously 
		if(self.state==0):
			# bkg_sub, if the background is fit properly, will contain just the absorption profile from van hove singularity transitions from nanotubes
			bkg_sub = self.Y - temp_background_model
			
			# Solve the system with swcnts:
			self.swcnts_soln, residual = scipy.optimize.nnls(self.SWCNT_matrix, bkg_sub)

			# Change the amplitudes for each SWCNT in the SWCNT object
			for i, swcnt in enumerate(self.SWCNT_list):
				swcnt.line = self.swcnts_soln[i] * np.array(self.SWCNT_matrix[:,i])

			temp_model_without_background = np.inner(self.SWCNT_matrix, self.swcnts_soln)
				
		temp_model = temp_model_without_background + temp_background_model

		# The background model should not exceed the given spectrum at any point
		# Therefore, if it does, apply a large residual to that point
		for x_index in range(0, len(self.X)):
			if(temp_background_model[x_index] > self.Y[x_index]):
				residual_array[x_index] = -999.*(temp_model[x_index] - self.Y[x_index])
			else:
				residual_array[x_index] = temp_model[x_index] - self.Y[x_index]

		# Update and store all of the values in the Spectra object
		self.model_without_background = temp_model_without_background
		self.background_model = temp_background_model
		self.model = temp_model
		self.species_percentage_dictionary = self.calc_species_percentage_dictionary()
		self.species_percentage_error_dictionary = self.calc_species_percentage_error_dictionary()
		self.metallic_percentage = self.calc_metallic_percentage()
		self.mean_diameter = self.calc_mean_diameter()
		return np.sum(residual_array**2)
	
	def send_update_GUI(self):
		self.update_signal.emit(self)
		return

# Each tab which holds a spectra gets its own object
class Spectra_Tab(QtGui.QTabWidget):
	start_comp = QtCore.pyqtSignal()
	kill_thread = QtCore.pyqtSignal()
	def __init__(self, parent, temp_spectra):
		self.parent = parent
		QtGui.QTabWidget.__init__(self, parent)
		self.temp_spectra = temp_spectra
		self.top_layer_grid = QtGui.QGridLayout(self)
		
		self.canvas_frame = QtGui.QFrame(self)
		self.canvas_frame.setFrameShape(QtGui.QFrame.StyledPanel)
		self.results_frame = QtGui.QFrame(self)
		self.results_frame.setFrameShape(QtGui.QFrame.StyledPanel)
		
		self.top_layer_grid.addWidget(self.canvas_frame)
		self.top_layer_grid.addWidget(self.results_frame)

		self.canvas_grid = QtGui.QGridLayout(self.canvas_frame)
	
		self.top_left_frame = QtGui.QFrame(self.canvas_frame)
		self.top_left_frame.setFrameShape(QtGui.QFrame.StyledPanel)
		self.canvas_grid.addWidget(self.top_left_frame)
	
		self.bottom_canvas_frame = QtGui.QFrame(self.canvas_frame)
		self.bottom_canvas_frame.setFrameShape(QtGui.QFrame.StyledPanel)
		self.canvas_grid.addWidget(self.bottom_canvas_frame)
	
		vertical_splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
		vertical_splitter.addWidget(self.top_left_frame)
		vertical_splitter.addWidget(self.bottom_canvas_frame)
		self.canvas_grid.addWidget(vertical_splitter)

		self.results_grid = QtGui.QGridLayout(self.results_frame)
		self.treeWidget = QtGui.QTreeWidget(self.results_frame)
		self.treeWidget.setFocusPolicy(QtCore.Qt.WheelFocus)
		self.treeWidget.setAutoFillBackground(True)
		self.treeWidget.setAlternatingRowColors(True)
		self.treeWidget.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
		self.treeWidget.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
		self.treeWidget.setHorizontalScrollMode(QtGui.QAbstractItemView.ScrollPerItem)
		self.treeWidget.setAutoExpandDelay(-1)
		self.treeWidget.setHeaderLabels(["(n,m)/Property","%, [value]"])
		self.other_properties = QtGui.QTreeWidgetItem(self.treeWidget, ["Properties"])
		self.nm_species = QtGui.QTreeWidgetItem(self.treeWidget, ["(n,m)"])
		self.semiconducting = QtGui.QTreeWidgetItem(self.other_properties, ["Semiconducting %"])
		self.metallic = QtGui.QTreeWidgetItem(self.other_properties, ["Metallic %"])
		self.avg_diameter = QtGui.QTreeWidgetItem(self.other_properties, ["Average Diameter"])
		self.step_in_tree = QtGui.QTreeWidgetItem(self.other_properties, ["Iteration #"])
		self.dict_of_nm_tree = {}
		for swcnt in temp_spectra.SWCNT_list:
			self.dict_of_nm_tree[swcnt] = QtGui.QTreeWidgetItem(self.nm_species, [swcnt.strNM()])
		self.results_grid.addWidget(self.treeWidget)

		graph_results_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
		graph_results_splitter.addWidget(self.canvas_frame)
		graph_results_splitter.addWidget(self.results_frame)
		self.top_layer_grid.addWidget(graph_results_splitter)

		policy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,QtGui.QSizePolicy.Preferred)
		policy.setHorizontalStretch(8)
		self.canvas_frame.setSizePolicy(policy)

		# Make figure for original line, background line, and total fit line
		self.top_left_fig = matplotlib.figure.Figure()
		self.top_left_plot = self.top_left_fig.add_subplot(111)
		self.top_left_plot.set_ylabel('Absorbance [a.u.]')
		self.top_left_plot.set_xlabel('Photon Energy [eV]')
		self.top_left_plot.set_title('Total Absorbance Fit')
		init_values = np.zeros(len(self.temp_spectra.X))
		self.top_left_line, = self.top_left_plot.plot(self.temp_spectra.X, self.temp_spectra.Y, 'r-')
		self.top_left_background_line, self.top_left_total_fit_line, = self.top_left_plot.plot(self.temp_spectra.X, init_values, 'k-', self.temp_spectra.X, init_values, 'b-', animated=True)
		self.top_left_canvas = FigureCanvas(self.top_left_fig)
		plotLayout = QtGui.QVBoxLayout()
		plotLayout.addWidget(self.top_left_canvas)
		self.top_left_frame.setLayout(plotLayout)	   
		self.top_left_canvas.show()
		self.top_left_canvas.draw()
		self.top_left_canvas_BBox = self.top_left_plot.figure.canvas.copy_from_bbox(self.top_left_plot.bbox)
		self.ax1 = self.top_left_plot.figure.axes[0]
		self.ax1.set_xlim(self.temp_spectra.X.min(), self.temp_spectra.X.max())
		self.ax1.set_ylim(0, self.temp_spectra.Y.max() + .05*self.temp_spectra.Y.max())
		self.top_left_plot_old_size = self.top_left_plot.bbox.width, self.top_left_plot.bbox.height

		# Make bottom figure
		self.bottom_fig = matplotlib.figure.Figure()
		self.bottom_plot = self.bottom_fig.add_subplot(111)
		self.bottom_plot.set_ylabel('Absorbance [a.u.]')
		self.bottom_plot.set_xlabel('Photon Energy [eV]')
		self.bottom_plot.set_title('Background Subtracted Fit')
		self.bottom_line_original_without_background, = self.bottom_plot.plot(self.temp_spectra.X, self.temp_spectra.Y, 'r-', linewidth=3, animated=True)
		self.bottom_line, = self.bottom_plot.plot(self.temp_spectra.X, init_values, 'b-', linewidth=3, animated=True)
		self.swcnt_line_dict = {}
		for swcnt in temp_spectra.SWCNT_list:
			self.swcnt_line_dict[swcnt], = self.bottom_plot.plot(self.temp_spectra.X, swcnt.line, animated=True)
		self.bottom_canvas = FigureCanvas(self.bottom_fig)
		bottomplotLayout = QtGui.QVBoxLayout()
		bottomplotLayout.addWidget(self.bottom_canvas)
		self.bottom_canvas_frame.setLayout(bottomplotLayout)
		self.bottom_canvas.show()
		self.bottom_canvas.draw()
		self.bottom_canvas_BBox = self.bottom_plot.figure.canvas.copy_from_bbox(self.bottom_plot.bbox)
		self.bottom_ax1 = self.bottom_plot.figure.axes[0]
		self.bottom_ax1.set_xlim(self.temp_spectra.X.min(), self.temp_spectra.X.max())
		self.bottom_ax1.set_ylim(0, self.temp_spectra.Y.max() + .05*self.temp_spectra.Y.max())
		self.bottom_plot_old_size = self.bottom_plot.bbox.width, self.bottom_plot.bbox.height
		
		# Make Thread associated with the tab
		thread = QtCore.QThread(parent=self)
		self.worker = self.temp_spectra
		self.worker.moveToThread(thread)
		self.worker.update_signal.connect(self.update_GUI)
		self.worker.done_signal.connect(self.closeEvent)
		self.start_comp.connect(self.worker.deconvolute)
		self.kill_thread.connect(thread.quit)
		thread.start()
		
	@QtCore.pyqtSlot(Spectra)
	def update_GUI(self, tmp_spectra):
		# change the GUI to reflect changes made to Spectra
		# Get the first background of the plots to blits lines to
		if(tmp_spectra.step==1):
			self.top_left_canvas_BBox = self.top_left_plot.figure.canvas.copy_from_bbox(self.top_left_plot.bbox)
			self.bottom_canvas_BBox = self.bottom_plot.figure.canvas.copy_from_bbox(self.bottom_plot.bbox)
		# If the size of the box changes, get that background instead
		top_left_plot_current_size = self.top_left_plot.bbox.width, self.top_left_plot.bbox.height
		bottom_plot_current_size = self.bottom_plot.bbox.width, self.bottom_plot.bbox.height
		if( self.top_left_plot_old_size != top_left_plot_current_size or self.bottom_plot_old_size != bottom_plot_current_size):
			self.top_left_plot_old_size = top_left_plot_current_size
			self.top_left_plot.clear()
			self.top_left_canvas.draw()
			self.top_left_canvas_BBox = self.top_left_plot.figure.canvas.copy_from_bbox(self.top_left_plot.bbox)
			self.top_left_plot.set_ylabel('Absorbance [a.u.]')
			self.top_left_plot.set_xlabel('Photon Energy [eV]')
			self.top_left_plot.set_title('Total Absorbance Fit')
			self.bottom_plot_old_size = bottom_plot_current_size
			self.bottom_plot.clear()
			self.bottom_canvas.draw()
			self.bottom_canvas_BBox = self.bottom_plot.figure.canvas.copy_from_bbox(self.bottom_plot.bbox)
			self.bottom_plot.set_ylabel('Absorbance [a.u.]')
			self.bottom_plot.set_xlabel('Photon Energy [eV]')
			self.bottom_plot.set_title('Background Subtracted Fit')
		
		# Write to the Top Left Plot with original data, background data, and total fit
		self.top_left_background_line.set_ydata(tmp_spectra.background_model)
		self.top_left_total_fit_line.set_ydata(tmp_spectra.model)
		self.top_left_plot.figure.canvas.restore_region(self.top_left_canvas_BBox)
		if( tmp_spectra.background_model.max() > tmp_spectra.Y.max()):
			self.ax1.set_ylim(0, 1.05*tmp_spectra.background_model.max())
		elif(tmp_spectra.model.max() > tmp_spectra.Y.max()):
			self.ax1.set_ylim(0, 1.05*tmp_spectra.model.max())
		else:
			self.ax1.set_ylim(0, 1.05*tmp_spectra.Y.max())
		self.top_left_plot.draw_artist(self.top_left_line)
		self.top_left_plot.draw_artist(self.top_left_background_line)
		self.top_left_plot.draw_artist(self.top_left_total_fit_line)
		self.top_left_plot.figure.canvas.blit(self.top_left_plot.bbox)

		# Write to the Bottom Plot with each nanotube peak
		self.bottom_line_original_without_background.set_ydata(tmp_spectra.Y-tmp_spectra.background_model)
		self.bottom_line.set_ydata(tmp_spectra.model_without_background)
		try:
			for swcnt in self.dict_of_nm_tree:
				self.swcnt_line_dict[swcnt].set_ydata(swcnt.line)
				self.swcnt_line_dict[swcnt].set_linewidth(1)
			current_swcnt = None
			for swcnt in self.dict_of_nm_tree:
				if(self.dict_of_nm_tree[swcnt] == self.treeWidget.currentItem()):
					current_swcnt = swcnt
					break
			self.swcnt_line_dict[current_swcnt].set_linewidth(4)
		except KeyError:
			pass
		self.bottom_plot.figure.canvas.restore_region(self.bottom_canvas_BBox)
		if( np.amax(tmp_spectra.Y-tmp_spectra.background_model) > np.amax(tmp_spectra.model_without_background) ):
			self.bottom_ax1.set_ylim(0, 1.05*np.amax(tmp_spectra.Y-tmp_spectra.background_model))
		if( np.amax(tmp_spectra.model_without_background) < 0.05):
			self.bottom_ax1.set_ylim(0, 0.05)
		else:
			self.bottom_ax1.set_ylim(0, 1.05*np.amax(tmp_spectra.model_without_background))
		self.bottom_plot.draw_artist(self.bottom_line_original_without_background)
		self.bottom_plot.draw_artist(self.bottom_line)
		for swcnt in tmp_spectra.SWCNT_list:
			self.bottom_plot.draw_artist(self.swcnt_line_dict[swcnt])
		self.bottom_plot.figure.canvas.blit(self.bottom_plot.bbox)

		# Show percentage of species on the side bar
		try:
			percent_dict = tmp_spectra.species_percentage_dictionary
			percent_error_dict = tmp_spectra.species_percentage_error_dictionary
			for swcnt in tmp_spectra.SWCNT_list:
				self.dict_of_nm_tree[swcnt].setText(1, str(round(percent_dict[swcnt], 0)).rstrip('0') + ' % +-' + str(round(percent_error_dict[swcnt], 1)))
			self.semiconducting.setText(1, str(round(100.-tmp_spectra.metallic_percentage, 0)).rstrip('0') + ' %')
			self.metallic.setText(1, str(round(tmp_spectra.metallic_percentage, 0)).rstrip('0') + ' %')
			self.avg_diameter.setText(1, str(round(tmp_spectra.mean_diameter,2)) + ' nm')
			self.step_in_tree.setText(1, str(tmp_spectra.step))
		except KeyError:
			pass


	def output_results(self):
		print "Making Excel Workbook..."

		date_time = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S)")
		name = str(self.temp_spectra.spectra_name)
		book = xlsxwriter.Workbook(name + ' -- ' + date_time +'_OA_Results.xlsx')
		OA_sheet_name = "Optical Absorption Data"
		Results_sheet_name = "Results"
		Other_params_name = "Other Parameters"
		OA_sheet = book.add_worksheet(OA_sheet_name)
		Results_sheet = book.add_worksheet(Results_sheet_name)
		Other_params_sheet = book.add_worksheet(Other_params_name)

		# Write x, y data for main and all species
		OA_sheet.write('A1', "Energy (eV)")
		OA_sheet.write_column('A2', self.temp_spectra.X)
		OA_sheet.write('B1', name )
		OA_sheet.write_column('B2', self.temp_spectra.Y)
		OA_sheet.write('C1', "Model")
		OA_sheet.write_column('C2', self.temp_spectra.model)
		OA_sheet.write('D1', "Background")
		OA_sheet.write_column('D2', self.temp_spectra.background_model)
		for i, swcnt in enumerate(self.temp_spectra.SWCNT_list):
			OA_sheet.write(0, 3+i, swcnt.strNM())
			OA_sheet.write_column(1, 3+i, swcnt.line)

		Results_sheet.write('A1', "(n,m)")
		Results_sheet.write('B1', "%")
		quant_dict = self.temp_spectra.species_percentage_dictionary.iteritems()
		for i, (swcnt, amount) in enumerate(sorted(quant_dict)):
			Results_sheet.write(i+1, 0, swcnt.strNM())
			Results_sheet.write(i+1, 1, round(amount,1))
		Results_sheet.write('D1', "Semiconducting %")
		Results_sheet.write('D2', "Metallic %")
		Results_sheet.write('E1', round(100-self.temp_spectra.calc_metallic_percentage(),1))
		Results_sheet.write('E2', round(self.temp_spectra.calc_metallic_percentage(),1))


		Other_params_sheet.write("A1", "(n,m)")
		Other_params_sheet.write_column("A2", [swcnt.strNM() for swcnt in self.temp_spectra.SWCNT_list])
		Other_params_sheet.write("B1", "SWCNT Solution Vector")
		Other_params_sheet.write_column("B2", self.temp_spectra.swcnts_soln)

		book.close()
		print "Excel Workbook Made."
			


	def start_computation(self):
		self.start_comp.emit()
		return
		
	def closeEvent(self):
		print 'done with processing'
		self.kill_thread.emit()
	
class MainWindow(QtGui.QMainWindow):
	def __init__(self, parent = None):
		self.spectra_list = []
		self.tab_list = []
		QtGui.QMainWindow.__init__(self)
		self.setWindowTitle("Spectro")
		screen_height = app.desktop().screenGeometry().height() 
		screen_width = app.desktop().screenGeometry().width()
		self.resize(int(screen_width*APP_SCREEN_RATIO), int(screen_height*APP_SCREEN_RATIO))
	
		self.setTabShape(QtGui.QTabWidget.Rounded)
		self.centralwidget = QtGui.QWidget(self)
		self.top_level_layout = QtGui.QGridLayout(self.centralwidget)

		self.tabWidget = QtGui.QTabWidget(self.centralwidget)
		self.top_level_layout.addWidget(self.tabWidget, 1, 0, 25, 25)
		
		open_spectra_button = QtGui.QPushButton("Open Spectra")
		self.top_level_layout.addWidget(open_spectra_button, 0, 0)
		QtCore.QObject.connect(open_spectra_button, QtCore.SIGNAL("clicked()"), self.open_spectra)
		
		process_spectra_button = QtGui.QPushButton("Process Spectra")
		self.top_level_layout.addWidget(process_spectra_button, 0, 1)
		QtCore.QObject.connect(process_spectra_button, QtCore.SIGNAL("clicked()"), self.process_spectra)

		save_results_button = QtGui.QPushButton("Save Results")
		self.top_level_layout.addWidget(save_results_button, 0, 2)
		QtCore.QObject.connect(save_results_button, QtCore.SIGNAL("clicked()"), self.output_results)

		self.setCentralWidget(self.centralwidget)
		self.centralwidget.setLayout(self.top_level_layout)
		
	def open_spectra(self):
		fileNameList = QtGui.QFileDialog.getOpenFileNames(caption="Select Files for Processing")
		for file_name in fileNameList:
			# file_name is form ~ "C:/Users/you/someData.asc", so split it after the last "/" and before the "." in ".asc" or ".xls"
			spectra_name = file_name.split('\\')[-1].split('.')[0]
			xy = retrieve_XY(file_name)
			X = WAVELENGTH_TO_ENERGY_CONVERSION/xy[:,0]
			Y = xy[:,1]
			self.spectra_list.append(Spectra(spectra_name, X, Y))
			self.tab_list.append(Spectra_Tab(self.tabWidget, self.spectra_list[-1]))
			self.tabWidget.addTab(self.tab_list[-1], spectra_name)
		return
	
	def process_spectra(self):
		for tab in self.tab_list:
			tab.start_computation()
		return
	
	def output_results(self):
		for tab in self.tab_list:
			tab.output_results()
		return 
	
if __name__ == "__main__":
	app = QtGui.QApplication([])
	win = MainWindow()
	win.show()
	sys.exit(app.exec_())

