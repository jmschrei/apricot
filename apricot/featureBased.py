# featureBased.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 

"""
This file contains code that implements feature based submodular selection
algorithms.
"""

import numpy

from .base import SubmodularSelection

from tqdm import tqdm
from numba import njit, int64, float32, int8, jit, prange

from joblib import Parallel
from joblib import delayed

dtypes = 'int64(float64[:,:], float64[:], float64[:], float64[:], int8[:])'

@njit(dtypes, nogil=True, parallel=True)
def select_sqrt_next(X, gains, current_values, current_concave_values, mask):
	for idx in prange(X.shape[0]):
		if mask[idx] == 1:
			continue

		a = numpy.sqrt(current_values + X[idx])
		gains[idx] = (a - current_concave_values).sum()

	return numpy.argmax(gains)

@njit(dtypes, nogil=True, parallel=True)
def select_log_next(X, gains, current_values, current_concave_values, mask):
	for idx in prange(X.shape[0]):
		if mask[idx] == 1:
			continue

		a = numpy.log(current_values + X[idx] + 1)
		gains[idx] = (a - current_concave_values).sum()

	return numpy.argmax(gains)

@njit(dtypes, nogil=True, parallel=True)
def select_inv_next(X, gains, current_values, current_concave_values, mask):
	for idx in prange(X.shape[0]):
		if mask[idx] == 1:
			continue

		a = 1. / (1. + current_values + X[idx])
		gains[idx] = (a - current_concave_values).sum()

	return numpy.argmax(gains)

@njit(dtypes, nogil=True, parallel=True)
def select_min_next(X, gains, current_values, current_concave_values, mask):
	for idx in prange(X.shape[0]):
		if mask[idx] == 1:
			continue

		a = numpy.fmin(current_values, X[idx])
		gains[idx] = (a - current_concave_values).sum()

	return numpy.argmax(gains)

def select_custom_next(X, gains, current_values, current_concave_values, mask, 
	concave_func):
	best_gain = 0.
	best_idx = -1

	for idx in range(X.shape[0]):
		if mask[idx] == 1:
			continue

		a = concave_func(current_values + X[idx])
		gains[idx] = (a - current_concave_values).sum()

		if gains[idx] > best_gain:
			best_gain = gain
			best_idx = idx

	return best_idx

class FeatureBasedSelection(SubmodularSelection):
	"""A feature based submodular selection algorithm.

	NOTE: All values in your data must be positive for this selection to work.

	This function will use a feature based submodular selection algorithm to
	identify a representative subset of the data. The feature based functions
	use the values of the features themselves, rather than a transformation of
	the values, in order to select a diverse subset. The goal of this approach
	is to find a representative subset that sees each ~feature~ at a certain
	saturation across the selected points, rather than trying to uniformly
	sample the space.

	This implementation uses the lazy greedy algorithm so that multiple passes
	over the whole data set are not required each time a sample is selected. The
	benefit of this effect is not typically seen until many (over 100) passes are
	seen of the data set.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

	concave_func : str or callable
		The type of concave function to apply to the feature values. You can
		pass in your own function to apply. Otherwise must be one of the
		following:

			'log' : log(1 + X)
			'sqrt' : sqrt(X)
			'min' : min(X, 1)
			'inverse' : 1 / (1 + X)

	verbose : bool
		Whether to print output during the selection process.

	Attributes
	----------
	pq : PriorityQueue
		The priority queue used to implement the lazy greedy algorithm.

	n_samples : int
		The number of samples to select.

	concave_func : callable
		A concave function for transforming feature values, often referred to as
		phi in the literature.

	ranking : numpy.array int
		The selected samples in the order of their gain with the first number in
		the ranking corresponding to the index of the first sample that was
		selected by the greedy procedure.
	"""

	def __init__(self, n_samples, concave_func='sqrt', n_greedy_samples=3, 
		verbose=False):
		self.concave_func_name = concave_func
		self.n_greedy_samples = n_greedy_samples

		if concave_func == 'log':
			self.concave_func = lambda X: numpy.log(X + 1)
		elif concave_func == 'sqrt':
			self.concave_func = lambda X: numpy.sqrt(X)
		elif concave_func == 'min':
			self.concave_func = lambda X: numpy.fmin(X, numpy.ones_like(X))
		elif concave_func == 'inverse':
			self.concave_func = lambda X: 1. / (1. + X)
		elif callable(concave_func):
			self.concave_func = concave_func
		else:
			raise KeyError("Must be one of 'log', 'sqrt', 'min', 'inverse', or a custom function.")

		super(FeatureBasedSelection, self).__init__(n_samples, verbose)

	def fit(self, X, y=None):
		"""Perform selection and return the subset of the data set.

		This method will take in a full data set and return the selected subset
		according to the feature based function. The data will be returned in
		the order that it was selected, with the first row corresponding to
		the best first selection, the second row corresponding to the second
		best selection, etc.

		Parameters
		----------
		X : list or numpy.ndarray, shape=(n, d)
			The data set to transform. Must be numeric.

		y : list or numpy.ndarray, shape=(n,), optional
			The labels to transform. If passed in this function will return
			both the data and th corresponding labels for the rows that have
			been selected.

		Returns
		-------
		self : FeatureBasedSelection
			The fit step returns itself.
		"""

		if not isinstance(X, (list, numpy.ndarray)):
			raise ValueError("X must be either a list of lists or a 2D numpy array.")
		if isinstance(X, numpy.ndarray) and len(X.shape) != 2:
			raise ValueError("X must have exactly two dimensions.")
 		if numpy.min(X) < 0.0:
			raise ValueError("X cannot contain negative values.")

		if self.verbose == True:
			pbar = tqdm(total=self.n_samples)

		X = numpy.array(X, dtype='float64')
		n, d = X.shape

		mask = numpy.zeros(X.shape[0], dtype='int8')
		ranking = []

		current_values = numpy.zeros(d, dtype='float64')
		current_concave_values = numpy.zeros(d, dtype='float64')
		
		for i in range(self.n_greedy_samples):
			gains = numpy.zeros(n, dtype='float64')

			if self.concave_func_name == 'sqrt':
				best_idx = select_sqrt_next(X, gains, current_values, 
					current_concave_values, mask)
			elif self.concave_func_name == 'log':
				best_idx = select_log_next(X, gains, current_values, 
					current_concave_values, mask)
			elif self.concave_func_name == 'inverse':
				best_idx = select_inv_next(X, gains, current_values, 
					current_concave_values, mask)
			elif self.concave_func_name == 'min':
				best_idx = select_min_next(X, gains, current_values, 
					current_concave_values, mask)
			else:
				best_idx = select_custom_next(X, gains, current_values, 
					current_concave_values, mask, self.concave_func)			

			ranking.append(best_idx)
			mask[best_idx] = True
			current_values += X[best_idx]
			current_concave_values = self.concave_func(current_values)

			if self.verbose == True:
				pbar.update(1)

		for idx, gain in enumerate(gains):
			if mask[idx] != 1:
				self.pq.add(idx, -gain)

		for i in range(self.n_greedy_samples, self.n_samples):
			best_gain = 0.
			best_idx = None
			
			while True:
				prev_gain, idx = self.pq.pop()
				prev_gain = -prev_gain
				
				if best_gain >= prev_gain:
					self.pq.add(idx, -prev_gain)
					self.pq.remove(best_idx)
					break
				
				a = self.concave_func(current_values + X[idx])
				gain = (a - current_concave_values).sum()
				
				self.pq.add(idx, -gain)
				
				if gain > best_gain:
					best_gain = gain
					best_idx = idx

			ranking.append(best_idx)
			mask[best_idx] = True
			current_values += X[best_idx]
			current_concave_values = self.concave_func(current_values)

			if self.verbose == True:
				pbar.update(1)

		if self.verbose == True:
			pbar.close()

		self.ranking = numpy.array(ranking, dtype='int32')
		return self

"""
[ 60562 478758  61159  56653  60561 478759  60262  57263 478760  60861  56344  60862 478492  56652
  59086  50618  57566 478494  61453  56959 478493  56651  59681   9619  59970  59387  60560 478495
  56957  56958  57870 478217  56031  59085 478218  54064  61750  59084  56343 478219  58785  56650
  54063  61452 477931  56956  59384  59385 477932  60261  57262  58174 477933  61160  58786  59083
    293  60559  14528  61158  59680  56030  50617  60860 477641  53722  59082  59386 477642  58481
  57260  57258  55383  60563  55713  59383 477643  52703  44486  56637  47277  60859  59081  57259
  57563  56342 477350  55056  60858  50269  59382  59969  63827   9630  62047  63828  53045  60265
  57867 478761  56955  59381  53387  46882 478762  60558  57565 479027  54399  60264  54401  64428
  58171 479028  60564  63528  50966  63234  62936 479029  59679  56341  57257  61751  49916  59080
 479293  52015  59967 496832  54730  60259  52357  62045 479030  58172  59673  59678  61454  61156
  64728  58482 479294  58170  49915  57560  56649  61455 479295  59380  51318  59674  55712  54402
  59974  52014  54733 479296  62636  51667  59087  54400  54398  54062  55055 479564  68863  62044
  49206 479297  59088  62041  64727  57869  64729  63826  58175  61449  52356  56345  58173 479830
  61746  49565  60266  54729  62343  53719  46881  59675  14546  59388  56954 479565  64129  53385
  64128  56029  54734  50268  59973  63233  65026  54061  53721  60263  51317  46482  55051  62935
  60260  53046 479566  53379  61749  60863  53716    308  49207  46483  64127  65024  54060  57562
  59079  65325  48832 479831  58478  59975  57561  61752  61157 480105  59972  55382  62040  68862
  57866  65937 479567  62337  58782  48831  58781  60267  65920  66560  63825  57864 480106  58169
  57868  53384  65919  50967  63527  49564  64125  54397   9634  62931  46481  58784  62340  53386
  70095 473342 479832  59089  62932  52699  57256  49914  61155  51666  59389 479833  61448  44875
  53723  61450  59962  46086  66894  51315  59968  65025  61747  50616  47674  65023  55378  53382
  56346  54056   9645  52704  66227  54731  56032  57559 479834  62639  65323  56961 480677    354
 554406  64427  52018  58479  58168  56328 480108  51670  58477  63824  49205  69496  69180  60857
  53715  68543  59672  53042  65622 480950  53040  57865  48451  50964  64126 480107  61154  52702
  66895  58783  56953  58480  47276 480391  65918  53383  59078  62046  60557  60258  65324  49563
  66226  53378  56655  57863  47673  64425  60864  49913  64726 476165  55707  45685  64426  69181
  57564  54728  50615 480387  50965  59379  58788  53044 490812  62930  59683  55054  52361  47672
  66892 480392  48450  58177  53381  63823  61745 480390  59684  59963  67560  56962  61162  56648
  67226  59676  60255  52355  54732  65621  69179 480388  52698  58780  61753  52358 480397  63524
  60256  66893  60565  44485  67558  57264  63230  66558 480670  59971  52013 480669  59961  63227
  51314  59390  66559  50266 480678  67557  56340  56025  58787  62637  59077  67559  53720  64424
  50267 480672  68219  63229  62634  54393 480389  68861  56654  50963  47279  58476  55047  54057
  62338  52696  62635  65022  47275 480673  61161  67225  60856  65620  61456  62342   9666  60855
  50961  67227  49566 496831  60555  58167  61451 480674  51320  53041  47671 480671  58789  49208
  67224  52700  53718  55050  59682  56647  65917  53380 480945  58779  54394  69178  67888    353
 481468  52354  53729 481736  58790  64124  51669  56028 481737  54055]
 """
