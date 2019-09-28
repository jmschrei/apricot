# base.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 

"""
This file contains code that implements the core of the submodular selection
algorithms.
"""

try:
	import cupy
except:
	import numpy as cupy

import numpy

from .optimizers import Optimizer
from .optimizers import NaiveGreedy
from .optimizers import LazyGreedy
from .optimizers import TwoStageGreedy
from .optimizers import BidirectionalGreedy

from .utils import PriorityQueue

from scipy.sparse import csr_matrix

class SubmodularSelection(object):
	"""The base selection object.

	This object defines the structures that all submodular selection algorithms
	should follow. All algorithms will have the same public methods and the
	same attributes.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

	n_greedy_samples : int
		The number of samples to perform the naive greedy algorithm on
		before switching to the lazy greedy algorithm. The lazy greedy
		algorithm is faster once features begin to saturate, but is slower
		in the initial few selections. This is, in part, because the naive
		greedy algorithm is parallelized whereas the lazy greedy
		algorithm currently is not.

	initial_subset : list, numpy.ndarray or None
		If provided, this should be a list of indices into the data matrix
		to use as the initial subset, or a group of examples that may not be
		in the provided data should beused as the initial subset. If indices, 
		the provided array should be one-dimensional. If a group of examples,
		the data should be 2 dimensional.

	verbose : bool
		Whether to print output during the selection process.

	Attributes
	----------
	pq : PriorityQueue
		The priority queue used to implement the lazy greedy algorithm.

	n_samples : int
		The number of samples to select.

	ranking : numpy.array int
		The selected samples in the order of their gain with the first number in
		the ranking corresponding to the index of the first sample that was
		selected by the greedy procedure.

	gains : numpy.array float
		The gain of each sample in the returned set when it was added to the
		growing subset. The first number corresponds to the gain of the first
		added sample, the second corresponds to the gain of the second added
		sample, and so forth.
	"""

	def __init__(self, n_samples, n_greedy_samples=1, initial_subset=None, 
		optimizer='two-stage', verbose=False):
		if type(n_samples) != int:
			raise ValueError("n_samples must be a positive integer.")
		if n_samples < 1:
			raise ValueError("n_samples must be a positive integer.")

		if type(n_greedy_samples) != int:
			raise ValueError("n_greedy_samples must be a positive integer")
		if n_greedy_samples < 1:
			raise ValueError("n_greedy_samples must be a positive integer.")
		if n_greedy_samples > n_samples:
			raise ValueError("n_samples must be larger than n_greedy_samples")

		if not isinstance(initial_subset, (list, numpy.ndarray)) and initial_subset is not None: 
			raise ValueError("initial_subset must be a list, numpy array, or None")
		if isinstance(initial_subset, (list, numpy.ndarray)):
			initial_subset = numpy.array(initial_subset)

		if not isinstance(optimizer, Optimizer):
			if optimizer not in ('naive', 'lazy', 'two-stage', 'bidirectional'):
				raise ValueError("Optimizer must be a string or an optimizer object.")

		if verbose not in (True, False):
			raise ValueError("verbosity must be True or False")

		self.n_samples = n_samples
		self.n_greedy_samples = n_greedy_samples
		self.optimizer = optimizer
		self.verbose = verbose
		self.ranking = None
		self.gains = None
		self.sparse = None
		self.cupy = None
		self.initial_subset = initial_subset

	def fit(self, X, y=None):
		"""Fit a ranking to the data set of the top n_sample elements.

		This method will take in the data set (and optionally a label) set
		and fit a ranking to it. This runs the greedy select procedure n_samples
		times, selecting at each iteration the next best element. The ranking
		is then stored for future use.

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
		self : SubmodularSelection
		"""

		allowed_dtypes = list, numpy.ndarray, csr_matrix, cupy.ndarray

		if not isinstance(X, allowed_dtypes):
			raise ValueError("X must be either a list of lists, a 2D numpy " \
				"array, or a scipy.sparse.csr_matrix.")
		if isinstance(X, numpy.ndarray) and len(X.shape) != 2:
			raise ValueError("X must have exactly two dimensions.")
		if numpy.min(X) < 0.0 and numpy.max(X) > 0.:
			raise ValueError("X cannot contain negative values or must be entirely "\
				"negative values.")
		if self.n_samples > X.shape[0]:
			raise ValueError("Cannot select more examples than the number in" \
				" the data set.")

		self._initialize(X)

		if not self.sparse and not self.cupy:
			X = X.astype('float64')

		optimizers = {
			'naive' : NaiveGreedy(self, self.verbose),
			'lazy' : LazyGreedy(self, self.verbose),
			'two-stage' : TwoStageGreedy(self, self.n_greedy_samples, self.verbose),
			'bidirectional' : BidirectionalGreedy(self, self.verbose)
		}

		optimizer = optimizers[self.optimizer]
		optimizer.select(X, self.n_samples)

		if self.verbose == True:
			self.pbar.close()

		self.ranking = numpy.array(self.ranking)
		return self

	def transform(self, X, y=None):
		"""Transform the data set by selecting the top n_samples samples.

		This method will use the fit ranking to transform the data set by
		returning only the top n_samples elements as defined by the
		ranking. The data will be returned in the order that it was selected
		during the selection process.

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
		X_subset : numpy.ndarray, shape=(n_samples, d)
			A subset of the data such that n_samples < n and n_samples is the
			integer provided at initialization.

		y_subset : numpy.ndarray, shape=(n_samples,)
			The labels that match with the indices of the samples if y is
			passed in. Only returned if passed in.
		"""

		if y is not None:
			return X[self.ranking], y[self.ranking]
		return X[self.ranking]

	def fit_transform(self, X, y=None):
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
		X_subset : numpy.ndarray, shape=(n_samples, d)
			A subset of the data such that n_samples < n and n_samples is the
			integer provided at initialization.

		y_subset : numpy.ndarray, shape=(n_samples,)
			The labels that match with the indices of the samples if y is
			passed in. Only returned if passed in.
		"""

		return self.fit(X, y).transform(X, y)

	def _initialize(self, X):
		self.sparse = isinstance(X, csr_matrix)
		self.cupy = isinstance(X, cupy.ndarray) and not isinstance(X, numpy.ndarray)
		self.ranking = []
		self.gains = []

		if self.cupy:
			self.current_values = cupy.zeros(X.shape[1], dtype='float64')
			self.current_concave_values = cupy.zeros(X.shape[1], dtype='float64')
			self.mask = cupy.zeros(X.shape[0], dtype='int8')
		else:
			self.current_values = numpy.zeros(X.shape[1], dtype='float64')
			self.current_concave_values = numpy.zeros(X.shape[1], dtype='float64')
			self.mask = numpy.zeros(X.shape[0], dtype='int8')


		if self.initial_subset is not None:
			if self.initial_subset.ndim == 1:
				if self.initial_subset.dtype == bool:
					self.initial_subset = numpy.where(self.initial_subset == 1)[0]
				
				if len(self.initial_subset) + self.n_samples > X.shape[0]:
					raise ValueError("When using a mask for the initial subset" \
						" must selected fewer than the size of the subset minus" \
						" the initial subset size, i.e., n_samples < X.shape[0] -"\
						" initial_subset.shape[0].")

				if self.initial_subset.max() > X.shape[0]:
					raise ValueError("When passing in an integer mask for the initial subset"\
						" the maximum value cannot exceed the size of the data set.")
				elif self.initial_subset.min() < 0:
					raise ValueError("When passing in an integer mask for the initial subset"\
						" the minimum value cannot be negative.")
				
				self.mask[self.initial_subset] = 1

	def _calculate_gains(self, X):
		raise NotImplementedError

	def _select_next(self, X, gain, idx):
		self.ranking.append(idx)
		self.gains.append(gain)
		self.mask[idx] = True
