# mixture.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This file contains code that implements mixtures of submodular functions.
"""

try:
	import cupy
except:
	import numpy as cupy
	
import time
import numpy

from .base import SubmodularSelection

from tqdm import tqdm

from numba import njit, jit
from numba import prange

class SubmodularMixtureSelection(SubmodularSelection):
	"""A selection approach based on a mixture of submodular functions.

	This class implements a simple mixture of submodular functions for the
	purpose of selecting a representative subset of the data. The user passes
	in a list of instantiated submodular functions and their respective weights
	to the initialization. At each iteration in the selection procedure the 
	gains from each submodular functions will be scaled by their respective 
	weight and added together.

	This class can also be used to add regularizers to the selection procedure.
	If a submodular function is mixed with another submodular function that
	acts as a regularizer, such as feature based selection mixed with a
	custom function measuring some property of the selected subset.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

	submodular_functions : list
		The list of submodular functions to mix together. The submodular
		functions should be instantiated.

	weights : list, numpy.ndarray or None
		The relative weight of each submodular function. This is the value
		that the gain from each submodular function is multiplied by before
		being added together. The default is equal weight for each function.

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

	submodular_functions : list
		A concave function for transforming feature values, often referred to as
		phi in the literature.

	weights : numpy.ndarray
		The weights of each submodular function.

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

	def __init__(self, n_samples, submodular_functions, weights=None, 
		n_greedy_samples=3, initial_subset=None, optimizer='two-stage',
		verbose=False):

		if len(submodular_functions) < 2:
			raise ValueError("Must mix at least two submodular functions.")

		self.m = len(submodular_functions)
		self.submodular_functions = submodular_functions

		if weights is None:
			self.weights = numpy.ones(self.m, dtype='float64')
		else:
			self.weights = weights

		super(SubmodularMixtureSelection, self).__init__(n_samples=n_samples, 
			n_greedy_samples=n_greedy_samples, initial_subset=initial_subset,
			optimizer=optimizer, verbose=verbose) 

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

		if self.verbose:
			self.pbar = tqdm(total=self.n_samples)

		return super(SubmodularMixtureSelection, self).fit(X, y)

	def _initialize(self, X):
		super(SubmodularMixtureSelection, self)._initialize(X)

		for function in self.submodular_functions:
			function._initialize(X)

	def _calculate_gains(self, X):
		"""This function will return the gain that each example would give.

		This function will return the gains that each example would give if
		added to the selected set. When a matrix of examples is given, a
		vector will be returned showing the gain for each example. When
		a single element is passed in, it will return a singe value."""

		if len(X.shape) == 1:
			gain = 0.0
			for function in self.submodular_functions:
				gain += function._calculate_gains(X)

			return gain

		else:
			if self.cupy:
				gains = cupy.zeros(X.shape[0], dtype='float64')
			else:
				gains = numpy.zeros(X.shape[0], dtype='float64')

			for i, function in enumerate(self.submodular_functions):
				gains += function._calculate_gains(X) * self.weights[i]

			return gains

	def _select_next(self, X, gain, idx):
		"""This function will add the given item to the selected set."""

		for function in self.submodular_functions:
			function._select_next(X, gain, idx)

		super(SubmodularMixtureSelection, self)._select_next(X, gain, idx)
