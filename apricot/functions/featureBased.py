# featureBased.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 
	
import math
import numpy

from .base import BaseSelection
from ..optimizers import LazyGreedy
from ..optimizers import ApproximateLazyGreedy

from tqdm import tqdm

from numba import njit
from numba import prange

@njit('float64[:](float64[:])', fastmath=True)
def sigmoid(X):
	return X / (1. + X)

dtypes = 'void(float64[:,:], float64[:], float64[:], int64[:])'
sparse_dtypes = 'void(float64[:], int32[:], int32[:], float64[:],' \
	'float64[:], float64[:], int64[:])'
sieve_dtypes = 'void(float64[:,:], int64, float64[:,:], int64[:,:],' \
	'float64[:,:], float64[:], float64[:], int64[:], int64[:])' 
sieve_sparse_dtypes = 'void(float64[:], int32[:], int32[:], int64,' \
	'float64[:,:], int64[:,:], float64[:,:], float64[:], float64[:],' \
	'int64[:], int64[:])'

def calculate_gains(func, dtypes, parallel, fastmath, cache):
	@njit(dtypes, parallel=parallel, fastmath=fastmath, cache=cache)
	def calculate_gains_(X, gains, current_values, idxs):
		for i in prange(idxs.shape[0]):
			idx = idxs[i] 
			gains[i] = func(current_values + X[idx]).sum()

	return calculate_gains_


def calculate_gains_sparse(func, dtypes, parallel, fastmath, cache):
	@njit(dtypes, parallel=parallel, fastmath=fastmath, cache=cache)
	def calculate_gains_sparse_(X_data, X_indices, X_indptr, gains, 
		current_values, current_concave_values, idxs):
		for i in prange(idxs.shape[0]):
			idx = idxs[i]
			start = X_indptr[idx]
			end = X_indptr[idx+1]

			for j in range(start, end):
				k = X_indices[j]
				gains[i] += (func(X_data[j] + current_values[k]) 
					- current_concave_values[k])

	return calculate_gains_sparse_


def calculate_gains_sieve(func, dtypes, parallel, fastmath, cache):
	@njit(dtypes, parallel=parallel, fastmath=fastmath, cache=cache)
	def calculate_gains_sieve_(X, k, current_values, selections, gains, 
		total_gains, max_values, n_selected, idxs):
		n = X.shape[0]
		d = max_values.shape[0]

		for j in prange(d):
			for i in range(n):
				if n_selected[j] == k:
					break

				idx = idxs[i]
				threshold = (max_values[j] / 2. - total_gains[j]) / (k - n_selected[j])
				gain = func(current_values[j] + X[i]).sum() - total_gains[j]

				if gain > threshold:
					current_values[j] += X[i]
					total_gains[j] += gain

					selections[j, n_selected[j]] = idx
					gains[j, n_selected[j]] = gain
					n_selected[j] += 1

	return calculate_gains_sieve_


def calculate_gains_sieve_sparse(func, dtypes, parallel, fastmath, cache):
	@njit(dtypes, parallel=parallel, fastmath=fastmath, cache=cache)
	def calculate_gains_sieve_sparse_(X_data, X_indices, X_indptr, k, 
		current_values, selections, gains, total_gains, max_values, 
		n_selected, idxs):
		d = max_values.shape[0]

		for j in prange(d):
			for i in range(idxs.shape[0]):
				if n_selected[j] == k:
					break
				
				idx = idxs[i]
				start = X_indptr[i]
				end = X_indptr[i+1]
				threshold = (max_values[j] / 2. - total_gains[j]) / (k - n_selected[j])

				gain = 0.0
				for l in range(start, end):
					m = X_indices[l]
					gain += (func(current_values[j, m] + X_data[l]) - 
						func(current_values[j, m]))

				if gain > threshold:
					for l in range(start, end):
						m = X_indices[l]
						current_values[j, m] += X_data[l]

					total_gains[j] += gain
					selections[j, n_selected[j]] = idx
					gains[j, n_selected[j]] = gain
					n_selected[j] += 1

	return calculate_gains_sieve_sparse_


class FeatureBasedSelection(BaseSelection):
	"""A selector based off a feature based submodular function.

	Feature-based functions are those that operate on the feature values of 
	examples directly, rather than on a similarity matrix (or graph) derived 
	from those features, as graph-based functions do. Because these functions 
	do not require calculating and storing a :math:`\\mathcal{O}(n^{2})` sized 
	matrix of similarities, they can easily scale to data sets with millions of 
	examples. 
	
	.. note:: 
		All values in your data must be positive for this selection to work.

	The general form of a feature-based function is:

	.. math::
		f(X) = \\sum\\limits_{d=1}^{D}\\phi\\left(\\sum\\limits_{i=1}^{N} X_{i, d}\\right)

	where :math:`f` indicates the function that uses the concave function 
	:math:`\\phi` and is operating on a subset :math:`X` that has :math:`N` 
	examples and :math:`D` dimensions. Importantly, :math:`X` is the subset 
	and not the ground set, meaning that the time it takes to evaluate this 
	function is proportional only to the size of the selected subset and not 
	the size of the full data set, like it is for graph-based functions.  

	See https://ieeexplore.ieee.org/document/6854213 for more details on
	feature based functions.

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
			'sigmoid' : X / (1 + X)

	initial_subset : list, numpy.ndarray or None
		If provided, this should be a list of indices into the data matrix
		to use as the initial subset, or a group of examples that may not be
		in the provided data should beused as the initial subset. If indices, 
		the provided array should be one-dimensional. If a group of examples,
		the data should be 2 dimensional.

	optimizer : string or optimizers.BaseOptimizer, optional
		The optimization approach to use for the selection. Default is
		'two-stage', which makes selections using the naive greedy algorithm
		initially and then switches to the lazy greedy algorithm. Must be
		one of

			'random' : randomly select elements (dummy optimizer)
			'modular' : approximate the function using its modular upper bound
			'naive' : the naive greedy algorithm
			'lazy' : the lazy (or accelerated) greedy algorithm
			'approximate-lazy' : the approximate lazy greedy algorithm
			'two-stage' : starts with naive and switches to lazy
			'stochastic' : the stochastic greedy algorithm
			'sample' : randomly take a subset and perform selection on that
			'greedi' : the GreeDi distributed algorithm
			'bidirectional' : the bidirectional greedy algorithm

		Default is 'two-stage'.

	optimizer_kwds : dict or None
		A dictionary of arguments to pass into the optimizer object. The keys
		of this dictionary should be the names of the parameters in the optimizer
		and the values in the dictionary should be the values that these
		parameters take. Default is None.

	reservoir : numpy.ndarray or None
		The reservoir to use when calculating gains in the sieve greedy
		streaming optimization algorithm in the `partial_fit` method.
		Currently only used for graph-based functions. If a numpy array
		is passed in, it will be used as the reservoir. If None is passed in,
		will use reservoir sampling to collect a reservoir. Default is None.

	max_reservoir_size : int 
		The maximum size that the reservoir can take. If a reservoir is passed
		in, this value is set to the size of that array. Default is 1000.

	n_jobs : int
		The number of threads to use when performing computation in parallel.
		Currently, this parameter is exposed but does not actually do anything.
		This will be fixed soon.

	random_state : int or RandomState or None, optional
		The random seed to use for the random selection process. Only used
		for stochastic greedy.

	verbose : bool
		Whether to print output during the selection process.

	Attributes
	----------
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

	def __init__(self, n_samples, concave_func='sqrt', initial_subset=None, 
		optimizer='two-stage', optimizer_kwds={}, reservoir=None,
		max_reservoir_size=1000, n_jobs=1, random_state=None, 
		verbose=False):
		self.concave_func_name = concave_func

		if concave_func == 'log':
			self.concave_func = numpy.log1p
		elif concave_func == 'sqrt':
			self.concave_func = numpy.sqrt
		elif concave_func == 'sigmoid':
			self.concave_func = sigmoid
		elif callable(concave_func):
			self.concave_func = concave_func
		else:
			raise KeyError("Must be one of 'log', 'sqrt', 'sigmoid', or a any other numpy, scipy, or numba-compiled function.")

		super(FeatureBasedSelection, self).__init__(n_samples=n_samples, 
			initial_subset=initial_subset, optimizer=optimizer, 
			optimizer_kwds=optimizer_kwds, reservoir=reservoir,
			max_reservoir_size=max_reservoir_size, n_jobs=n_jobs, 
			random_state=random_state, verbose=verbose) 

	def fit(self, X, y=None, sample_weight=None, sample_cost=None):
		"""Run submodular optimization to select the examples.

		This method is a wrapper for the full submodular optimization process.
		It takes in some data set (and optionally labels that are ignored
		during this process) and selects `n_samples` from it in the greedy
		manner specified by the optimizer.

		This method will return the selector object itself, not the transformed
		data set. The `transform` method will then transform a data set to the
		selected points, or alternatively one can use the ranking stored in
		the `self.ranking` attribute. The `fit_transform` method will perform
		both optimization and selection and return the selected items.

		Parameters
		----------
		X : list or numpy.ndarray, shape=(n, d)
			The data set to transform. Must be numeric.

		y : list or numpy.ndarray or None, shape=(n,), optional
			The labels to transform. If passed in this function will return
			both the data and th corresponding labels for the rows that have
			been selected.

		sample_weight : list or numpy.ndarray or None, shape=(n,), optional
			The weight of each example. Currently ignored in apricot but
			included to maintain compatibility with sklearn pipelines. 

		sample_cost : list or numpy.ndarray or None, shape=(n,), optional
			The cost of each item. If set, indicates that optimization should
			be performed with respect to a knapsack constraint.

		Returns
		-------
		self : FeatureBasedSelection
			The fit step returns this selector object.
		"""

		return super(FeatureBasedSelection, self).fit(X, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)

	def _initialize(self, X):
		super(FeatureBasedSelection, self)._initialize(X)

		if self.initial_subset is None:
			pass
		elif self.initial_subset.ndim == 2:
			self.current_values = self.initial_subset.sum(axis=0).astype('float64')
		elif self.initial_subset.ndim == 1:
			self.current_values = X[self.initial_subset].sum(axis=0).astype('float64')
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

		self.current_concave_values = self.concave_func(self.current_values)
		self.current_concave_values_sum = self.current_concave_values.sum()

		calculate_gains_ = calculate_gains_sparse if self.sparse else calculate_gains
		dtypes_ = sparse_dtypes if self.sparse else dtypes

		if self.optimizer in (LazyGreedy, ApproximateLazyGreedy):
			self.calculate_gains_ = calculate_gains_(self.concave_func, 
				dtypes_, False, True, False)
		elif self.optimizer in ('lazy', 'approimate-lazy'):
			self.calculate_gains_ = calculate_gains_(self.concave_func, 
				dtypes_, False, True, False)
		else: 
			self.calculate_gains_ = calculate_gains_(self.concave_func, 
				dtypes_, True, True, False)

		calculate_sieve_gains_ = calculate_gains_sieve_sparse if self.sparse else calculate_gains_sieve
		dtypes_ = sieve_sparse_dtypes if self.sparse else sieve_dtypes 
		self.calculate_sieve_gains_ = calculate_sieve_gains_(self.concave_func,
			dtypes_, True, True, False)

	def _calculate_gains(self, X, idxs=None):
		"""This function will return the gain that each example would give.

		This function will return the gains that each example would give if
		added to the selected set. When a matrix of examples is given, a
		vector will be returned showing the gain for each example. When
		a single element is passed in, it will return a singe value."""

		idxs = idxs if idxs is not None else self.idxs
		gains = numpy.zeros(idxs.shape[0], dtype='float64')

		if self.sparse:
			self.calculate_gains_(X.data, X.indices, X.indptr, gains, 
				self.current_values, self.current_concave_values, idxs)
		else:
			self.calculate_gains_(X, gains, self.current_values, idxs)
			gains -= self.current_concave_values_sum

		return gains

	def _calculate_sieve_gains(self, X, thresholds, idxs):
		"""This function will update the internal statistics from a stream.

		This function will update the various internal statistics that are a
		part of the sieve algorithm for streaming submodular optimization. This
		function does not directly return gains but it updates the values
		used by a streaming optimizer.
		"""

		super(FeatureBasedSelection, self)._calculate_sieve_gains(X,
			thresholds, idxs)

		if self.sparse:
			self.calculate_sieve_gains_(X.data, X.indices, X.indptr, 
				self.n_samples, self.sieve_current_values_, 
				self.sieve_selections_, self.sieve_gains_, 
				self.sieve_total_gains_, thresholds, 
				self.sieve_n_selected_, idxs)
		else:
			self.calculate_sieve_gains_(X, self.n_samples, 
				self.sieve_current_values_, self.sieve_selections_, 
				self.sieve_gains_, self.sieve_total_gains_, thresholds, 
				self.sieve_n_selected_, idxs)

	def _select_next(self, X, gain, idx):
		"""This function will add the given item to the selected set."""

		if self.sparse:
			self.current_values += X.toarray()[0]
		else:
			self.current_values += X

		self.current_concave_values = self.concave_func(self.current_values)
		self.current_concave_values_sum = self.current_concave_values.sum()

		super(FeatureBasedSelection, self)._select_next(
			X, gain, idx)

