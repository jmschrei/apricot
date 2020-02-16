# featureBased.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 

"""
This file contains code that implements feature based submodular selection
algorithms.
"""

try:
	import cupy
except:
	import numpy as cupy
	
import numpy

from .base import BaseSelection

from tqdm import tqdm

from numba import njit, jit
from numba import prange

dtypes = 'void(float64[:,:], float64[:], float64[:], int64[:])'
sdtypes = 'void(float64[:], int32[:], int32[:], float64[:], float64[:], float64[:], int64[:])'

@njit(dtypes, parallel=True, fastmath=True)
def select_sqrt_next(X, gains, current_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i] 
		gains[i] = numpy.sqrt(current_values + X[idx]).sum()


@njit(dtypes, parallel=True, fastmath=True)
def select_log_next(X, gains, current_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i]
		gains[i] = numpy.log(current_values + X[idx] + 1).sum()

@njit(dtypes, parallel=True, fastmath=True)
def select_inv_next(X, gains, current_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i]
		gains[i] = ((current_values + X[idx]) / (1. 
			+ current_values + X[idx])).sum()

@njit(dtypes, parallel=True, fastmath=True)
def select_min_next(X, gains, current_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i]
		gains[i] = numpy.fmin(current_values + X[idx], 
			numpy.ones(X.shape[1])).sum()

@njit(sdtypes, parallel=True, fastmath=True)
def select_sqrt_next_sparse(X_data, X_indices, X_indptr, gains, current_values, 
	current_concave_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i]
		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for j in range(start, end):
			k = X_indices[j]
			gains[i] += numpy.sqrt(X_data[j] + current_values[k]) - current_concave_values[k]

@njit(sdtypes, parallel=True, fastmath=True)
def select_log_next_sparse(X_data, X_indices, X_indptr, gains, current_values, 
	current_concave_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i]
		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for j in range(start, end):
			k = X_indices[j]
			gains[i] += numpy.log(X_data[j] + current_values[k] + 1) - current_concave_values[k]

@njit(sdtypes, parallel=True, fastmath=True)
def select_inv_next_sparse(X_data, X_indices, X_indptr, gains, current_values, 
	current_concave_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i]
		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for j in range(start, end):
			k = X_indices[j]
			gains[i] += (current_values[k] + X_data[j]) / (1.
				+ current_values[k] + X_data[j]) - current_concave_values[k]

@njit(sdtypes, parallel=True, fastmath=True)
def select_min_next_sparse(X_data, X_indices, X_indptr, gains, current_values, 
	current_concave_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i]
		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for j in range(start, end):
			k = X_indices[j]
			gains[i] += min(X_data[j] + current_values[k], 1) - current_concave_values[k]

def select_sqrt_next_cupy(X, gains, current_values, idxs):
	gains[:] = cupy.sum(cupy.sqrt(current_values + X), axis=1)[idxs]

def select_log_next_cupy(X, gains, current_values, idxs):
	gains[:] = cupy.sum(cupy.log(current_values + X + 1), axis=1)[idxs]

def select_inv_next_cupy(X, gains, current_values, idxs):
	gains[:] = cupy.sum((current_values + X) / 
		(1. + current_values + X), axis=1)[idxs]

def select_min_next_cupy(X, gains, current_values, idxs):
	gains[:] = cupy.sum(cupy.min(current_values + X, 1), axis=1)[idxs]


class FeatureBasedSelection(BaseSelection):
	"""A selector based off a feature based submodular function.

	NOTE: All values in your data must be positive for this selection to work.

	This selector will optimize a feature based submodular function. Feature
	based functions are those that use feature values of the examples directly,
	like most machine learning methods do, rather than only using them 
	indirectly through the calculation of similarity matrices, as kernel methods 
	and facility location functions do.

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
			'min' : min(X, 1)
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

	optimizer_kwds : dict, optional
		Arguments to pass into the optimizer object upon initialization.
		Default is {}.

	n_jobs : int, optional
		The number of cores to use for processing. This value is multiplied
		by 2 when used to set the number of threads. If set to -1, use all
		cores and threads. Default is -1.

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
		optimizer='two-stage', optimizer_kwds={}, n_jobs=1, random_state=None, 
		verbose=False):
		self.concave_func_name = concave_func

		if concave_func == 'log':
			self.concave_func = lambda X: numpy.log(X + 1)
		elif concave_func == 'sqrt':
			self.concave_func = lambda X: numpy.sqrt(X)
		elif concave_func == 'min':
			self.concave_func = lambda X: numpy.fmin(X, numpy.ones_like(X))
		elif concave_func == 'sigmoid':
			self.concave_func = lambda X: X / (1. + X)
		elif callable(concave_func):
			self.concave_func = concave_func
		else:
			raise KeyError("Must be one of 'log', 'sqrt', 'min', 'sigmoid', or a custom function.")

		super(FeatureBasedSelection, self).__init__(n_samples=n_samples, 
			initial_subset=initial_subset, optimizer=optimizer, 
			optimizer_kwds=optimizer_kwds, n_jobs=n_jobs, 
			random_state=random_state, verbose=verbose) 

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

		return super(FeatureBasedSelection, self).fit(X, y)

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

	def _calculate_gains(self, X, idxs=None):
		"""This function will return the gain that each example would give.

		This function will return the gains that each example would give if
		added to the selected set. When a matrix of examples is given, a
		vector will be returned showing the gain for each example. When
		a single element is passed in, it will return a singe value."""

		concave_funcs = {
			'sqrt': select_sqrt_next,
			'sqrt_sparse': select_sqrt_next_sparse,
			'sqrt_cupy': select_sqrt_next_cupy,
			'log': select_log_next,
			'log_sparse': select_log_next_sparse,
			'log_cupy': select_log_next_cupy,
			'sigmoid': select_inv_next,
			'sigmoid_sparse': select_inv_next_sparse,
			'sigmoid_cupy': select_inv_next_cupy,
			'min': select_min_next,
			'min_sparse': select_min_next_sparse,
			'min_cupy': select_min_next_cupy
		}

		name = '{}{}'.format(self.concave_func_name,
							   '_sparse' if self.sparse else
							   '_cupy' if self.cupy else '')
		concave_func = concave_funcs.get(name, self.concave_func)

		idxs = idxs if idxs is not None else self.idxs

		if self.cupy:
			gains = cupy.zeros(idxs.shape[0], dtype='float64')
		else:
			gains = numpy.zeros(idxs.shape[0], dtype='float64')

		if concave_func is not None:
			if self.sparse:
				concave_func(X.data, X.indices, X.indptr, gains, 
					self.current_values, self.current_concave_values, idxs)
			else:
				concave_func(X, gains, self.current_values, idxs)
				gains -= self.current_concave_values_sum
		else:
			select_custom_next(X, gains, self.current_values, 
				self.mask, self.concave_func)

		return gains

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

