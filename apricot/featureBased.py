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
	
import time
import numpy

from .base import SubmodularSelection

from tqdm import tqdm

from numba import njit, jit
from numba import prange

dtypes = 'int64(float64[:,:], float64[:], float64[:], int8[:])'
sdtypes = 'int64(float64[:], int32[:], int32[:], float64[:], float64[:], float64[:], int8[:])'

@njit(dtypes, nogil=True, parallel=True)
def select_sqrt_next(X, gains, current_values, mask):
	for idx in prange(X.shape[0]):
		if mask[idx] == 1:
			continue

		gains[idx] = numpy.sqrt(current_values + X[idx]).sum()

	return numpy.argmax(gains)


@njit(dtypes, nogil=True, parallel=True)
def select_log_next(X, gains, current_values, mask):
	for idx in prange(X.shape[0]):
		if mask[idx] == 1:
			continue

		gains[idx] = numpy.log(current_values + X[idx] + 1).sum()

	return numpy.argmax(gains)

@njit(dtypes, nogil=True, parallel=True)
def select_inv_next(X, gains, current_values, mask):
	for idx in prange(X.shape[0]):
		if mask[idx] == 1:
			continue

		gains[idx] = ((current_values + X[idx]) / (1. 
			+ current_values + X[idx])).sum()

	return numpy.argmax(gains)

@njit(dtypes, nogil=True, parallel=True)
def select_min_next(X, gains, current_values, mask):
	for idx in prange(X.shape[0]):
		if mask[idx] == 1:
			continue

		gains[idx] = numpy.fmin(current_values + X[idx], 
			numpy.ones(X.shape[1])).sum()

	return numpy.argmax(gains)

@njit(sdtypes, nogil=True, parallel=True)
def select_sqrt_next_sparse(X_data, X_indices, X_indptr, gains, current_values, 
	current_concave_values, mask):
	for idx in range(X_indptr.shape[0] - 1):
		if mask[idx] == 1:
			continue

		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for i in range(start, end):
			j = X_indices[i]
			gains[idx] += numpy.sqrt(X_data[i] + current_values[j]) - current_concave_values[j]

	return numpy.argmax(gains)

@njit(sdtypes, nogil=True, parallel=True)
def select_log_next_sparse(X_data, X_indices, X_indptr, gains, current_values, 
	current_concave_values, mask):
	for idx in range(X_indptr.shape[0] - 1):
		if mask[idx] == 1:
			continue

		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for i in range(start, end):
			j = X_indices[i]
			gains[idx] += numpy.log(X_data[i] + current_values[j] + 1) - current_concave_values[j]

	return numpy.argmax(gains)

@njit(sdtypes, nogil=True, parallel=True)
def select_inv_next_sparse(X_data, X_indices, X_indptr, gains, current_values, 
	current_concave_values, mask):
	for idx in range(X_indptr.shape[0] - 1):
		if mask[idx] == 1:
			continue

		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for i in range(start, end):
			j = X_indices[i]
			gains[idx] += (current_values[j] + X_data[i]) / (1.
				+ current_values[j] + X_data[i]) - current_concave_values[j]

	return numpy.argmax(gains)

@njit(sdtypes, nogil=True, parallel=True)
def select_min_next_sparse(X_data, X_indices, X_indptr, gains, current_values, 
	current_concave_values, mask):
	for idx in range(X_indptr.shape[0] - 1):
		if mask[idx] == 1:
			continue

		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for i in range(start, end):
			j = X_indices[i]
			gains[idx] += min(X_data[i] + current_values[j], 1) - current_concave_values[j]

	return numpy.argmax(gains)

def select_sqrt_next_cupy(X, gains, current_values, mask):
	gains[:] = cupy.sum(cupy.sqrt(current_values + X), axis=1)
	gains[:] = gains * (1 - mask)
	return int(cupy.argmax(gains))

def select_log_next_cupy(X, gains, current_values, mask):
	gains[:] = cupy.sum(cupy.log(current_values + X + 1), axis=1)
	gains[:] = gains * (1 - mask)
	return int(cupy.argmax(gains))

def select_inv_next_cupy(X, gains, current_values, mask):
	gains[:] = cupy.sum((current_values + X) / 
		(1. + current_values + X), axis=1)
	gains[:] = gains * (1 - mask)
	return int(cupy.argmax(gains))

def select_min_next_cupy(X, gains, current_values, mask):
	gains[:] = cupy.sum(cupy.min(current_values + X, 1), axis=1)
	gains[:] = gains * (1 - mask)
	return int(cupy.argmax(gains))

def select_custom_next(X, gains, current_values, mask, 
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
			'inverse' : X / (1 + X)

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

	concave_func : callable
		A concave function for transforming feature values, often referred to as
		phi in the literature.

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

	def __init__(self, n_samples, concave_func='sqrt', n_greedy_samples=3, 
		initial_subset=None, verbose=False):
		self.concave_func_name = concave_func

		if concave_func == 'log':
			self.concave_func = lambda X: numpy.log(X + 1)
		elif concave_func == 'sqrt':
			self.concave_func = lambda X: numpy.sqrt(X)
		elif concave_func == 'min':
			self.concave_func = lambda X: numpy.fmin(X, numpy.ones_like(X))
		elif concave_func == 'inverse':
			self.concave_func = lambda X: X / (1. + X)
		elif callable(concave_func):
			self.concave_func = concave_func
		else:
			raise KeyError("Must be one of 'log', 'sqrt', 'min', 'inverse', or a custom function.")

		super(FeatureBasedSelection, self).__init__(n_samples, n_greedy_samples, 
			initial_subset, verbose)

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

		return super(FeatureBasedSelection, self).fit(X, y)

	def _initialize_with_subset(self, X):
		if self.initial_subset.ndim == 2:
			self.current_values = self.initial_subset.sum(axis=0).astype('float64')
		elif self.initial_subset.ndim == 1:
			self.current_values = X[self.initial_subset].sum(axis=0).astype('float64')
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

		self.current_concave_values = self.concave_func(self.current_values)

	def _greedy_select(self, X):
		"""Select elements in a naive greedy manner."""

		concave_funcs = {
			'sqrt': select_sqrt_next,
			'sqrt_sparse': select_sqrt_next_sparse,
			'sqrt_cupy': select_sqrt_next_cupy,
			'log': select_log_next,
			'log_sparse': select_log_next_sparse,
			'log_cupy': select_log_next_cupy,
			'inverse': select_inv_next,
			'inverse_sparse': select_inv_next_sparse,
			'inverse_cupy': select_inv_next_cupy,
			'min': select_min_next,
			'min_sparse': select_min_next_sparse,
			'min_cupy': select_min_next_cupy
		}

		self.concave_func_name += '_sparse' if self.sparse else ''
		self.concave_func_name += '_cupy' if self.cupy else ''

		concave_func = concave_funcs.get(self.concave_func_name,  
			None)

		self.concave_func_ = concave_func

		for i in range(self.n_greedy_samples):
			gains = self._calculate_gains(X)
			best_idx = gains.argmax()
			
			'''
			if self.cupy:
				gains = cupy.zeros(X.shape[0], dtype='float64')
			else:
				gains = numpy.zeros(X.shape[0], dtype='float64')

			if self.concave_func_name in concave_funcs:
				if self.sparse:
					best_idx = concave_func(X.data, X.indices, X.indptr, gains, 
						self.current_values, self.current_concave_values, 
						self.mask)
					self.current_values += X[best_idx].toarray()[0]
				else:
					best_idx = concave_func(X, gains, self.current_values, 
						self.mask)
					self.current_values += X[best_idx]
					gains -= self.current_concave_values.sum()
			else:
				best_idx = select_custom_next(X, gains, self.current_values, 
					self.mask, self.concave_func)
				self.current_values += X[best_idx]
				gains -= self.current_concave_values.sum()
			

			self.ranking.append(best_idx)
			self.gains.append(gains[best_idx])
			self.mask[best_idx] = True
			self.current_concave_values = self.concave_func(self.current_values)
			'''

			self._select_next(X, gains, best_idx)

			if self.verbose == True:
				self.pbar.update(1)

		return gains

	def _calculate_gains(self, X):
		if self.cupy:
			gains = cupy.zeros(X.shape[0], dtype='float64')
		else:
			gains = numpy.zeros(X.shape[0], dtype='float64')

		if self.concave_func_ is not None:
			if self.sparse:
				self.concave_func_(X.data, X.indices, X.indptr, gains, 
					self.current_values, self.current_concave_values, 
					self.mask)
			else:
				self.concave_func_(X, gains, self.current_values, 
					self.mask)
				gains -= self.current_concave_values.sum()
		else:
			select_custom_next(X, gains, self.current_values, 
				self.mask, self.concave_func)
			gains -= self.current_concave_values.sum()

		return gains

	def _select_next(self, X, gains, idx):
		if self.sparse:
			self.current_values += X[idx].toarray()[0]
		else:
			self.current_values += X[idx]

		self.ranking.append(idx)
		self.gains.append(gains[idx])
		self.mask[idx] = True
		self.current_concave_values = self.concave_func(self.current_values)

	def _lazy_greedy_select(self, X):
		"""Select elements from a dense matrix in a lazy greedy manner."""

		if self.sparse:
			X_data = X.data
			X_indptr = X.indptr
			X_indices = X.indices

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
				
				if self.sparse:
					start = X_indptr[idx] 
					end = X_indptr[idx+1]
					idxs = X_indices[start:end]
					
					gain = numpy.sum(self.concave_func(self.current_values[idxs]
						+ X_data[start:end]) - self.current_concave_values[idxs])
				else:
					gain = self.concave_func(self.current_values + X[idx]).sum()
					gain -= self.current_concave_values.sum()

				self.pq.add(idx, -gain)
				if gain > best_gain:
					best_gain = gain
					best_idx = idx

			self.ranking.append(best_idx)
			self.gains.append(best_gain)
			self.mask[best_idx] = True

			if self.sparse:
				self.current_values += X[best_idx].toarray()[0]
			else:
				self.current_values += X[best_idx]
			self.current_concave_values = self.concave_func(self.current_values)

			if self.verbose == True:
				self.pbar.update(1)

	def _batch_lazy_greedy_select(self, X):
		batch_size = 100

		gains = numpy.zeros(batch_size, dtype='float64')
		batch_idxs = numpy.zeros(batch_size, dtype='int32')

		concave_funcs = {
			'sqrt': select_sqrt_next,
			'log': select_log_next,
			'inverse': select_inv_next,
			'min': select_min_next
		}

		concave_func = select_sqrt_next_idxs

		tic = time.time()
		tictoc = 0.

		#concave_func = concave_funcs.get(self.concave_func_name, 
		#	select_custom_next)

		for i in range(self.n_greedy_samples, self.n_samples):
			batch_size = max(batch_size - int(i / (self.n_samples - self.n_greedy_samples)), 1)
			size = min(batch_size, len(self.pq.pq))


			while True:
				prev_best_gain, prev_best_idx = self.pq.pq[0]
				prev_best_gain = -prev_best_gain

				for j in range(size):
					_, idx = self.pq.pop()
					batch_idxs[j] = idx

				best_idx_ = concave_func(X, gains, self.current_values,
					self.current_concave_values, batch_idxs[:size])

				for j in range(size):
					if j != best_idx_:
						self.pq.add(batch_idxs[j], -gains[j])

				if gains[best_idx_] >= -self.pq.pq[0][0]:
					best_idx = batch_idxs[best_idx_]
					best_gain = gains[best_idx_]
					break
				else:
					self.pq.add(batch_idxs[best_idx_], -gains[best_idx_])

			self.ranking.append(best_idx)
			self.gains.append(best_gain)
			self.mask[best_idx] = True
			self.current_values += X[best_idx]
			self.current_concave_values = self.concave_func(self.current_values)

			if self.verbose == True:
				self.pbar.update(1)

		print(time.time() - tic, tictoc)
