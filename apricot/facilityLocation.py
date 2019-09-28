# facilityLocation.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code implements facility location functions.
"""

try:
	import cupy
except:
	import numpy as cupy

import numpy

from .base import SubmodularSelection

from tqdm import tqdm

from numba import njit
from numba import prange

from scipy.sparse import csr_matrix

dtypes = 'int64(float64[:,:], float64[:], float64[:], int8[:])'
sdtypes = 'int64(float64[:], int32[:], int32[:], float64[:], float64[:], int8[:])'

@njit(dtypes, nogil=True, parallel=True)
def select_next(X, gains, current_values, mask):
	for idx in prange(X.shape[0]):
		if mask[idx] == 1:
			continue

		gains[idx] = numpy.maximum(X[idx], current_values).sum()

	return numpy.argmax(gains)

@njit(sdtypes, nogil=True, parallel=True)
def select_next_sparse(X_data, X_indices, X_indptr, gains, current_values, mask):
	for idx in range(X_indptr.shape[0] - 1):
		if mask[idx] == 1:
			continue

		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for i in range(start, end):
			j = X_indices[i]
			gains[idx] += max(X_data[i], current_values[j])

	return numpy.argmax(gains)

def select_next_cupy(X, gains, current_values, mask):
	gains[:] = cupy.sum(cupy.maximum(X, current_values), axis=1)
	gains[:] = gains * (1 - mask)
	return int(cupy.argmax(gains))

class FacilityLocationSelection(SubmodularSelection):
	"""A facility location submodular selection algorithm.

	NOTE: All ~pairwise~ values in your data must be positive for this 
	selection to work.

	This function uses a facility location based submodular selection algorithm
	to identify a representative subset of the data. This feature based function
	works on pairwise relationships between each of the samples. This can be
	the correlation, a dot product, or any other such function where a higher
	value corresponds to a higher similarity and a lower value corresponds to
	a lower similarity.

	This implementation allows users to pass in either their own symmetric
	square matrix of similarity values, or a data matrix as normal and a function
	that calculates these pairwise values.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

	pairwise_func : str or callable
		The method for converting a data matrix into a square symmetric matrix
		of pairwise similarities. If a string, can be any of the following:

			'euclidean' : The negative euclidean distance
			'corr' : The squared correlation matrix
			'cosine' : The normalized dot product of the matrix
			'precomputed' : User passes in a NxN matrix of distances themselves

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
	n_samples : int
		The number of samples to select.

	pairwise_func : callable
		A function that takes in a data matrix and converts it to a square
		symmetric matrix.

	ranking : numpy.array int
		The selected samples in the order of their gain.

	gains : numpy.array float
		The gain of each sample in the returned set when it was added to the
		growing subset. The first number corresponds to the gain of the first
		added sample, the second corresponds to the gain of the second added
		sample, and so forth.
	"""

	def __init__(self, n_samples=10, pairwise_func='euclidean', n_greedy_samples=1, 
		initial_subset=None, optimizer='two-stage', verbose=False):
		self.pairwise_func_name = pairwise_func
		
		norm = lambda x: numpy.sqrt((x*x).sum(axis=1)).reshape(x.shape[0], 1)
		norm2 = lambda x: (x*x).sum(axis=1).reshape(x.shape[0], 1)

		if pairwise_func == 'corr':
			self.pairwise_func = lambda X: numpy.corrcoef(X, rowvar=True) ** 2.
		elif pairwise_func == 'cosine':
			self.pairwise_func = lambda X: numpy.abs(numpy.dot(X, X.T) / (norm(X).dot(norm(X).T)))
		elif pairwise_func == 'euclidean':
			self.pairwise_func = lambda X: (-2 * numpy.dot(X, X.T) + norm2(X)).T + norm2(X)
		elif pairwise_func == 'precomputed':
			self.pairwise_func = pairwise_func
		elif callable(pairwise_func):
			self.pairwise_func = pairwise_func
		else:
			raise KeyError("Must be one of 'euclidean', 'corr', 'cosine', 'precomputed'" \
				" or a custom function.")

		super(FacilityLocationSelection, self).__init__(n_samples=n_samples, 
			n_greedy_samples=n_greedy_samples, initial_subset=initial_subset, 
			optimizer=optimizer, verbose=verbose)

	def fit(self, X, y=None):
		"""Perform selection and return the subset of the data set.

		This method will take in a full data set and return the selected subset
		according to the facility location function. The data will be returned in
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
		self : FacilityLocationSelection
			The fit step returns itself.
		"""

		f = self.pairwise_func

		if isinstance(X, csr_matrix) and f != "precomputed":
			raise ValueError("Must passed in a precomputed sparse " \
				"similarity  matrix or a dense feature matrix.")
		if f == 'precomputed' and X.shape[0] != X.shape[1]:
			raise ValueError("Precomputed similarity matrices " \
				"must be square and symmetric.")

		if self.verbose == True:
			self.pbar = tqdm(total=self.n_samples)

		if self.pairwise_func == 'precomputed':
			X_pairwise = X
		else:
			X = numpy.array(X, dtype='float64')
			X_pairwise = self.pairwise_func(X)

			if self.pairwise_func_name == 'euclidean':
				X_pairwise = numpy.ones_like(X_pairwise) * X_pairwise.max() - X_pairwise

		return super(FacilityLocationSelection, self).fit(X_pairwise, y)

	def _initialize(self, X_pairwise):
		super(FacilityLocationSelection, self)._initialize(X_pairwise)

		if self.initial_subset is None:
			return
		elif self.initial_subset.ndim == 2:
			raise ValueError("When using facility location, the initial subset"\
				" must be a one dimensional array of indices.")
		elif self.initial_subset.ndim == 1:
			if not self.sparse:
				for i in self.initial_subset:
					self.current_values = numpy.maximum(X_pairwise[i],
						self.current_values).astype('float64')
			else:
				for i in self.initial_subset:
					self.current_values = numpy.maximum(
						X_pairwise[i].toarray()[0], self.current_values).astype('float64')
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

	def _calculate_gains(self, X_pairwise):
		if len(X_pairwise.shape) == 1:
			if not self.sparse:
				gain = numpy.maximum(X_pairwise, 
					self.current_values).sum()
			else:
				gain = numpy.sum(numpy.maximum(X_pairwise.data, 
					self.current_values[X.indices]))

			gain -= self.current_values.sum()
			return gain

		else:
			if self.cupy:
				gains = cupy.zeros(X_pairwise.shape[0], dtype='float64')
			else:
				gains = numpy.zeros(X_pairwise.shape[0], dtype='float64')

			if self.cupy:
				select_next_cupy(X_pairwise, gains, self.current_values,
					self.mask)

			elif self.sparse:
				select_next_sparse(X_pairwise.data,
					X_pairwise.indices, X_pairwise.indptr, gains,
					self.current_values, self.mask)
			else:
				select_next(X_pairwise, gains, self.current_values,
					self.mask)

			gains -= self.current_values.sum()
			return gains

	def _select_next(self, X_pairwise, gain, idx):
		"""This function will add the given item to the selected set."""

		if self.cupy:
			self.current_values = cupy.maximum(X_pairwise, 
				self.current_values)
		elif self.sparse:
			self.current_values = numpy.maximum(
				X_pairwise.toarray()[0], self.current_values)
		else:
			self.current_values = numpy.maximum(X_pairwise, 
				self.current_values)

		super(FacilityLocationSelection, self)._select_next(
			X_pairwise, gain, idx)
