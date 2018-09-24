# facilityLocation.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code implements facility location functions.
"""

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

		a = numpy.maximum(X[idx], current_values)
		gains[idx] = (a - current_values).sum()

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

			if X_data[i] > current_values[j]:
				gains[idx] += X_data[i] - current_values[j]

	return numpy.argmax(gains)


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

	Attributes
	----------
	n_samples : int
		The number of samples to select.

	pairwise_func : callable
		A function that takes in a data matrix and converts it to a square
		symmetric matrix.

	ranking : numpy.array int
		The selected samples in the order of their gain.
	"""

	def __init__(self, n_samples=10, pairwise_func='euclidean', n_greedy_samples=1, 
		verbose=False):
		self.pairwise_func_name = pairwise_func
		
		norm = lambda x: numpy.sqrt((x*x).sum(axis=1)).reshape(x.shape[0], 1)
		norm2 = lambda x: (x*x).sum(axis=1).reshape(x.shape[0], 1)

		if pairwise_func == 'corr':
			self.pairwise_func = lambda X: numpy.corrcoef(X, rowvar=True) ** 2.
		elif pairwise_func == 'cosine':
			self.pairwise_func = lambda X: numpy.abs(numpy.dot(X, X.T) / (norm(X).dot(norm(X).T)))
		elif pairwise_func == 'euclidean':
			self.pairwise_func = lambda X: -((-2 * numpy.dot(X, X.T) + norm2(X)).T + norm2(X))
		elif pairwise_func == 'precomputed':
			self.pairwise_func = pairwise_func
		elif callable(pairwise_func):
			self.pairwise_func = pairwise_func
		else:
			raise KeyError("Must be one of 'euclidean', 'corr', 'cosine', 'precomputed'" \
				" or a custom function.")

		super(FacilityLocationSelection, self).__init__(n_samples, 
			n_greedy_samples, verbose)

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
			self.pbar.update(1)

		self.sparse = isinstance(X, csr_matrix)

		if self.pairwise_func == 'precomputed':
			X_pairwise = X
		else:
			X = numpy.array(X, dtype='float64')
			X_pairwise = self.pairwise_func(X)

		return super(FacilityLocationSelection, self).fit(X_pairwise, y)

	def _greedy_select(self, X_pairwise):
		"""Select elements in a naive greedy manner."""

		for i in range(self.n_greedy_samples):
			gains = numpy.zeros(X_pairwise.shape[0], dtype='float64')
		
			if not self.sparse:
				best_idx = select_next(X_pairwise, gains, self.current_values,
					self.mask)
				self.current_values = numpy.maximum(X_pairwise[best_idx], 
					self.current_values)
			else:
				best_idx = select_next_sparse(X_pairwise.data,
					X_pairwise.indices, X_pairwise.indptr, gains,
					self.current_values, self.mask)
				self.current_values = numpy.maximum(
					X_pairwise[best_idx].toarray()[0], self.current_values)

			self.ranking.append(best_idx)
			self.mask[best_idx] = 1

			if self.verbose == True:
				self.pbar.update(1)

		return gains

	def _lazy_greedy_select(self, X_pairwise):
		"""Select elements from a dense matrix in a lazy greedy manner."""

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
				
				if not self.sparse:
					a = numpy.maximum(X_pairwise[:, idx], 
						self.current_values)

					gain = (a - self.current_values).sum()
				else:
					gain = 0.
					start = X_pairwise.indptr[idx]
					end = X_pairwise.indptr[idx+1]

					for k in range(start, end):
						j = X_pairwise.indices[k]

						if X_pairwise.data[k] > self.current_values[j]:
							gain += (X_pairwise.data[k] - 
								self.current_values[j])

				self.pq.add(idx, -gain)
				
				if gain > best_gain:
					best_gain = gain
					best_idx = idx

			self.ranking.append(best_idx)
			self.mask[best_idx] = True

			if not self.sparse:
				self.current_values = numpy.maximum(X_pairwise[best_idx],
					self.current_values)
			else:
				start = X_pairwise.indptr[best_idx]
				end = X_pairwise.indptr[best_idx+1]

				for k in range(start, end):
					j = X_pairwise.indices[k]
					self.current_values[j] = max(X_pairwise.data[k], 
						self.current_values[j])

			if self.verbose == True:
				self.pbar.update(1)

