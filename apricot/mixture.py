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
		n_greedy_samples=3, initial_subset=None, verbose=False):

		if len(submodular_functions) < 2:
			raise ValueError("Must mix at least two submodular functions.")

		self.m = len(submodular_functions)
		self.submodular_functions = submodular_functions

		if weights is None:
			self.weights = numpy.ones(self.m, dtype='float64')
		else:
			self.weights = weights

		super(FeatureBasedSelection, self).__init__(n_samples, n_greedy_samples, 
			initial_subset, verbose)

	def _greedy_select(self, X):
		"""Select elements in a naive greedy manner."""

		for i in range(self.n_greedy_samples):
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

			if self.verbose == True:
				self.pbar.update(1)

		return gains

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