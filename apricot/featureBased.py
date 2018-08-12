# featureBased.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 

"""
This file contains code that implements feature based submodular selection
algorithms.
"""

import numpy

from .utils import PriorityQueue

from tqdm import tqdm


class FeatureBasedSelection(object):
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

	indices : numpy.array int
		The selected samples in the order of their gain.
	"""

	def __init__(self, n_samples, concave_func='log', verbose=False):
		self.pq = PriorityQueue()

		if type(n_samples) != int:
			raise ValueError("n_samples must be a positive integer.")
		if n_samples < 1:
			raise ValueError("n_samples must be a positive integer.")

		self.n_samples = n_samples
		self.verbose = verbose
		
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
		"""

		if not isinstance(X, (list, numpy.ndarray)):
			raise ValueError("X must be either a list of lists or a 2D numpy array.")
		if isinstance(X, numpy.ndarray) and len(X.shape) != 2:
			raise ValueError("X must have exactly two dimensions.")
 
		X = numpy.array(X)

		mask = numpy.zeros(X.shape[0], dtype=bool)
		indices = []
		
		current_values = numpy.zeros(X.shape[1])
		current_concave_values = numpy.zeros(X.shape[1])
		
		for i in range(X.shape[0]):
			score = numpy.sum(self.concave_func(X[i]))
			self.pq.add(i, -score)
		
		gain, idx = self.pq.pop()
		mask[idx] = True
		indices.append(idx)

		if self.verbose == True:
			pbar = tqdm(total=self.n_samples)
			pbar.update(1)

		for i in range(1, self.n_samples):
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
			

			indices.append(best_idx)
			mask[best_idx] = True
			current_values += X[best_idx]
			current_concave_values = self.concave_func(current_values)

			if self.verbose == True:
				pbar.update(1)

		if self.verbose == True:
			pbar.close()

		self.indices = numpy.array(indices, dtype='int32')

		if y is None:
			return X[self.indices]
		return X[self.indices], y[self.indices]