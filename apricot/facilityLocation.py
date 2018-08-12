# facilityLocation.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code implements facility location functions.
"""

import numpy

from tqdm import tqdm

class FacilityLocationSelection(object):
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

			'corr' : The squared correlation matrix
			'cosine' : The normalized dot product of the matrix

	Attributes
	----------
	n_samples : int
		The number of samples to select.

	pairwise_func : callable
		A function that takes in a data matrix and converts it to a square
		symmetric matrix.

	indices : numpy.array int
		The selected samples in the order of their gain.
	"""

	def __init__(self, n_samples=10, pairwise_func='corr', verbose=False, random_seed=None):
		self.n_samples = n_samples
		self.verbose = verbose
		self.random_seed = numpy.random.RandomState(random_seed)
		
		norm = lambda x: numpy.sum(x*x, axis=1).reshape(x.shape[0], 1)

		if pairwise_func == 'corr':
			self.pairwise_func = lambda X: numpy.corrcoef(X, rowvar=True)
		elif pairwise_func == 'cosine':
			self.pairwise_func = lambda X: numpy.dot(X, X.T) / (norm(X).dot(norm(X).T))
		elif pairwise_func == 'euclidean':
			self.pairwise_func = lambda X: -(-2 * numpy.dot(X, X.T) + norm(X)).T + norm(X)
		elif callable(pairwise_func):
			self.pairwise_func = pairwise_func
		else:
			raise KeyError("Must be one of 'corr' or 'cosine' or a custom function.")

	def fit_transform(self, X, y=None):

		if not isinstance(X, (list, numpy.ndarray)):
			raise ValueError("X must be either a list of lists or a 2D numpy array.")
		if isinstance(X, numpy.ndarray) and len(X.shape) != 2:
			raise ValueError("X must have exactly two dimensions.")

		X = numpy.array(X, dtype='float32')

		n = X.shape[0]
		mask = numpy.zeros(n, dtype=bool)
		indices = []
		
		idx = self.random_seed.randint(n)
		indices.append(idx)
		mask[idx] = True

		if self.verbose == True:
			pbar = tqdm(total=self.n_samples)
			pbar.update(1)
		
		X_pairwise = self.pairwise_func(X)
		numpy.fill_diagonal(X_pairwise, 0)
		score = X_pairwise[:,idx]


		for i in range(self.n_samples-1):
			best_gain = 0.
			best_idx = None
			
			for j in range(n):
				if mask[j] == True:
					continue
				
				score_i = numpy.maximum(X_pairwise[:,j], score)
				gain = (score_i - score).sum()
				
				if gain >= best_gain:
					best_gain = gain
					best_idx = j
					best_score = score_i

			score = best_score
			indices.append(best_idx)
			mask[best_idx] = True

			if self.verbose == True:
				pbar.update(1)

		if self.verbose == True:
			pbar.close()
		
		self.indices = numpy.array(indices)

		if y is None:
			return X[self.indices]
		else:
			return X[self.indices], y[self.indices]
