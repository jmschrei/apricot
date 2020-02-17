# sumRedundancy.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code implements the graph cut function.
"""

try:
	import cupy
except:
	import numpy as cupy

import numpy

from .base import BaseGraphSelection

from tqdm import tqdm

class SumRedundancySelection(BaseGraphSelection):
	"""A selector based off a sum redundancy submodular function.

	NOTE: All ~pairwise~ values in your data must be positive for this 
	selection to work.

	This selector uses a sum redundancy function to perform selection. The
	sum redundancy function is based on maximizing the difference between
	the 

	This selector uses a sum redundancy submodular function to perform
	selection. The sum redundancy function is based on maximizing the
	pairwise similarities between the points in the data set and their nearest
	chosen point. The similarity function can be species by the user but must
	be non-negative where a higher value indicates more similar.

	This implementation allows users to pass in either their own symmetric
	square matrix of similarity values, or a data matrix as normal and a function
	that calculates these pairwise values.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

	metric : str, optional
		The method for converting a data matrix into a square symmetric matrix
		of pairwise similarities. If a string, can be any of the metrics
		implemented in sklearn (see https://scikit-learn.org/stable/modules/
		generated/sklearn.metrics.pairwise_distances.html), including
		"precomputed" if one has already generated a similarity matrix. Note
		that sklearn calculates distance matrices whereas apricot operates on
		similarity matrices, and so a distances.max() - distances transformation
		is performed on the resulting distances. For backcompatibility,
		'corr' will be read as 'correlation'. Default is 'euclidean'.

	n_naive_samples : int, optional
		The number of samples to perform the naive greedy algorithm on
		before switching to the lazy greedy algorithm. The lazy greedy
		algorithm is faster once features begin to saturate, but is slower
		in the initial few selections. This is, in part, because the naive
		greedy algorithm is parallelized whereas the lazy greedy
		algorithm currently is not. Default is 1.

	initial_subset : list, numpy.ndarray or None, optional
		If provided, this should be a list of indices into the data matrix
		to use as the initial subset, or a group of examples that may not be
		in the provided data should beused as the initial subset. If indices, 
		the provided array should be one-dimensional. If a group of examples,
		the data should be 2 dimensional. Default is None.

	optimizer : string or optimizers.BaseOptimizer, optional
		The optimization approach to use for the selection. Default is
		'two-stage', which makes selections using the naive greedy algorithm
		initially and then switches to the lazy greedy algorithm. Must be
		one of

			'naive' : the naive greedy algorithm
			'lazy' : the lazy (or accelerated) greedy algorithm
			'approximate-lazy' : the approximate lazy greedy algorithm
			'two-stage' : starts with naive and switches to lazy
			'stochastic' : the stochastic greedy algorithm
			'greedi' : the GreeDi distributed algorithm
			'bidirectional' : the bidirectional greedy algorithm

		Default is 'naive'.

	epsilon : float, optional
		The inverse of the sampling probability of any particular point being 
		included in the subset, such that 1 - epsilon is the probability that
		a point is included. Only used for stochastic greedy. Default is 0.9.

	random_state : int or RandomState or None, optional
		The random seed to use for the random selection process. Only used
		for stochastic greedy.

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

	def __init__(self, n_samples=10, metric='euclidean', 
		initial_subset=None, optimizer='two-stage', n_neighbors=None, n_jobs=1, 
		random_state=None, optimizer_kwds={}, verbose=False):

		super(SumRedundancySelection, self).__init__(n_samples=n_samples, 
			metric=metric, initial_subset=initial_subset, optimizer=optimizer, 
			n_neighbors=n_neighbors, n_jobs=n_jobs, random_state=random_state, 
			optimizer_kwds=optimizer_kwds, verbose=verbose)

	def fit(self, X, y=None):
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

		y : list or numpy.ndarray, shape=(n,), optional
			The labels to transform. If passed in this function will return
			both the data and th corresponding labels for the rows that have
			been selected.

		Returns
		-------
		self : SumRedundancySelection
			The fit step returns this selector object.
		"""

		return super(SumRedundancySelection, self).fit(X, y)

	def _initialize(self, X_pairwise, idxs=None):
		super(SumRedundancySelection, self)._initialize(X_pairwise, idxs=idxs)
		idxs = idxs if idxs is not None else numpy.arange(X_pairwise.shape[0])

		for i, idx in enumerate(idxs):
			self.current_values[i] = X_pairwise[i, idx]

		if self.initial_subset is None:
			return
		elif self.initial_subset.ndim == 2:
			raise ValueError("When using saturated coverage, the initial subset"\
				" must be a one dimensional array of indices.")
		elif self.initial_subset.ndim == 1:
			if not self.sparse:
				for i in self.initial_subset:
					self.current_values += X_pairwise[i] * 2 
			else:
				for i in self.initial_subset:
					self.current_values += X_pairwise[i].toarray()[0] * 2
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

	def _calculate_gains(self, X_pairwise, idxs=None):
		idxs = idxs if idxs is not None else self.idxs
		return -self.current_values[idxs]

	def _select_next(self, X_pairwise, gain, idx):
		"""This function will add the given item to the selected set."""

		if self.sparse:
			self.current_values += X_pairwise.toarray()[0] * 2
		else:
			self.current_values += X_pairwise * 2

		super(SumRedundancySelection, self)._select_next(
			X_pairwise, gain, idx)
