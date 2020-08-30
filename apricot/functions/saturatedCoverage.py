# saturatedCoverage.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from .base import BaseGraphSelection

from tqdm import tqdm

from numba import njit
from numba import prange

from scipy.sparse import csr_matrix

dtypes = 'void(float64[:,:], float64[:], float64[:], float64[:], int64[:])'
sdtypes = 'void(float64[:], int32[:], int32[:], float64[:], float64[:], float64[:], int64[:])'

@njit(dtypes, nogil=True, parallel=True)
def select_next(X, gains, current_values, max_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i]
		gains[i] = numpy.minimum(current_values + X[idx], max_values).sum()

@njit(sdtypes, nogil=True, parallel=True)
def select_next_sparse(X_data, X_indices, X_indptr, gains, current_values, max_values, idxs):
	for i in prange(idxs.shape[0]):
		idx = idxs[i]

		start = X_indptr[idx]
		end = X_indptr[idx+1]

		for j in range(start, end):
			k = X_indices[j]
			gains[i] += min(X_data[j] + current_values[k], max_values[k]) - current_values[k]

class SaturatedCoverageSelection(BaseGraphSelection):
	"""A saturated coverage submodular selection algorithm.

	The saturated coverage function is a graph-based function where the gain 
	is the sum of the similarities between the candidate example and the 
	entire ground set up until a certain point. Essentially, each example in 
	the ground set contributes to the overall function until some prespecified 
	level of "coverage" is met. Once an example is similar enough to the 
	selected examples, it stops contributing to the function.

	.. note::
		All ~pairwise~ values in your data must be positive for this 
		selection to work.

	The general form of a saturated coverage function is 

	.. math::
		f(X, V) = \\sum_{v \\in V} \\min\\{\\sum_{x \\in X} \\phi(x, v), \\alpha\\}

	where :math:`f` indicates the function, :math:`X` is a subset, :math:`V` 
	is the ground set, and :math:`\\phi` is the similarity measure between two 
	examples, and :math:`\\alpha` is a parameter that, for each point in the 
	ground set, specifies the maximum similarity that each example can have 
	to the selected set (the saturation). Like most graph-based functons, 
	the saturated coverage function requires access to the full ground set.

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

	def __init__(self, n_samples=10, metric='euclidean', alpha=0.1,
		initial_subset=None, optimizer='two-stage', n_neighbors=None, n_jobs=1, 
		random_state=None, optimizer_kwds={}, verbose=False):
		self.alpha = alpha

		super(SaturatedCoverageSelection, self).__init__(n_samples=n_samples, 
			metric=metric,initial_subset=initial_subset, optimizer=optimizer, 
			optimizer_kwds=optimizer_kwds, n_neighbors=n_neighbors, 
			n_jobs=n_jobs, random_state=random_state, verbose=verbose)

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
		self : SaturatedCoverageSelection
			The fit step returns this selector object.
		"""

		return super(SaturatedCoverageSelection, self).fit(X, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)

	def _initialize(self, X_pairwise):
		super(SaturatedCoverageSelection, self)._initialize(X_pairwise)

		if self.sparse:
			self.max_values = self.alpha * numpy.array(
				X_pairwise.sum(axis=1))[:,0]
		else:
			self.max_values = self.alpha * X_pairwise.sum(axis=1)

		if self.initial_subset is None:
			return
		elif self.initial_subset.ndim == 2:
			raise ValueError("When using saturated coverage, the initial subset"\
				" must be a one dimensional array of indices.")
		elif self.initial_subset.ndim == 1:
			if not self.sparse:
				for i in self.initial_subset:
					self.current_values = numpy.minimum(self.max_values,
						self.current_values + X_pairwise[i])
			else:
				for i in self.initial_subset:
					self.current_values = numpy.minimum(self.max_values,
						self.current_values + X_pairwise[i].toaray()[0])
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

	def _calculate_gains(self, X_pairwise, idxs=None):
		idxs = idxs if idxs is not None else self.idxs
		gains = numpy.zeros(idxs.shape[0], dtype='float64')

		if self.sparse:
			select_next_sparse(X_pairwise.data,
				X_pairwise.indices, X_pairwise.indptr, gains,
				self.current_values, self.max_values, idxs)
		else:
			select_next(X_pairwise, gains, self.current_values,
				self.max_values, idxs)
			gains -= self.current_values.sum()

		return gains

	def _select_next(self, X_pairwise, gain, idx):
		"""This function will add the given item to the selected set."""

		if self.sparse:
			self.current_values = numpy.minimum(self.max_values,
				X_pairwise.toarray()[0] + self.current_values)
		else:
			self.current_values = numpy.minimum(self.max_values,
				self.current_values + X_pairwise)

		super(SaturatedCoverageSelection, self)._select_next(
			X_pairwise, gain, idx)
