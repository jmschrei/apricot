# graphCut.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from .base import BaseGraphSelection

from ..utils import _calculate_pairwise_distances

from tqdm import tqdm

from numba import njit
from numba import prange

from scipy.sparse import csr_matrix

sieve_dtypes = 'void(float64[:,:], int64, float64[:,:], int64[:,:],' \
	'float64[:,:], float64[:], float64[:], int64[:], float64[:], int64[:])' 

def calculate_gains_sieve(dtypes, parallel, fastmath, cache):
	@njit(dtypes, parallel=parallel, fastmath=fastmath, cache=cache)
	def calculate_gains_sieve_(X, k, current_values, selections, gains, 
		total_gains, max_values, n_selected, row_sums, idxs):
		n, d = X.shape
		t = max_values.shape[0]

		for j in prange(t):
			for i in range(n):
				if n_selected[j] == k:
					break

				idx = idxs[i]
				threshold = (max_values[j] / 2. - total_gains[j]) / (k - n_selected[j])
				gain = row_sums[i] - current_values[j, i]

				if gain > threshold:
					total_gains[j] += gain

					selections[j, n_selected[j]] = idx
					gains[j, n_selected[j]] = gain
					n_selected[j] += 1

	return calculate_gains_sieve_


class GraphCutSelection(BaseGraphSelection):
	"""A selector based on using a graph-cut function.

	Graph cuts are canonical class of functions that involves selecting 
	examples that split the similarity matrix into subgraphs well. 

	.. note:: 
		All ~pairwise~ values in your data must be non-negative for this 
		selection to work.

	The general form of a graph cut function is 

	.. math::
		f(X, V) = \\lambda\\sum_{v \\in V} \\sum_{x \\in X} \\phi(x, v) - \\sum_{x, y \\in X} \\phi(x, y)

	where :math:`f` indicates the function, :math:`X` is a subset, :math:`V` 
	is the ground set, and :math:`\\phi` is the similarity measure between 
	two examples. Like most graph-based functons, the graph-cut function 
	requires access to the full similarity matrix.

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

	alpha : float
		The weight of the first term in the graph-cut objective, which
		measures the how representative the selected examples are of the
		ground set. The larger this is, the more likely examples are chosen
		that are near points in the ground set, even if there are other
		already-selected examples that are similar. Default is 1.

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

	n_neighbors : int or None
		When constructing a similarity matrix, the number of nearest neighbors
		whose similarity values will be kept. The result is a sparse similarity
		matrix which can significantly speed up computation at the cost of
		accuracy. Default is None.

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

	def __init__(self, n_samples=10, metric='euclidean', alpha=1,
		initial_subset=None, optimizer='naive', optimizer_kwds={},
		n_neighbors=None, reservoir=None, max_reservoir_size=1000, 
		n_jobs=1, random_state=None, verbose=False):
		self.alpha = alpha

		super(GraphCutSelection, self).__init__(n_samples=n_samples, 
			metric=metric, initial_subset=initial_subset, optimizer=optimizer,  
			n_neighbors=n_neighbors, reservoir=reservoir, 
			max_reservoir_size=max_reservoir_size, n_jobs=n_jobs, 
			random_state=random_state, optimizer_kwds={}, verbose=verbose)

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
		self : GraphCutSelection
			The fit step returns this selector object.
		"""

		return super(GraphCutSelection, self).fit(X, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)

	def _initialize(self, X_pairwise):
		super(GraphCutSelection, self)._initialize(X_pairwise)

		if self.reservoir is not None:
			X_pairwise = _calculate_pairwise_distances(self._X, 
				metric=self.metric)

		if self.sparse:
			self.row_sums = self.alpha * numpy.array(X_pairwise.sum(axis=1))[:,0]
			self.current_values = X_pairwise.diagonal().astype('float64')
		else:
			self.row_sums = self.alpha * X_pairwise.sum(axis=1)
			self.current_values = numpy.diag(X_pairwise).astype('float64')

		if self.initial_subset is None:
			pass
		elif self.initial_subset.ndim == 2:
			raise ValueError("When using graph-cut, the initial subset"\
				" must be a one dimensional array of indices.")
		elif self.initial_subset.ndim == 1:
			if self.sparse:
				for i in self.initial_subset:
					self.current_values += X_pairwise[i].toarray()[0] * 2
			else:
				for i in self.initial_subset:
					self.current_values += X_pairwise[i] * 2
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

		self.calculate_sieve_gains_ = calculate_gains_sieve(sieve_dtypes, 
			True, True, False)

	def _calculate_gains(self, X_pairwise, idxs=None):
		idxs = idxs if idxs is not None else self.idxs
		gains = self.row_sums[idxs] - self.current_values[idxs]
		return gains

	def _calculate_sieve_gains(self, X_pairwise, thresholds, idxs):
		"""This function will update the internal statistics from a stream.

		This function will update the various internal statistics that are a
		part of the sieve algorithm for streaming submodular optimization. This
		function does not directly return gains but it updates the values
		used by a streaming optimizer.
		"""

		super(GraphCutSelection, self)._calculate_sieve_gains(X_pairwise,
			thresholds, idxs)

		n, m = X_pairwise.shape[0], len(thresholds)
		row_sums = self.alpha * X_pairwise.mean(axis=1)
		sieve_current_values_ = numpy.tile(numpy.diag(
			_calculate_pairwise_distances(self._X, metric=self.metric)), (m, 1))

		for i in range(m):
			l = self.sieve_n_selected_[i]
			if l == self.n_samples or l == 0:
				continue

			sieve_current_values_[i] = _calculate_pairwise_distances(
				self._X, Y=self.sieve_subsets_[i, :l], 
				metric=self.metric).mean(axis=1)

		if self.sparse:
			self.calculate_sieve_gains_(X_pairwise.data, 
				X_pairwise.indices, X_pairwise.indptr, 
				self.n_samples, sieve_current_values_, 
				self.sieve_selections_, self.sieve_gains_, 
				self.sieve_total_gains_, thresholds, 
				self.sieve_n_selected_, idxs)
		else:
			self.calculate_sieve_gains_(X_pairwise, self.n_samples, 
				sieve_current_values_, self.sieve_selections_, 
				self.sieve_gains_, self.sieve_total_gains_, thresholds, 
				self.sieve_n_selected_, row_sums, idxs)

	def _select_next(self, X_pairwise, gain, idx):
		"""This function will add the given item to the selected set."""

		if self.sparse:
			self.current_values += X_pairwise.toarray()[0] * 2
		else:
			self.current_values += X_pairwise * 2

		super(GraphCutSelection, self)._select_next(
			X_pairwise, gain, idx)
