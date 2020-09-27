# sumRedundancy.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from .base import BaseGraphSelection

from tqdm import tqdm

class SumRedundancySelection(BaseGraphSelection):
	"""A selector based off a sum redundancy submodular function.
	
	The sum redundancy function is a graph-based function that penalizes 
	redundancy among the selected set. This approach is straightforward, 
	in that it simply involved a sum. It is also fast in comparison to a 
	facility location function because it involves only performing calculation 
	over the selected set as opposed to the entire ground set. Because the sum 
	of the similarities is not submodular, it is subtracted from the sum of 
	the entire similarity matrix, such that examples that are highly similar 
	to each other result in a lower value than examples that are not very 
	similar.

	.. note:: 
		All ~pairwise~ values in your data must be positive for this 
		selection to work.

	The general form of a sum redundancy function is 

	.. math::
		f(X, V) = \sum_{x, y \in V} \phi(x, y) - \sum_{x, y\in X} \phi(x,y)

	where :math:`f` indicates the function, :math:`X` is the selected subset, 
	:math:`V` is the ground set, and :math:`\phi` is the similarity measure 
	between two examples. While sum redundancy functions involves calculating 
	the sum of the entire similarity matrix in principle, in practice if one 
	is only calculating the gains this step can be ignored.
	
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

	def __init__(self, n_samples=10, metric='euclidean', 
		initial_subset=None, optimizer='two-stage', n_neighbors=None, 
		reservoir=None, max_reservoir_size=1000, n_jobs=1, 
		random_state=None, optimizer_kwds={}, verbose=False):

		super(SumRedundancySelection, self).__init__(n_samples=n_samples, 
			metric=metric, initial_subset=initial_subset, optimizer=optimizer, 
			n_neighbors=n_neighbors, reservoir=reservoir, 
			max_reservoir_size=max_reservoir_size, n_jobs=n_jobs, 
			random_state=random_state, optimizer_kwds=optimizer_kwds, 
			verbose=verbose)

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
		self : SumRedundancySelection
			The fit step returns this selector object.
		"""

		return super(SumRedundancySelection, self).fit(X, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)

	def _initialize(self, X_pairwise, idxs=None):
		super(SumRedundancySelection, self)._initialize(X_pairwise, idxs=idxs)
		idxs = idxs if idxs is not None else numpy.arange(X_pairwise.shape[0])

		for i, idx in enumerate(idxs):
			self.current_values[i] = X_pairwise[idx, idx]

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
