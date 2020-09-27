# mixture.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
	
import numpy

from .base import BaseSelection
from .base import BaseGraphSelection
from ..utils import _calculate_pairwise_distances

from tqdm import tqdm

class MixtureSelection(BaseSelection):
	"""A selection approach based on a mixture of submodular functions.

	A convenient property of submodular functions is that any linear 
	combination of submodular functions is still submodular. More generally, 
	the linear combination of any number of submodular functions (assuming 
	non-negative weights) is still a submodular function. Because of this, 
	a mixture of submodular functions can be optimized using the same 
	algorithms as an individual submodular function. Mixtures can be 
	useful in situations where there are different important aspects 
	of data that are each submodular.

	The general form of a mixture function is 

	.. math::
		f(X) = \\sum\\limits_{i=1}^{M} \\alpha_{i} g_{i}(X) 

	where :math:`f` indicates the mixture function, :math:`M` is the number of 
	functions in the mixture, :math:`X` is a subset, :math:`\\alpha_{i}` is the 
	weight of the :math:`i`-th function and :math:`g_{i}` is the :math:`i`-th 
	function.

	.. note::
		There must be at least two components to the mixture, and all 
		:math:`\\alpha` must be non-negative.

	This class can also be used to add regularizers to the selection procedure.
	If a submodular function is mixed with another submodular function that
	acts as a regularizer, such as feature based selection mixed with a
	custom function measuring some property of the selected subset.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

	functions : list
		The list of submodular functions to mix together. The submodular
		functions should be instantiated.

	weights : list, numpy.ndarray or None, optional
		The relative weight of each submodular function. This is the value
		that the gain from each submodular function is multiplied by before
		being added together. The default is equal weight for each function.

	initial_subset : list, numpy.ndarray or None, optional
		If provided, this should be a list of indices into the data matrix
		to use as the initial subset, or a group of examples that may not be
		in the provided data should beused as the initial subset. If indices, 
		the provided array should be one-dimensional. If a group of examples,
		the data should be 2 dimensional.

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

	verbose : bool, optional
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

	def __init__(self, n_samples, functions, weights=None, metric='ignore',
		initial_subset=None, optimizer='two-stage', optimizer_kwds={}, n_neighbors=None, 
		reservoir=None, max_reservoir_size=1000, n_jobs=1, random_state=None, 
		verbose=False):

		if len(functions) < 2:
			raise ValueError("Must mix at least two functions.")

		self.m = len(functions)
		self.functions = functions

		if weights is None:
			self.weights = numpy.ones(self.m, dtype='float64')
		else:
			self.weights = numpy.array(weights, dtype='float64', copy=False)

		super(MixtureSelection, self).__init__(n_samples=n_samples, 
			initial_subset=initial_subset, optimizer=optimizer, 
			optimizer_kwds=optimizer_kwds, reservoir=reservoir,
			max_reservoir_size=max_reservoir_size, n_jobs=n_jobs, 
			random_state=random_state, verbose=verbose)

		self.metric = metric.replace("corr", "correlation")
		self.n_neighbors = n_neighbors

		for function in self.functions:
			function.initial_subset = self.initial_subset
			function.reservoir = reservoir
			function.max_reservoir_size = max_reservoir_size
			function.metric = 'precomputed'

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
		self : MixtureSelection
			The fit step returns this selector object.
		"""

		# If self.metric is ignore, this will return the same matrix.
		# Otherwise, it will convert it to a pairwise similarity matrix.
		self._X = X
		X = _calculate_pairwise_distances(X, metric=self.metric, 
			n_neighbors=self.n_neighbors)

		return super(MixtureSelection, self).fit(X, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)

	def _initialize(self, X):
		super(MixtureSelection, self)._initialize(X)

		for function in self.functions:
			function._initialize(X)

	def _calculate_gains(self, X, idxs=None):
		"""This function will return the gain that each example would give.

		This function will return the gains that each example would give if
		added to the selected set. When a matrix of examples is given, a
		vector will be returned showing the gain for each example. When
		a single element is passed in, it will return a singe value."""

		idxs = idxs if idxs is not None else self.idxs
		gains = numpy.zeros(idxs.shape[0], dtype='float64')

		for i, function in enumerate(self.functions):
			gains += function._calculate_gains(X, idxs) * self.weights[i]

		return gains

	def _calculate_sieve_gains(self, X, thresholds, idxs):
		super(MixtureSelection, self)._calculate_sieve_gains(X, 
			thresholds, idxs)

		for function in self.functions:
			super(function.__class__, function)._calculate_sieve_gains(X,
				thresholds, idxs)

		t = numpy.zeros(thresholds.shape[0], dtype='float64') - 1

		for j in range(X.shape[0]):
			x = X[j:j+1]

			current_values = []
			total_gains = []

			gain = numpy.zeros((len(thresholds), self.n_samples), dtype='float64')
			for i, function in enumerate(self.functions):
				current_values.append(function.sieve_current_values_.copy())
				total_gains.append(function.sieve_total_gains_.copy())

				function._calculate_sieve_gains(x, t, idxs)
				gain += function.sieve_gains_ * self.weights[i]

			for l in range(len(thresholds)):
				if self.sieve_n_selected_[l] == self.n_samples:
					continue

				threshold = ((thresholds[l] / 2. - self.sieve_total_gains_[l]) 
					/ (self.n_samples - self.sieve_n_selected_[l]))

				if gain[l, self.sieve_n_selected_[l]] > threshold:
					self.sieve_total_gains_[l] += gain[l, self.sieve_n_selected_[l]]
					self.sieve_selections_[l, self.sieve_n_selected_[l]] = idxs[j]
					self.sieve_gains_[l, self.sieve_n_selected_[l]] = gain[l, self.sieve_n_selected_[l]]
					self.sieve_n_selected_[l] += 1
				else:
					v = self.sieve_n_selected_[l]

					for i, function in enumerate(self.functions):
						vi = function.sieve_n_selected_[l]
						if vi != self.sieve_n_selected_[l]:
							function.sieve_current_values_[l] = current_values[i][l]
							function.sieve_total_gains_[l] = total_gains[i][l]
							function.sieve_n_selected_[l] = vi - 1
							if vi < self.n_samples:
								function.sieve_selections_[l, vi] = 0
								function.sieve_gains_[l, vi] = 0
								function.sieve_subsets_[l, vi] = 0

	def _select_next(self, X, gain, idx):
		"""This function will add the given item to the selected set."""

		for function in self.functions:
			function._select_next(X, gain, idx)

		super(MixtureSelection, self)._select_next(X, gain, idx)
