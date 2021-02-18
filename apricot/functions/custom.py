# custom.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from .base import BaseSelection
from .base import BaseGraphSelection

class CustomSelection(BaseSelection):
	"""A selector based off a custom, user-defined feature-based function.

	This selector wraps a custom function that is passed in by the user. This
	function should take in a matrix containing a subset of the ground set and
	output the standard measure of quality of the subset. Each row in the matrix
	should be an example and each column should be a feature value for the 
	example (like a standard data matrix for machine learning).

	.. warning::
		If the function that the user wants to optimize is graph-based,
		i.e., that it determines the quality of a subset using similarities
		with other examples instead of the feature values directly, 
		e.g. facility location, CustomGraphSelection should be used instead. 
	
	.. note:: 
		Although apricot is built for submodular functions, there is no explicit
		restriction that the function passed in be submodular. Sometimes,
		supermodular functions can be reasonably maximized using the same
		greedy approaches used on submodular functions.


	Parameters
	----------
	n_samples : int
		The number of samples to return.

	func : function
		The feature-based set function to that this selector should wrap.

	initial_subset : list, numpy.ndarray or None
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

	optimizer_kwds : dict, optional
		Arguments to pass into the optimizer object upon initialization.
		Default is {}.

	function_kwds : dict, optional
		Arguments to pass into the function object for each call.
		Default is {}.

	n_jobs : int, optional
		The number of cores to use for processing. This value is multiplied
		by 2 when used to set the number of threads. If set to -1, use all
		cores and threads. Default is -1.

	random_state : int or RandomState or None, optional
		The random seed to use for the random selection process. Only used
		for stochastic greedy.

	verbose : bool
		Whether to print output during the selection process.

	Attributes
	----------
	n_samples : int
		The number of samples to select.

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

	def __init__(self, n_samples, function, initial_subset=None, 
		optimizer='two-stage', optimizer_kwds={}, function_kwds={},
		n_jobs=1, random_state=None, verbose=False):

		if callable(function) == False:
			raise ValueError("Passed in function must be callable.")

		self.function = function
		self.function_kwds = function_kwds

		super(CustomSelection, self).__init__(n_samples=n_samples, 
			initial_subset=initial_subset, optimizer=optimizer, 
			optimizer_kwds=optimizer_kwds, n_jobs=n_jobs, 
			random_state=random_state, verbose=verbose) 

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
		self : CustomSelection
			The fit step returns this selector object.
		"""

		return super(CustomSelection, self).fit(X, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)

	def _initialize(self, X):
		super(CustomSelection, self)._initialize(X)

		if self.initial_subset is None:
			pass
		elif self.initial_subset.ndim == 2:
			if self.initial_subset.shape[1] != X.shape[1]:
				raise ValueError("The number of columns in the initial subset must " \
					"match the number of columns in X.")
		elif self.initial_subset.ndim == 1:
			self.initial_subset = X[self.initial_subset]
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

		if self.initial_subset is None:
			self.total_gain = 0
		else:
			self.total_gain = self.function(self.initial_subset)


	def _calculate_gains(self, X, idxs=None):
		"""This function will return the gain that each example would give.

		This function will return the gains that each example would give if
		added to the selected set. When a matrix of examples is given, a
		vector will be returned showing the gain for each example. When
		a single element is passed in, it will return a singe value."""

		idxs = idxs if idxs is not None else self.idxs
		gains = numpy.zeros(idxs.shape[0], dtype='float64')

		x0 = numpy.zeros((1, X.shape[1]))

		if self.initial_subset is not None:
			X_ = numpy.concatenate([self.initial_subset, self.subset, x0])
		else:
			X_ = numpy.concatenate([self.subset, x0])

		for i, idx in enumerate(idxs):
			X_[-1] = X[idx]
			gains[i] = self.function(X_, **self.function_kwds) - self.total_gain

		return gains

	def _calculate_sieve_gains(self, X, thresholds, idxs):
		"""This function will update the internal statistics from a stream.

		This function will update the various internal statistics that are a
		part of the sieve algorithm for streaming submodular optimization. This
		function does not directly return gains but it updates the values
		used by a streaming optimizer.
		"""

		super(CustomSelection, self)._calculate_sieve_gains(X,
			thresholds, idxs)

		raise NotImplementedError

	def _select_next(self, X, gain, idx):
		"""This function will add the given item to the selected set."""

		self.total_gain += gain

		super(CustomSelection, self)._select_next(
			X, gain, idx)


class CustomGraphSelection(BaseGraphSelection):
	"""A selector based off a custom, user-defined graph-based function.

	This selector wraps a custom graph-based function that is passed in by the user. 
	This function should take in a matrix containing a subset of the ground set and
	output the standard measure of quality of the subset. Each row in the matrix
	should be an example and each column should be the similarity to each element
	in the ground set, with the number of columns being equal to the size of the
	ground set.
	
	.. note:: 
		Although apricot is built for submodular functions, there is no explicit
		restriction that the function passed in be submodular. Sometimes,
		supermodular functions can be reasonably maximized using the same
		greedy approaches used on submodular functions.


	Parameters
	----------
	n_samples : int
		The number of samples to return.

	func : function
		The graph-based set function to that this selector should wrap.

	initial_subset : list, numpy.ndarray or None
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

	optimizer_kwds : dict, optional
		Arguments to pass into the optimizer object upon initialization.
		Default is {}.

	function_kwds : dict, optional
		Arguments to pass into the function object for each call.
		Default is {}.

	n_jobs : int, optional
		The number of cores to use for processing. This value is multiplied
		by 2 when used to set the number of threads. If set to -1, use all
		cores and threads. Default is -1.

	random_state : int or RandomState or None, optional
		The random seed to use for the random selection process. Only used
		for stochastic greedy.

	verbose : bool
		Whether to print output during the selection process.

	Attributes
	----------
	n_samples : int
		The number of samples to select.

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

	def __init__(self, n_samples, function, metric='euclidean', 
		initial_subset=None, optimizer='two-stage', optimizer_kwds={}, 
		function_kwds={}, n_jobs=1, random_state=None, verbose=False):

		if callable(function) == False:
			raise ValueError("Passed in function must be callable.")

		self.function = function
		self.function_kwds = function_kwds

		super(CustomGraphSelection, self).__init__(n_samples=n_samples, 
			metric=metric, initial_subset=initial_subset, optimizer=optimizer, 
			optimizer_kwds=optimizer_kwds, n_jobs=n_jobs, 
			random_state=random_state, verbose=verbose) 

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
		self : CustomSelection
			The fit step returns this selector object.
		"""

		return super(CustomGraphSelection, self).fit(X, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)

	def _initialize(self, X):
		super(CustomGraphSelection, self)._initialize(X)

		if self.initial_subset is None:
			pass
		elif self.initial_subset.ndim == 2:
			if self.initial_subset.shape[1] != X.shape[1]:
				raise ValueError("The number of columns in the initial subset must " \
					"match the number of columns in X.")
		elif self.initial_subset.ndim == 1:
			self.initial_subset = X[self.initial_subset]
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

		if self.initial_subset is None:
			self.total_gain = 0
		else:
			self.total_gain = self.func(self.initial_subset)


	def _calculate_gains(self, X, idxs=None):
		"""This function will return the gain that each example would give.

		This function will return the gains that each example would give if
		added to the selected set. When a matrix of examples is given, a
		vector will be returned showing the gain for each example. When
		a single element is passed in, it will return a singe value."""

		idxs = idxs if idxs is not None else self.idxs
		gains = numpy.zeros(idxs.shape[0], dtype='float64')

		x0 = numpy.zeros((1, X.shape[1]))

		if self.initial_subset is not None:
			X_ = numpy.concatenate([self.initial_subset, self.subset, x0])
		else:
			X_ = numpy.concatenate([self.subset, x0])

		for i, idx in enumerate(idxs):
			X_[-1] = X[idx]
			gains[i] = self.function(X_, **self.function_kwds) - self.total_gain

		return gains

	def _calculate_sieve_gains(self, X, thresholds, idxs):
		"""This function will update the internal statistics from a stream.

		This function will update the various internal statistics that are a
		part of the sieve algorithm for streaming submodular optimization. This
		function does not directly return gains but it updates the values
		used by a streaming optimizer.
		"""

		super(CustomGraphSelection, self)._calculate_sieve_gains(X,
			thresholds, idxs)

		raise NotImplementedError

	def _select_next(self, X, gain, idx):
		"""This function will add the given item to the selected set."""

		self.total_gain += gain

		super(CustomGraphSelection, self)._select_next(
			X, gain, idx)
