# base.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 

"""
This file contains code that implements the core of the submodular selection
algorithms.
"""

import numpy
from tqdm import tqdm

from ..optimizers import BaseOptimizer
from ..optimizers import NaiveGreedy
from ..optimizers import LazyGreedy
from ..optimizers import ApproximateLazyGreedy
from ..optimizers import TwoStageGreedy
from ..optimizers import StochasticGreedy
from ..optimizers import BidirectionalGreedy
from ..optimizers import GreeDi
from ..optimizers import SieveGreedy
from ..optimizers import OPTIMIZERS

from ..utils import PriorityQueue
from ..utils import check_random_state
from ..utils import _calculate_pairwise_distances

from scipy.sparse import csr_matrix


class BaseSelection(object):
	"""The base selection object.

	This object defines the structures that all submodular selection algorithms
	should follow. All algorithms will have the same public methods and the
	same attributes.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

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

			'naive' : the naive greedy algorithm
			'lazy' : the lazy (or accelerated) greedy algorithm
			'approximate-lazy' : the approximate lazy greedy algorithm
			'two-stage' : starts with naive and switches to lazy
			'stochastic' : the stochastic greedy algorithm
			'greedi' : the GreeDi distributed algorithm
			'bidirectional' : the bidirectional greedy algorithm

		Default is 'naive'.

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

	def __init__(self, n_samples, initial_subset=None, optimizer='lazy', 
		optimizer_kwds={}, reservoir=None, max_reservoir_size=1000, 
		n_jobs=1, random_state=None, verbose=False):
		if n_samples <= 0:
			raise ValueError("n_samples must be a positive value.")

		if not isinstance(initial_subset, (list, numpy.ndarray)) and initial_subset is not None: 
			raise ValueError("initial_subset must be a list, numpy array, or None")
		if isinstance(initial_subset, (list, numpy.ndarray)):
			initial_subset = numpy.array(initial_subset)

		if not isinstance(optimizer, BaseOptimizer):
			if optimizer not in OPTIMIZERS.keys():
				raise ValueError("Optimizer must be an optimizer object or " \
					"a str in {}.".format(str(OPTIMIZERS.keys())))

		if isinstance(optimizer, BaseOptimizer):
			optimizer.function = self

		if verbose not in (True, False):
			raise ValueError("verbosity must be True or False")

		self.n_samples = n_samples
		self.metric = 'ignore'
		self.random_state = check_random_state(random_state)
		self.optimizer = optimizer
		self.optimizer_kwds = optimizer_kwds
		self.n_jobs = n_jobs
		self.verbose = verbose
		self.initial_subset = initial_subset

		self.ranking = None
		self.idxs = None
		self.gains = None
		self.subset = None
		self.sparse = None
		self._X = None
		
		self.sieve_current_values_ = None
		self.n_seen_ = 0
		self.reservoir_size = 0 if reservoir is None else reservoir.shape[0]
		self.reservoir = reservoir
		self.max_reservoir_size = max_reservoir_size if reservoir is None else reservoir.shape[0]
		self.update_reservoir_ = reservoir is None

	def fit(self, X, y=None, sample_weight=None, sample_cost=None):
		"""Run submodular optimization to select a subset of examples.

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
		self : BaseGraphSelection
			The fit step returns this selector object.
		"""

		allowed_dtypes = list, numpy.ndarray, csr_matrix

		if not isinstance(X, allowed_dtypes):
			raise ValueError("X must be either a list of lists, a 2D numpy " \
				"array, or a scipy.sparse.csr_matrix.")
		if isinstance(X, numpy.ndarray) and len(X.shape) != 2:
			raise ValueError("X must have exactly two dimensions.")
		if numpy.min(X) < 0.0 and numpy.max(X) > 0.:
			raise ValueError("X cannot contain negative values or must be entirely "\
				"negative values.")
		if self.n_samples > X.shape[0]:
			raise ValueError("Cannot select more examples than the number in" \
				" the data set.")

		if not self.sparse:
			if X.dtype != 'float64':
				X = X.astype('float64')

		if isinstance(self.optimizer, str):
			optimizer = OPTIMIZERS[self.optimizer](function=self, 
				verbose=self.verbose, random_state=self.random_state,
				**self.optimizer_kwds)
		else:
			optimizer = self.optimizer

		self._X = X if self._X is None else self._X
		self._initialize(X)

		if self.verbose:
			self.pbar = tqdm(total=self.n_samples, unit_scale=True)

		optimizer.select(X, self.n_samples, sample_cost=sample_cost)

		if self.verbose == True:
			self.pbar.close()

		self.ranking = numpy.array(self.ranking)
		self.gains = numpy.array(self.gains)
		return self

	def partial_fit(self, X, y=None, sample_weight=None, sample_cost=None):
		allowed_dtypes = list, numpy.ndarray, csr_matrix

		if not isinstance(X, allowed_dtypes):
			raise ValueError("X must be either a list of lists, a 2D numpy " \
				"array, or a scipy.sparse.csr_matrix.")
		if isinstance(X, numpy.ndarray) and len(X.shape) != 2:
			raise ValueError("X must have exactly two dimensions.")

		if not self.sparse:
			if X.dtype != 'float64':
				X = X.astype('float64')

		if not isinstance(self.optimizer, SieveGreedy):
			self.optimizer = OPTIMIZERS['sieve'](function=self, 
				verbose=self.verbose, random_state=self.random_state,
				**self.optimizer_kwds)

		self._X = X if self._X is None else self._X
		self._initialize(X)

		if self.verbose:
			self.pbar = tqdm(total=self.n_samples, unit_scale=True)

		self.optimizer.select(X, self.n_samples, sample_cost=sample_cost)

		if self.verbose == True:
			self.pbar.close()

		self.ranking = numpy.array(self.ranking)
		self.gains = numpy.array(self.gains)
		self._X = None
		return self

	def transform(self, X, y=None, sample_weight=None):
		"""Transform a data set to include only the selected examples.

		This method will return a selection of X and optionally selections
		of y and sample_weight. The default setting is to select items based
		on the ranking determined in the `fit` step with examples in the same
		order as that ranking. Optionally, the whole data set can be returned,
		with the weights corresponding to samples that were not selected set
		to 0. This setting can be controlled by setting `pipeline=True`. 

		Parameters
		----------
		X : list or numpy.ndarray, shape=(n, d)
			The data set to transform. Must be numeric.

		y : list or numpy.ndarray or None, shape=(n,), optional
			The labels to transform. If passed in this function will return
			both the data and the corresponding labels for the rows that have
			been selected. Default is None. 

		sample_weight : list or numpy.ndarray or None, shape=(n,), optional
			The sample weights to transform. If passed in this function will
			return the selected labels (y) and the selected samples, even
			if no labels were passed in. Default is None.

		Returns
		-------
		X_subset : numpy.ndarray, shape=(n_samples, d)
			A subset of the data such that n_samples < n and n_samples is the
			integer provided at initialization.

		y_subset : numpy.ndarray, shape=(n_samples,), optional
			The labels that match with the indices of the samples if y is
			passed in. Only returned if passed in.

		sample_weight_subset : numpy.ndarray, shape=(n_samples,), optional
			The weight of each example.
		"""

		r = self.ranking

		if sample_weight is not None:
			if y is None:
				return X[r], None, sample_weight[r]
			else:
				return X[r], y[r], sample_weight[r]

		else:
			if y is None:
				return X[r]
			else:
				return X[r], y[r]

	def fit_transform(self, X, y=None, sample_weight=None, sample_cost=None):
		"""Run optimization and select a subset of examples.

		This method will first perform the `fit` step and then perform the
		`transform` step, returning a transformed data set. 

		Parameters
		----------
		X : list or numpy.ndarray, shape=(n, d)
			The data set to transform. Must be numeric.

		y : list or numpy.ndarray or None, shape=(n,), optional
			The labels to transform. If passed in this function will return
			both the data and the corresponding labels for the rows that have
			been selected. Default is None. 

		sample_weight : list or numpy.ndarray or None, shape=(n,), optional
			The sample weights to transform. If passed in this function will
			return the selected labels (y) and the selected samples, even
			if no labels were passed in. Default is None.

		sample_cost : list or numpy.ndarray or None, shape=(n,), optional
			The cost of each item. If set, indicates that optimization should
			be performed with respect to a knapsack constraint.

		Returns
		-------
		X_subset : numpy.ndarray, shape=(n_samples, d)
			A subset of the data such that n_samples < n and n_samples is the
			integer provided at initialization.

		y_subset : numpy.ndarray, shape=(n_samples,), optional
			The labels that match with the indices of the samples if y is
			passed in. Only returned if passed in.

		sample_weight_subset : numpy.ndarray, shape=(n_samples,), optional
			The weight of each example.
		"""

		return self.fit(X, y=y, sample_weight=sample_weight, 
			sample_cost=sample_cost).transform(X, y=y, 
			sample_weight=sample_weight)

	def _initialize(self, X, idxs=None):
		n, d = X.shape
		self._X = X if self._X is None else self._X

		self.sparse = isinstance(X, csr_matrix)
		self.ranking = []
		self.gains = []
		self.subset = numpy.zeros((0, self._X.shape[1]), dtype='float64')

		self.current_values = numpy.zeros(d, dtype='float64')
		self.current_concave_values = numpy.zeros(d, dtype='float64')
		self.mask = numpy.zeros(n, dtype='int8')

		if self.initial_subset is not None:
			if self.initial_subset.ndim == 1:
				if self.initial_subset.dtype == bool:
					self.initial_subset = numpy.where(self.initial_subset == 1)[0]
				
				if len(self.initial_subset) + self.n_samples > X.shape[0]:
					raise ValueError("When using a mask for the initial subset" \
						" must selected fewer than the size of the subset minus" \
						" the initial subset size, i.e., n_samples < X.shape[0] -"\
						" initial_subset.shape[0].")

				if self.initial_subset.max() > X.shape[0]:
					raise ValueError("When passing in an integer mask for the initial subset"\
						" the maximum value cannot exceed the size of the data set.")
				elif self.initial_subset.min() < 0:
					raise ValueError("When passing in an integer mask for the initial subset"\
						" the minimum value cannot be negative.")
				
				self.mask[self.initial_subset] = 1

		self.idxs = numpy.where(self.mask == 0)[0]

	def _calculate_gains(self, X, idxs=None):
		raise NotImplementedError

	def _calculate_sieve_gains(self, X, thresholds, idxs):
		n = X.shape[0]
		d = X.shape[1] if self.reservoir is None else self.max_reservoir_size
		l = len(thresholds)

		if self.sieve_current_values_ is None:
			self.sieve_current_values_ = numpy.zeros((l, d), 
				dtype='float64')
			self.sieve_selections_ = numpy.zeros((l, self.n_samples), 
				dtype='int64') - 1
			self.sieve_gains_ = numpy.zeros((l, self.n_samples), 
				dtype='float64') - 1
			self.sieve_n_selected_ = numpy.zeros(l, 
				dtype='int64')
			self.sieve_total_gains_ = numpy.zeros(l, 
				dtype='float64')
			self.sieve_subsets_ = numpy.zeros((l, self.n_samples, 
				self._X.shape[1]), dtype='float64')
		else:
			j = l - self.sieve_current_values_.shape[0]
			if j > 0:
				self.sieve_current_values_ = numpy.vstack([
					self.sieve_current_values_, numpy.zeros((j, d), 
						dtype='float64')])
				self.sieve_selections_ = numpy.vstack([
					self.sieve_selections_, numpy.zeros((j, self.n_samples), 
						dtype='int64') - 1])
				self.sieve_gains_ = numpy.vstack([self.sieve_gains_, 
					numpy.zeros((j, self.n_samples), dtype='float64')])
				self.sieve_n_selected_ = numpy.concatenate([
					self.sieve_n_selected_, numpy.zeros(j, dtype='int64')])
				self.sieve_total_gains_ = numpy.concatenate([
					self.sieve_total_gains_, numpy.zeros(j, dtype='float64')])
				self.sieve_subsets_ = numpy.concatenate([self.sieve_subsets_, 
					numpy.zeros((j, self.n_samples, self._X.shape[1]), 
						dtype='float64')])

	def _select_next(self, X, gain, idx):
		self.ranking.append(idx)
		self.gains.append(gain)
		self.mask[idx] = True
		self.idxs = numpy.where(self.mask == 0)[0]

		if self.sparse:
			X = self._X[idx:idx+1].toarray()
		else:
			X = self._X[idx:idx+1]

		if self.metric != 'precomputed':
			self.subset = numpy.concatenate([self.subset, X])


class BaseGraphSelection(BaseSelection):
	"""The base graph selection object.

	This object defines the structures that all submodular selection algorithms
	should follow if they operate on a graph, such as pairwise similarity 
	measurements. All algorithms will have the same public methods and the same 
	attributes.

	NOTE: All ~pairwise~ values in your data must be positive for these
	selection methods to work.

	This implementation allows users to pass in either their own symmetric
	square matrix of similarity values, or a data matrix as normal and a function
	that calculates these pairwise values.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

	metric : str
		The method for converting a data matrix into a square symmetric matrix
		of pairwise similarities. If a string, can be any of the metrics
		implemented in sklearn (see https://scikit-learn.org/stable/modules/
		generated/sklearn.metrics.pairwise_distances.html), including
		"precomputed" if one has already generated a similarity matrix. Note
		that sklearn calculates distance matrices whereas apricot operates on
		similarity matrices, and so a distances.max() - distances transformation
		is performed on the resulting distances. For backcompatibility,
		'corr' will be read as 'correlation'.

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

			'naive' : the naive greedy algorithm
			'lazy' : the lazy (or accelerated) greedy algorithm
			'approximate-lazy' : the approximate lazy greedy algorithm
			'two-stage' : starts with naive and switches to lazy
			'stochastic' : the stochastic greedy algorithm
			'greedi' : the GreeDi distributed algorithm
			'bidirectional' : the bidirectional greedy algorithm

		Default is 'naive'.

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

	metric : callable
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

	def __init__(self, n_samples, metric='euclidean', 
		initial_subset=None, optimizer='two-stage', optimizer_kwds={},
		n_neighbors=None, reservoir=None, max_reservoir_size=1000, 
		n_jobs=1, random_state=None, verbose=False):

		super(BaseGraphSelection, self).__init__(n_samples=n_samples, 
			initial_subset=initial_subset, optimizer=optimizer, 
			optimizer_kwds=optimizer_kwds, reservoir=reservoir, 
			max_reservoir_size=max_reservoir_size, n_jobs=n_jobs, 
			random_state=random_state, verbose=verbose)

		self.metric = metric.replace("corr", "correlation")
		self.n_neighbors = n_neighbors


	def fit(self, X, y=None, sample_weight=None, sample_cost=None):
		"""Run submodular optimization to select a subset of examples.

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
		self : BaseGraphSelection
			The fit step returns this selector object.
		"""

		if isinstance(X, csr_matrix) and self.metric not in ("precomputed", "ignore"):
			raise ValueError("Must passed in a precomputed sparse " \
				"similarity matrix or a dense feature matrix.")
		if self.metric == 'precomputed' and X.shape[0] != X.shape[1]:
			raise ValueError("Precomputed similarity matrices " \
				"must be square and symmetric.")

		X_pairwise = _calculate_pairwise_distances(X, metric=self.metric, 
			n_neighbors=self.n_neighbors)
	
		self._X = X
		return super(BaseGraphSelection, self).fit(X_pairwise, y=y,
			sample_weight=sample_weight, sample_cost=sample_cost)

	def partial_fit(self, X, y=None, sample_weight=None, sample_cost=None):
		if self.reservoir is None:
			self.reservoir = numpy.empty((self.max_reservoir_size, X.shape[1]))

		if self.update_reservoir_:
			for i in range(X.shape[0]):
				if self.reservoir_size < self.max_reservoir_size:
					self.reservoir[self.reservoir_size] = X[i]
					self.reservoir_size += 1
				else:
					r = self.random_state.choice(self.n_seen_ + i)
					if r < self.max_reservoir_size:
						self.reservoir[r] = X[i]
						#self.current_values_[:, r] = 0.

		X_pairwise = _calculate_pairwise_distances(X, 
			Y=self.reservoir[:self.reservoir_size], metric=self.metric)

		self._X = X
		super(BaseGraphSelection, self).partial_fit(X_pairwise, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)

		self.current_values = numpy.zeros(self.reservoir_size, 
			dtype='float64')
		self.n_seen_ += X.shape[0]

	def _initialize(self, X_pairwise, idxs=None):
		super(BaseGraphSelection, self)._initialize(X_pairwise, idxs=idxs)

	def _calculate_gains(self, X_pairwise):
		super(BaseGraphSelection, self)._calculate_gains(X_pairwise)

	def _calculate_sieve_gains(self, X, thresholds, idxs):
		super(BaseGraphSelection, self)._calculate_sieve_gains(X, thresholds,
			idxs)

	def _select_next(self, X_pairwise, gain, idx):
		super(BaseGraphSelection, self)._select_next(X_pairwise, gain, idx)
