# base.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 

"""
This file contains code that implements the core of the submodular selection
algorithms.
"""

try:
	import cupy
except:
	import numpy as cupy

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
from ..optimizers import OPTIMIZERS

from ..utils import PriorityQueue

from scipy.sparse import csr_matrix

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsTransformer

def _calculate_pairwise_distances(X, metric, n_neighbors=None):
	if metric in ('precomputed', 'ignore'):
		return X

	if n_neighbors is None:
		if metric == 'euclidean':
			X_pairwise = pairwise_distances(X, metric=metric, squared=True)
		elif metric == 'correlation' or metric == 'cosine':
			# An in-place version of:
			# X_pairwise = 1 - (1 - pairwise_distances(X, metric=metric)) ** 2
			
			X_pairwise = pairwise_distances(X, metric=metric)
			X_pairwise = numpy.subtract(1, X_pairwise, out=X_pairwise)
			X_pairwise = numpy.square(X_pairwise, out=X_pairwise)
			X_pairwise = numpy.subtract(1, X_pairwise, out=X_pairwise)
		else:
			X_pairwise = pairwise_distances(X, metric=metric)
	else:
		if metric == 'correlation' or metric == 'cosine':
			# An in-place version of:
			# X = 1 - (1 - pairwise_distances(X, metric=metric)) ** 2

			X = pairwise_distances(X, metric=metric)
			X = numpy.subtract(1, X, out=X)
			X = numpy.square(X, out=X)
			X = numpy.subtract(1, X, out=X)
			metric = 'precomputed'

		if isinstance(n_neighbors, int):
			params = {'squared': True} if metric == 'euclidean' else None
			X_pairwise = KNeighborsTransformer(
				n_neighbors=n_neighbors, metric=metric, 
				metric_params=params).fit_transform(X)

		elif isinstance(n_neighbors, KNeighborsTransformer):
			X_pairwise = n_neighbors.fit_transform(X)

	if metric == 'correlation' or metric == 'cosine':
		if isinstance(X_pairwise, csr_matrix):
			X_pairwise.data = numpy.subtract(1, X_pairwise.data, 
				out=X_pairwise.data)
		else:
			X_pairwise = numpy.subtract(1, X_pairwise,
				out=X_pairwise)
	else:
		if isinstance(X_pairwise, csr_matrix):
			X_pairwise.data = numpy.subtract(X_pairwise.max(),
				X_pairwise.data, out=X_pairwise.data)
		else:
			X_pairwise = numpy.subtract(X_pairwise.max(), X_pairwise,
				out=X_pairwise)

	return numpy.array(X_pairwise, copy=False, dtype='float64')


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

	def __init__(self, n_samples, initial_subset=None, optimizer='two-stage', 
		optimizer_kwds={}, n_jobs=1, random_state=None, verbose=False):
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
		self.random_state = random_state
		self.optimizer = optimizer
		self.optimizer_kwds = optimizer_kwds
		self.n_jobs = n_jobs
		self.verbose = verbose
		self.ranking = None
		self.idxs = None
		self.gains = None
		self.sparse = None
		self.cupy = None
		self.initial_subset = initial_subset

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

		allowed_dtypes = list, numpy.ndarray, csr_matrix, cupy.ndarray

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

		self._initialize(X)

		if not self.sparse and not self.cupy:
			if X.dtype != 'float64':
				X = X.astype('float64')

		if isinstance(self.optimizer, str):
			optimizer = OPTIMIZERS[self.optimizer](function=self, 
				verbose=self.verbose, random_state=self.random_state,
				**self.optimizer_kwds)
		else:
			optimizer = self.optimizer

		if self.verbose:
			self.pbar = tqdm(total=self.n_samples, unit_scale=True)

		optimizer.select(X, self.n_samples, sample_cost=sample_cost)

		if self.verbose == True:
			self.pbar.close()

		self.ranking = numpy.array(self.ranking)
		self.gains = numpy.array(self.gains)
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
		self.sparse = isinstance(X, csr_matrix)
		self.cupy = isinstance(X, cupy.ndarray) and not isinstance(X, numpy.ndarray)
		self.ranking = []
		self.gains = []

		if self.cupy:
			self.current_values = cupy.zeros(X.shape[1], dtype='float64')
			self.current_concave_values = cupy.zeros(X.shape[1], dtype='float64')
			self.mask = cupy.zeros(X.shape[0], dtype='int8')
		else:
			self.current_values = numpy.zeros(X.shape[1], dtype='float64')
			self.current_concave_values = numpy.zeros(X.shape[1], dtype='float64')
			self.mask = numpy.zeros(X.shape[0], dtype='int8')

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

	def _select_next(self, X, gain, idx):
		self.ranking.append(idx)
		self.gains.append(gain)
		self.mask[idx] = True
		self.idxs = numpy.where(self.mask == 0)[0]


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

	def __init__(self, n_samples=10, metric='euclidean', 
		initial_subset=None, optimizer='two-stage', optimizer_kwds={},
		n_neighbors=None, n_jobs=1, random_state=None, verbose=False):
		
		self.metric = metric.replace("corr", "correlation")
		self.n_neighbors = n_neighbors

		super(BaseGraphSelection, self).__init__(n_samples=n_samples, 
			initial_subset=initial_subset, optimizer=optimizer, 
			optimizer_kwds=optimizer_kwds, n_jobs=n_jobs, 
			random_state=random_state, verbose=verbose)

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
	
		return super(BaseGraphSelection, self).fit(X_pairwise, y=y,
			sample_weight=sample_weight, sample_cost=sample_cost)

	def _initialize(self, X_pairwise, idxs=None):
		super(BaseGraphSelection, self)._initialize(X_pairwise, idxs=idxs)

	def _calculate_gains(self, X_pairwise):
		super(BaseGraphSelection, self)._calculate_gains(X_pairwise)

	def _select_next(self, X_pairwise, gain, idx):
		super(BaseGraphSelection, self)._select_next(X_pairwise, gain, idx)
