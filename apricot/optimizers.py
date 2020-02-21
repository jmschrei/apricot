# optimizers.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy

from tqdm import tqdm

from .utils import PriorityQueue
from .utils import check_random_state

class BaseOptimizer(object):
	"""An approach for optimizing submodular functions.

	This object contains a single method, `select`, that implements an
	algorithm for optimizing submodular functions. This method will take in the
	data set, the number of examples to select, and a stored reference to the
	submodular function to be optimized, and yield a subset of examples. The
	method operates on the gains provided in the `_calculate_gains` method from
	the submodular function object and stores examples using the `_select_next`
	method.

	Parameters
	----------
	function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.BaseSelection or None
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized. If None, will be set by the selector when passed in.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.
	"""

	def __init__(self, function=None, random_state=None, n_jobs=None, 
		verbose=False):
		self.function = function
		self.random_state = check_random_state(random_state)
		self.n_jobs = n_jobs
		self.verbose = verbose
		self.gains_ = None

	def select(self, X, k):
		raise NotImplementedError


class RandomGreedy(BaseOptimizer):
	"""The naive greedy algorithm for optimization.

	This optimization approach is the naive greedy algorithm. At each iteration
	of selection it will simply calculate the gain one would get from adding
	each example, and then will select the example that has the highest gain.
	This algorithm is conceptually simple and easy to parallelize and put on a
	GPU, but can be slower than other alternatives because it involves
	repeatedly evaluating examples that are not likely to be selected next.

	Parameters
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.
	"""

	def __init__(self, function=None, random_state=None, n_jobs=None, 
		verbose=False):
		super(NaiveGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		cost = 0.0
		if sample_cost is None:
			sample_cost = numpy.ones(X.shape[0], dtype='float64')

		i = 0
		while cost < k:
			idx = self.random_state.choice(self.random.idxs)
		
			if cost + sample_cost[idx] > k:
				continue

			cost += sample_cost[idx]
			gain = self.function._calculate_gains(X, idxs=numpy.array([idx]))[0]

			self.function._select_next(X[idx], gain, idx)

			if self.verbose == True:
				self.function.pbar.update(round(sample_cost[idx], 2))


class ModularGreedy(BaseOptimizer):
	"""The stochastic greedy algorithm for optimization.

	This optimization approach is the stochastic greedy algorithm proposed by
	Mirzasoleiman et al. (https://las.inf.ethz.ch/files/mirzasoleiman15lazier.pdf).
	This approach is conceptually similar to the naive greedy algorithm except
	that it only evaluates a subset of examples at each iteration. Thus, it is
	easy to parallelize and amenable to acceleration using a GPU while 
	maintaining nice theoretical guarantees.

	Parameters
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	epsilon : float, optional
		The inverse of the sampling probability of any particular point being 
		included in the subset, such that 1 - epsilon is the probability that
		a point is included. Default is 0.9.

	random_state : int or RandomState or None, optional
		The random seed to use for the random selection process.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function=None, random_state=None, n_jobs=None, 
		verbose=False):
		super(ModularGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		"""Select elements in a naive greedy manner."""

		cost = 0.0
		if sample_cost is None:
			sample_cost = numpy.ones(X.shape[0], dtype='float64')

		gains = self.function._calculate_gains(X) / sample_cost[self.function.idxs]
		idxs = gains.argsort()[::-1]

		for idx in idxs:
			if cost + sample_cost[idx] > k:
				continue

			cost += sample_cost[idx]
			gain = self.function._calculate_gains(X, idxs=numpy.array([idx]))[0]

			self.function._select_next(X[idx], gain, idx)

			if self.verbose == True:
				self.function.pbar.update(1)


class NaiveGreedy(BaseOptimizer):
	"""The naive greedy algorithm for optimization.

	This optimization approach is the naive greedy algorithm. At each iteration
	of selection it will simply calculate the gain one would get from adding
	each example, and then will select the example that has the highest gain.
	This algorithm is conceptually simple and easy to parallelize and put on a
	GPU, but can be slower than other alternatives because it involves
	repeatedly evaluating examples that are not likely to be selected next.

	Parameters
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function=None, random_state=None, n_jobs=None, 
		verbose=False):
		super(NaiveGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		cost = 0.0
		if sample_cost is None:
			sample_cost = numpy.ones(X.shape[0], dtype='float64')

		while cost < k:
			gains = self.function._calculate_gains(X) / sample_cost[self.function.idxs]
			idxs = numpy.lexsort((numpy.arange(gains.shape[0]), -gains))

			for idx in idxs:
				best_idx = self.function.idxs[idx]
				if cost + sample_cost[best_idx] <= k:
					break
			else:
				break

			cost += sample_cost[best_idx]
			gain = gains[idx] * sample_cost[best_idx]
			self.function._select_next(X[best_idx], gain, best_idx)

			if self.verbose == True:
				self.function.pbar.update(round(sample_cost[best_idx], 1))


class LazyGreedy(BaseOptimizer):
	"""The lazy/accelerated greedy algorithm for optimization.

	This optimization approach is the lazy/accelerated greedy algorithm. It
	will return the same subset as the naive greedy algorithm, but it uses a
	priority queue to store the examples to prevent the repeated evaluation
	of examples that are unlikely to be the next selected one. The benefit
	of using this approach are that using a priority queue can significantly
	improve the speed of optimization, but the downsides are that maintaining
	a priority queue can be costly and that it's difficult to parallelize the
	approach or put it on a GPU.

	Parameters
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.

	self.pq : utils.PriorityQueue
		The priority queue used to order examples for evaluation.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function=None, random_state=None, n_jobs=None, 
		verbose=False):
		super(LazyGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		cost = 0.0
		if sample_cost is None:
			sample_cost = numpy.ones(X.shape[0], dtype='float64')

		gains = self.function._calculate_gains(X) / sample_cost[self.function.idxs]
		self.pq = PriorityQueue(self.function.idxs, -gains)

		while cost < k:
			best_gain = float("-inf")
			best_idx = None
			
			while True:
				if len(self.pq.pq) == 0:
					return

				prev_gain, idx = self.pq.pop()
				prev_gain = -prev_gain

				if cost + sample_cost[idx] > k:
					continue

				if best_idx == idx:
					break
				
				idxs = numpy.array([idx])
				gain = self.function._calculate_gains(X, idxs)[0] / sample_cost[idx]
				self.pq.add(idx, -gain)

				if gain > best_gain:
					best_gain = gain
					best_idx = idx

			cost += sample_cost[best_idx]
			best_gain *= sample_cost[best_idx]
			self.function._select_next(X[best_idx], best_gain, best_idx)

			if self.verbose == True:
				self.function.pbar.update(1)

class ApproximateLazyGreedy(BaseOptimizer):
	"""The approximate lazy/accelerated greedy algorithm for optimization.

	This optimization approach is the lazy/accelerated greedy algorithm. It
	will return the same subset as the naive greedy algorithm, but it uses a
	priority queue to store the examples to prevent the repeated evaluation
	of examples that are unlikely to be the next selected one. The benefit
	of using this approach are that using a priority queue can significantly
	improve the speed of optimization, but the downsides are that maintaining
	a priority queue can be costly and that it's difficult to parallelize the
	approach or put it on a GPU.

	Parameters
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.

	self.pq : utils.PriorityQueue
		The priority queue used to order examples for evaluation.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function=None, beta=0.9, random_state=None, 
		n_jobs=None, verbose=False):
		self.beta = beta
		super(ApproximateLazyGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		cost = 0.0
		if sample_cost is None:
			sample_cost = numpy.ones(X.shape[0], dtype='float64')

		gains = self.function._calculate_gains(X) / sample_cost[self.function.idxs]
		self.pq = PriorityQueue(self.function.idxs, -gains)

		while cost < k:
			while True:
				if len(self.pq.pq) == 0:
					return

				prev_gain, idx = self.pq.pop()
				prev_gain = -prev_gain
				
				if cost + sample_cost[idx] > k:
					continue

				idxs = numpy.array([idx])
				gain = self.function._calculate_gains(X, idxs)[0] / sample_cost[idx]

				if gain >= self.beta * prev_gain:
					best_gain = gain
					best_idx = idx
					break
				else:
					self.pq.add(idx, -gain)


			cost += sample_cost[best_idx]
			best_gain *= sample_cost[best_idx]
			self.function._select_next(X[best_idx], best_gain, best_idx)

			if self.verbose == True:
				self.function.pbar.update(1)


class TwoStageGreedy(BaseOptimizer):
	"""An approach that uses both naive and lazy greedy algorithms.

	This optimization approach starts off by using the naive greedy algorithm
	to select the first few examples and then switches to use the lazy greedy
	algorithm. This two stage procedure is designed to overcome the limitations
	of the lazy greedy algorithm---specifically, the inability to parallelize
	the implementation. Additionally, early iterations frequently require
	iterating over most of the examples, which is particularly costly to do
	on a priority queue. Thus, one can frequently see large speed gains simply
	by doing the first few iterations using the naive greedy algorithm and
	then switching to the lazy greedy algorithm.

	Parameters
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.n_first_selections : int
		The number of selections to perform using the naive greedy algorithm
		before populating the priority queue and using the lazy greedy
		algorithm.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.
	"""

	def __init__(self, function=None, n_first_selections=10, 
		optimizer1='naive', optimizer2='lazy', random_state=None,
		n_jobs=None, verbose=False):
		self.n_first_selections = n_first_selections
		self.optimizer1 = optimizer1
		self.optimizer2 = optimizer2
		super(TwoStageGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		if isinstance(self.optimizer1, str):
			optimizer1 = OPTIMIZERS[self.optimizer1](function=self.function,
				verbose=self.verbose)
		elif not isinstance(self.optimizer1, BaseOptimizer):
			raise ValueError("optimizer1 must be either a string or an " /
				"optimized object.")
		else:
			optimizer1 = self.optimizer1

		if isinstance(self.optimizer2, str):
			optimizer2 = OPTIMIZERS[self.optimizer2](function=self.function, 
				verbose=self.verbose)
		elif not isinstance(self.optimizer2, BaseOptimizer):
			raise ValueError("optimizer1 must be either a string or an " /
				"optimized object.")
		else:
			optimizer2 = self.optimizer2

		optimizer1.select(X, self.n_first_selections, sample_cost=sample_cost)
		if k > self.n_first_selections:
			m = k - self.n_first_selections
			optimizer2.select(X, m, sample_cost=sample_cost)


class StochasticGreedy(BaseOptimizer):
	"""The stochastic greedy algorithm for optimization.

	This optimization approach is the stochastic greedy algorithm proposed by
	Mirzasoleiman et al. (https://las.inf.ethz.ch/files/mirzasoleiman15lazier.pdf).
	This approach is conceptually similar to the naive greedy algorithm except
	that it only evaluates a subset of examples at each iteration. Thus, it is
	easy to parallelize and amenable to acceleration using a GPU while 
	maintaining nice theoretical guarantees.

	Parameters
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	epsilon : float, optional
		The inverse of the sampling probability of any particular point being 
		included in the subset, such that 1 - epsilon is the probability that
		a point is included. Default is 0.9.

	random_state : int or RandomState or None, optional
		The random seed to use for the random selection process.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function=None, epsilon=0.9, random_state=None, 
		n_jobs=None, verbose=False):
		self.epsilon = epsilon
		super(StochasticGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		cost = 0.0
		if sample_cost is None:
			sample_cost = numpy.ones(X.shape[0], dtype='float64')

		n = X.shape[0]
		subset_size = -numpy.log(self.epsilon) * n / k
		subset_size = max(int(subset_size), 1)

		while cost < k:
			idxs = self.random_state.choice(self.function.idxs, 
				replace=False, size=min(subset_size, 
					self.function.idxs.shape[0]))

			gains = self.function._calculate_gains(X, idxs) / sample_cost[idxs]
			idxs_ = numpy.lexsort((numpy.arange(gains.shape[0]), -gains))

			for idx in idxs_:
				best_idx = idxs[idx]
				if cost + sample_cost[best_idx] <= k:
					break
			else:
				return

			cost += sample_cost[best_idx]
			self.function._select_next(X[best_idx], gains[idx], best_idx)

			if self.verbose == True:
				self.function.pbar.update(1)


class SampleGreedy(BaseOptimizer):
	"""The stochastic greedy algorithm for optimization.

	This optimization approach is the stochastic greedy algorithm proposed by
	Mirzasoleiman et al. (https://las.inf.ethz.ch/files/mirzasoleiman15lazier.pdf).
	This approach is conceptually similar to the naive greedy algorithm except
	that it only evaluates a subset of examples at each iteration. Thus, it is
	easy to parallelize and amenable to acceleration using a GPU while 
	maintaining nice theoretical guarantees.

	Parameters
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	epsilon : float, optional
		The sampling probability to use when constructing the subset. A
		subset of size n * epsilon will be selected from.

	random_state : int or RandomState or None, optional
		The random seed to use for the random selection process.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function=None, epsilon=0.9, optimizer='lazy', 
		random_state=None, n_jobs=None, verbose=False):
		self.epsilon = epsilon
		self.optimizer = optimizer
		super(SampleGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		n = X.shape[0]
		subset_size = max(int(n * self.epsilon), 1)

		idxs = self.random_state.choice(self.function.idxs,
			replace=False, size=n - subset_size)

		if isinstance(self.optimizer, str):
			optimizer = OPTIMIZERS[self.optimizer](function=self.function,
				verbose=self.verbose)
		elif not isinstance(self.optimizer, BaseOptimizer):
			raise ValueError("optimizer must be either a string or an " /
				"optimizer object.")
		else:
			optimizer = self.optimizer

		self.function.mask[idxs] = 1
		self.function.idxs = numpy.where(self.function.mask == 0)[0]
		optimizer.select(X, k, sample_cost=sample_cost)


class GreeDi(BaseOptimizer):
	"""An approach for optimizing submodular functions in parallel.

	This optimizer implements the GreeDi method for selecting sets in parallel
	by Mirzasoleiman et al. (https://papers.nips.cc/paper/5039-distributed-
	submodular-maximization-identifying-representative-elements-in-massive-
	data.pdf).

	Briefly, this approach splits the data into m partitions uniformly at
	random, selects l exemplars from each partition, and then runs a second
	iteration of greedy selection on the union of l*m exemplars from each
	partition to get the top k.

	Parameters
	----------
	function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	m : int
		The number of partitions to split the data into. 

	l : int
		The number of exemplars to select from each partition. l*m must
		be larger than the number of exemplars k that will be selected
		later on.

	optimizer1 : str or base.Optimizer, optional
		The optimizer to use in the first stage of the process where l
		exemplars are selected from each partition. Default is 'lazy'.

	optimizer2 : str or base.Optimizer, optional
		The optimizer to use in the second stage where k exemplars are
		selected from the l*m exemplars returned from the first stage.
		Default is 'lazy'.

	random_state : int or RandomState or None, optional
		The random seed to use for the random selection process.

	verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.BaseSelection or None
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized. If None, will be set by the selector when passed in.

	self.m : int
		The number of partitions that the data will be split into.

	self.l : int
		The number of exemplars that will be selected from each partition.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function=None, m=None, l=None, optimizer1='lazy', 
		optimizer2='lazy', random_state=None, n_jobs=None, verbose=False):
		self.m = m
		self.l = l
		self.optimizer1 = optimizer1
		self.optimizer2 = optimizer2
		super(GreeDi, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		if self.m is None and self.l is None:
			self.m = 8
			self.l = min(k // 4, X.shape[0] // self.m)

		if sample_cost is None:
			sample_cost = numpy.ones(X.shape[0], dtype='float64')

		if k > (self.m * self.l):
			raise ValueError("k must be smaller than m * l")

		partitions = numpy.arange(X.shape[0]) % self.m
		self.random_state.shuffle(partitions)

		optimizer1 = OPTIMIZERS[self.optimizer1](function=self.function, 
			verbose=self.verbose)
		optimizer2 = OPTIMIZERS[self.optimizer2](function=self.function, 
			verbose=self.verbose)

		rankings = []

		for i in range(self.m):
			idxs = numpy.where(partitions == i)[0]

			X_subset = X[idxs]
			if X.shape[0] == X.shape[1]:
				X_subset = X_subset[:, idxs]

			self.function._initialize(X_subset)
			optimizer1.select(X_subset, self.l, sample_cost=sample_cost[idxs])

			rankings.append(idxs[self.function.ranking])

		rankings = numpy.concatenate(rankings)

		if self.verbose:
			self.function.pbar.close()
			self.function.pbar = tqdm(total=k)

		X_subset = X[rankings]
		if X.shape[0] == X.shape[1]:
			X_subset = X_subset[:, rankings]

		self.function._initialize(X_subset)
		optimizer2.select(X_subset, k, sample_cost=sample_cost[rankings])
		rankings = rankings[self.function.ranking]

		self.function._initialize(X)
		for idx in rankings:
			gain = self.function._calculate_gains(X, numpy.array([idx]))[0]
			self.function._select_next(X[idx], gain, idx)


class BidirectionalGreedy(BaseOptimizer):
	"""The bidirectional greedy algorithm.

	This is a stochastic algorithm for the optimization of submodular
	functions that are not monotone, i.e., where f(A union {b}) 
	is not necessarily greater than f(A). When these functions are not
	monotone, the greedy algorithm does not have the same good 
	guarantees on convergence, whereas the bidirectional greedy algorithm
	does.

	Generally, it is difficult to control the number of examples that
	are returned by the bidirectional greedy algorithm. This can be done
	by tuning a hyperparameter that regularizes the gains until you get
	the right set size. Here we take another approach and randomly
	select examples for consideration until we achieve the appropriate
	set size. This means that not all examples will be considered for
	addition.

	Parameters
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.SubmodularSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function=None, random_state=None, n_jobs=None, 
		verbose=False):
		super(BidirectionalGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		"""Select elements in a naive greedy manner."""

		cost = 0.0
		if sample_cost is None:
			sample_cost = numpy.ones(X.shape[0], dtype='float64')

		A = numpy.zeros(X.shape[0], dtype=bool)
		B = numpy.ones(X.shape[0], dtype=bool)

		idxs = numpy.arange(X.shape[0])
		self.random_state.shuffle(idxs)
		gains = numpy.zeros(X.shape[0])

		while True: 
			for i in idxs:
				if A[i] == True:
					continue

				self.function.initial_subset = A
				self.function._initialize(X)
				gain_a = self.function._calculate_gains(X[i])


				B[i] = False
				self.function.initial_subset = B
				self.function._initialize(X)
				gain_b = -self.function._calculate_gains(X[i])
				B[i] = True

				if gain_a == gain_b == 0.0:
					p = 0.5
				else:
					p = gain_a / (gain_a + gain_b)

				self.gains_[i] = p

				if self.random_state.uniform(0, 1) <= p:
					A[i] = True
					self.function._select_next(X[i], gain_a, i)
					if A.sum() == n:
						return

				else:
					B[i] = False

OPTIMIZERS = {
	'random' : RandomGreedy,
	'modular' : ModularGreedy,
	'naive' : NaiveGreedy,
	'lazy' : LazyGreedy,
	'approximate-lazy' : ApproximateLazyGreedy,
	'two-stage' : TwoStageGreedy,
	'stochastic' : StochasticGreedy,
	'sample' : SampleGreedy,
	'greedi' : GreeDi,
	'bidirectional' : BidirectionalGreedy
}
