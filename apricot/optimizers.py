# optimizers.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import copy
import numpy
import scipy

from tqdm import tqdm

from .utils import PriorityQueue
from .utils import check_random_state
from .utils import _calculate_pairwise_distances

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
	function : base.BaseSelection
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


class NaiveGreedy(BaseOptimizer):
	"""The naive greedy algorithm for optimization.

	The naive greedy algorithm is the simplest greedy approach for optimizing 
	submodular functions. The approach simply iterates through each example in 
	the ground set that has not already been selected and calculates the gain 
	in function value that would result from adding that example to the 
	selected set. This process is embarassingly parallel and so is extremely 
	amenable both to parallel processing and distributed computing. Further, 
	because it is conceptually simple, it is also simple to implement.

	The naive greedy algorithm can be specified for any function by passing in 
	`optimizer='naive'` to the relevant selector object. Here is an example of 
	specifying the naive greedy algorithm for optimizing a feature-based function.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='naive')
		selector.fit(X)

	Parameters
	----------
	self.function : base.BaseSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.BaseSelection
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

	The lazy (or accelerated) greedy algorithm is a fast alternate to the 
	naive greedy algorithm that results in the same items being selected. 
	This algorithm exploits the diminishing returns property of submodular 
	functions in order to avoid re-evaluating examples that are known to 
	provide little gain. If an example has a small gain relative to other 
	examples, it is unlikely to be the next selected example because that 
	gain can only go down as more items are selected. Thus, the example 
	should only be re-evaluated once the gains of other examples have gotten 
	smaller.

	The key idea of the lazy greedy algorithm is to maintain a priority queue 
	where the examples are the elements in the queue and the priorities are 
	the gains the last time they were evaluated. The algorithm has two steps. 
	The first step is to calculate the gain that each example would have if 
	selected first (the modular upper bound) and populate the priority queue 
	using these values. The second step is to recalculate the gain of the 
	first example in the priority queue and then add the example back into 
	the queue. If the example remains at the front of the queue it is selected 
	because no other example could have a larger gain once re-evaluated (due 
	to the diminishing returns property).

	While the worst case time complexity of this algorithm is the same as the 
	naive greedy algorithm, in practice it can be orders of magnitude faster. 
	Empirically, it appears to accelerate graph-based functions much more 
	than it does feature-based ones. Functions also seem to be more 
	accelerated the more curved they are.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='lazy')
		selector.fit(X)

	Parameters
	----------
	self.function : base.BaseSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.BaseSelection
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
				elif gain == best_gain and best_gain == 0.0:
					best_gain = gain
					best_idx = idx
					break

			cost += sample_cost[best_idx]
			best_gain *= sample_cost[best_idx]
			self.function._select_next(X[best_idx], best_gain, best_idx)

			if self.verbose == True:
				self.function.pbar.update(1)

class ApproximateLazyGreedy(BaseOptimizer):
	"""The approximate lazy/accelerated greedy algorithm for optimization.

	The approximate lazy greedy algorithm is a simple extension of the lazy 
	greedy algorithm that, rather than requiring that an element remains at 
	the top of the priority queue after being re-evaluated, only requires 
	that the gain is within a certain user-defined percentage of the best 
	gain to be selected. The key point in this approach is that finding the 
	very best element while maintaining the priority queue may be expensive, 
	but finding elements that are good enough is simple. While the best 
	percentage to use is data set specific, even values near 1 can lead to 
	large savings in computation.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='approximate-lazy')
		selector.fit(X)

	Parameters
	----------
	self.function : base.BaseSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.BaseSelection
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
	"""An approach that switches between two optimizers midway through.

	The two-stage greedy optimizer is a general purpose framework for combining 
	two optimizers by making the first :math:`n` selections out of :math:`k` 
	total selections using one optimizer, and then making the remainder using 
	the other. When the first optimizer is random selection and the second 
	approach is naive/lazy greedy, this becomes partial enumeration. By 
	default, the first algorithm is the naive greedy optimizer and the second 
	algorithm is the lazy greedy. This combination results in the same 
	selection as either optimizer individually but replaces the 
	computationally intensive first few steps for the priority queue, where 
	the algorithm may require scanning through almost the entire queue, with 
	the parallelizable naive greedy algorithm. While, in theory, the lazy 
	greedy algorithm will never perform more function calls than the naive 
	greedy algorithm, there are costs associated both with maintaining a 
	priority queue and with evaluating a single example instead of a batch 
	of examples.

	This optimizer, with the naive greedy optimizer first and the lazy greedy 
	optimizer second, is the default optimizer for apricot selectors.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='two-stage')
		selector.fit(X)

	Parameters
	----------
	self.function : base.BaseSelection
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
	self.function : base.BaseSelection
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

		optimizer1.select(X, min(self.n_first_selections, k), sample_cost=sample_cost)
		if k > self.n_first_selections:
			m = k - self.n_first_selections
			optimizer2.select(X, m, sample_cost=sample_cost)


class StochasticGreedy(BaseOptimizer):
	"""The stochastic greedy algorithm for optimization.

	The stochastic greedy algorithm is a simple approach that, for each 
	iteration, randomly selects a subset of data and then finds the best next 
	example within that subset. The distinction between this approach and the 
	sample greedy algorithm is that this subset changes at each iteration, 
	meaning that the algorithm does cover the entire data set. In contrast, 
	the sample greedy algorithm is equivalent to manually subsampling the data 
	before running a selector on it. The size of this subset is proportional 
	to the number of examples that are chosen and determined in a manner that 
	results in the same amount of computation being done no matter how many 
	elements are selected. A key idea from this approach is that, while the 
	exact ranking of the elements may differ from the naive/lazy greedy 
	approaches, the set of selected elements is likely to be similar despite 
	the limited amount of computation.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='stochastic')
		selector.fit(X)

	Parameters
	----------
	self.function : base.BaseSelection
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
	self.function : base.BaseSelection
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
	"""The sample greedy algorithm for optimization.

	The sample greedy algorithm is a simple approach that subsamples the full 
	data set with a user-defined sampling probability and then runs an 
	optimization on that subset. This subsampling can lead to obvious speed 
	improvements because fewer elements as selected, but will generally find 
	a lower quality subset because fewer elements are present. This approach 
	is typically used a baseline for other approaches but can save a lot of 
	time on massive data sets that are known to be highly redundant.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='sample')
		selector.fit(X)

	Parameters
	----------
	self.function : base.BaseSelection
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
	self.function : base.BaseSelection
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
	"""An approach for optimizing submodular functions on massive data sets.

	GreeDi is an optimizer that was designed to work on data sets that are 
	too large to fit into memory. The approach involves first partitioning 
	the data into :math:`m` equally sized chunks without any overlap. Then, 
	:math:`l` elements are selected from each chunk using a standard optimizer 
	like naive or lazy greedy. Finally, these :math:`ml` examples are merged 
	and a standard optimizer selects :math:`k` examples from this set. In 
	this manner, the algorithm sacrifices exactness to ensure that memory 
	limitations are not an issue. 

	There are a few considerations to keep in mind when using GreeDi. 
	Naturally, :math:`ml` must both be larger than :math:`k` and also small 
	enough to fit into memory. The larger :math:`l`, the closer the solution 
	is to the exact solution but also the more compute time is required. 
	Conversely, the larger :math:`m` is, the less exact the solution is. 
	When using a graph-based function, increasing :math:`m` can dramatically 
	reduce the amount of computation that needs to be performed, as well as 
	the memory requirements, because the similarity matrix becomes smaller 
	in size. However, feature-based functions are likely to see less of a 
	speed improvement because the cost of evaluating an example is independent 
	of the size of ground set.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='greedi')
		selector.fit(X)

	Parameters
	----------
	function : base.BaseSelection
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
	self.function : base.BaseSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.BaseSelection
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
	"""An approach that approximates a submodular function as modular.

	This approach approximates the submodular function by using its modular 
	upper-bounds to do the selection. Essentially, a defining characteristic 
	of submodular functions is the *diminishing returns* property where the 
	gain of an example decreases with the number of selected examples. In 
	contrast, modular functions have constant gains for examples regardless 
	of the number of selected examples. Thus, approximating the submodular 
	function as a modular function can serve as an upper-bound to the gain 
	for each example during the selection process. This approximation makes 
	the function simple to optimize because one would simply calculate the 
	gain that each example yields before any examples are selected, sort the 
	examples by this gain, and select the top :math:`k` examples. While this 
	approach is fast, this approach is likely best paired with a traditional 
	optimization algorithm after the first few examples are selected.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='modular')
		selector.fit(X)

	Parameters
	----------
	self.function : base.BaseSelection
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
	self.function : base.BaseSelection
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


class BidirectionalGreedy(BaseOptimizer):
	"""The bidirectional greedy algorithm.

	Most submodular optimizers assume that the function is *monotone*, i.e., 
	that the gain from each successive example is positive. However, there 
	are some cases where the key diminishing returns property holds, but 
	the gains are not necessarily positive. The most obvious of these is a
	difference in submodular functions. In these cases, the naive greedy 
	algorithm is not guaranteed to return a good result. 

	The bidirectional greedy algorithm was developed to optimize non-monotone 
	submodular functions. While it has a guarantee that is lower than the 
	naive greedy algorithm has for monotone functions, it generally returns 
	better sets than the greedy algorithm.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='bidirectional')
		selector.fit(X)

	Parameters
	----------
	self.function : base.BaseSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.BaseSelection
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
					if A.sum() == k:
						return

				else:
					B[i] = False


class SieveGreedy(BaseOptimizer):
	"""The sieve stream greedy algorithm.

	Most submodular optimizers assume that the function is *monotone*, i.e., 
	that the gain from each successive example is positive. However, there 
	are some cases where the key diminishing returns property holds, but 
	the gains are not necessarily positive. The most obvious of these is a
	difference in submodular functions. In these cases, the naive greedy 
	algorithm is not guaranteed to return a good result. 

	The bidirectional greedy algorithm was developed to optimize non-monotone 
	submodular functions. While it has a guarantee that is lower than the 
	naive greedy algorithm has for monotone functions, it generally returns 
	better sets than the greedy algorithm.

	.. code::python

		from apricot import FeatureBasedSelection

		X = numpy.random.randint(10, size=(10000, 100))

		selector = FeatureBasedSelection(100, 'sqrt', optimizer='bidirectional')
		selector.fit(X)

	Parameters
	----------
	self.function : base.BaseSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.


	Attributes
	----------
	self.function : base.BaseSelection
		A submodular function that implements the `_calculate_gains` and
		`_select_next` methods. This is the function that will be
		optimized.

	self.verbose : bool
		Whether to display a progress bar during the optimization process.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function=None, epsilon=0.01, random_state=None, 
		n_jobs=None, verbose=False):
		self.epsilon = epsilon
		self.n_seen_ = 0
		self.thresholds = [1]
		self.max_gain = -1

		super(SieveGreedy, self).__init__(function=function, 
			random_state=random_state, n_jobs=n_jobs, verbose=verbose)

	def select(self, X, k, sample_cost=None):
		"""Select elements in a naive greedy manner."""

		# This is not a great use of the apricot API. An optimizer shouldn't
		# be using its own special gain calculating function. However, it is
		# much faster to do things this way so I'll keep it in for now.

		n, d = X.shape
		if sample_cost is None:
			sample_cost = numpy.ones(n, dtype='float64')

		r = X.shape[1] if self.function.reservoir is not None else 1
		marginal_gains = self.function._calculate_gains(X) / sample_cost / r
		max_marginal_gain = marginal_gains.max()
		if max_marginal_gain > self.max_gain:
			self.max_gain = max_marginal_gain
			j = len(self.thresholds)
			threshold = (1 + self.epsilon) ** j - 1
			while threshold <= self.max_gain * k:
				self.thresholds.append(threshold)

				j += 1
				threshold = (1 + self.epsilon) ** j - 1

		thresholds = numpy.array(self.thresholds, dtype='float64')
		idxs = numpy.arange(n, dtype='int64') + self.n_seen_

		self.function._calculate_sieve_gains(X, thresholds, idxs)

		best_idx = numpy.argmax(self.function.sieve_total_gains_)
		ranking = self.function.sieve_selections_[best_idx]
		ranking = ranking[:self.function.sieve_n_selected_[best_idx]]
		gain = self.function.sieve_gains_[best_idx]
		gain = gain[:self.function.sieve_n_selected_[best_idx]]

		for i in range(len(self.thresholds)):
			for j in range(k):
				m = self.function.sieve_selections_[i, j]
				if m >= self.n_seen_:
					if isinstance(self.function._X, scipy.sparse.csr_matrix):
						self.function.sieve_subsets_[i, j] = self.function._X[m - self.n_seen_].toarray()[0]
					else:
						self.function.sieve_subsets_[i, j] = self.function._X[m - self.n_seen_]

		self.function.ranking = ranking
		self.function.gains = gain
		self.function.subset = self.function.sieve_subsets_[best_idx]
		self.n_seen_ += X.shape[0]


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
	'bidirectional' : BidirectionalGreedy,
	'sieve' : SieveGreedy,
}
