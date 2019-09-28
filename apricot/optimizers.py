# optimizers.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from tqdm import tqdm
from .utils import PriorityQueue

class Optimizer(object):
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

	def __init__(self, function, verbose):
		self.function = function
		self.verbose = verbose
		self.gains_ = None

	def select(self, X, n):
		raise NotImplementedError

class NaiveGreedy(Optimizer):
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

	def __init__(self, function, verbose=False):
		super(NaiveGreedy, self).__init__(function, verbose)

	def select(self, X, n):
		"""Select elements in a naive greedy manner."""

		for i in range(n):
			self.gains_ = self.function._calculate_gains(X)
			best_idx = self.gains_.argmax()
			self.function._select_next(X[best_idx], self.gains_[best_idx], best_idx)

			if self.verbose == True:
				self.function.pbar.update(1)

class LazyGreedy(Optimizer):
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

	def __init__(self, function, verbose=False):
		self.pq = PriorityQueue()
		super(LazyGreedy, self).__init__(function, verbose)

	def select(self, X, n):
		gains = self.function._calculate_gains(X)
		for idx, gain in enumerate(gains):
			self.pq.add(idx, -gain) 

		for i in range(n):
			best_gain = 0.
			best_idx = None
			
			while True:
				prev_gain, idx = self.pq.pop()
				prev_gain = -prev_gain
				
				if best_gain >= prev_gain:
					self.pq.add(idx, -prev_gain)
					self.pq.remove(best_idx)
					break
				
				gain = self.function._calculate_gains(X[idx])

				self.pq.add(idx, -gain)
				
				if gain > best_gain:
					best_gain = gain
					best_idx = idx

			self.function._select_next(X[best_idx], best_gain, best_idx)

			if self.verbose == True:
				self.function.pbar.update(1)

class TwoStageGreedy(Optimizer):
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

	self.n_naive_selections : int
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

	self.pq : utils.PriorityQueue
		The priority queue used to order examples for evaluation.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function, n_naive_selections=10, verbose=False):
		self.n_naive_selections = n_naive_selections
		super(TwoStageGreedy, self).__init__(function, verbose)

	def select(self, X, n):
		optimizer = NaiveGreedy(self.function, self.verbose)
		optimizer.select(X, self.n_naive_selections)

		if n > self.n_naive_selections:
			optimizer = LazyGreedy(self.function, self.verbose)
			optimizer.select(X, n - self.n_naive_selections)

class TwoStageGreedy(Optimizer):
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

	self.n_naive_selections : int
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

	self.pq : utils.PriorityQueue
		The priority queue used to order examples for evaluation.

	self.gains_ : numpy.ndarray or None
		The gain that each example would give the last time that it was
		evaluated.
	"""

	def __init__(self, function, n_naive_selections=10, verbose=False):
		self.n_naive_selections = n_naive_selections
		super(TwoStageGreedy, self).__init__(function, verbose)

	def select(self, X, n):
		optimizer = NaiveGreedy(self.function, self.verbose)
		optimizer.select(X, self.n_naive_selections)

		if n > self.n_naive_selections:
			optimizer = LazyGreedy(self.function, self.verbose)
			optimizer.select(X, n - self.n_naive_selections)

class BidirectionalGreedy(Optimizer):
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

	def __init__(self, function, verbose=False):
		super(BidirectionalGreedy, self).__init__(function, verbose)

	def select(self, X, n):
		"""Select elements in a naive greedy manner."""

		A = numpy.zeros(X.shape[0], dtype=bool)
		B = numpy.ones(X.shape[0], dtype=bool)

		idxs = numpy.arange(X.shape[0])
		numpy.random.shuffle(idxs)
		self.gains_ = numpy.zeros(X.shape[0])

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

				if numpy.random.uniform(0, 1) <= p:
					A[i] = True
					self.function._select_next(X[i], gain_a, i)
					if A.sum() == n:
						return

				else:
					B[i] = False
