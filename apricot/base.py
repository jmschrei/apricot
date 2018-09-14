# base.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 

"""
This file contains code that implements the core of the submodular selection
algorithms.
"""

from .utils import PriorityQueue

from tqdm import tqdm

class SubmodularSelection(object):
	"""The base selection object.

	This object defines the structures that all submodular selection algorithms
	should follow. All algorithms will have the same public methods and the
	same attributes.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

	concave_func : str or callable
		The type of concave function to apply to the feature values. You can
		pass in your own function to apply. Otherwise must be one of the
		following:

			'log' : log(1 + X)
			'sqrt' : sqrt(X)
			'min' : min(X, 1)
			'inverse' : 1 / (1 + X)

	verbose : bool
		Whether to print output during the selection process.

	Attributes
	----------
	pq : PriorityQueue
		The priority queue used to implement the lazy greedy algorithm.

	n_samples : int
		The number of samples to select.

	ranking : numpy.array int
		The selected samples in the order of their gain with the first number in
		the ranking corresponding to the index of the first sample that was
		selected by the greedy procedure.
	"""

	def __init__(self, n_samples, verbose=False):
		self.pq = PriorityQueue()

		if type(n_samples) != int:
			raise ValueError("n_samples must be a positive integer.")
		if n_samples < 1:
			raise ValueError("n_samples must be a positive integer.")

		self.n_samples = n_samples
		self.verbose = verbose
		self.ranking = None
	
	def fit(self, X, y=None):
		"""Fit a ranking to the data set of the top n_sample elements.

		This method will take in the data set (and optionally a label) set
		and fit a ranking to it. This runs the greedy select procedure n_samples
		times, selecting at each iteration the next best element. The ranking
		is then stored for future use.

		Parameters
		----------
		X : list or numpy.ndarray, shape=(n, d)
			The data set to transform. Must be numeric.

		y : list or numpy.ndarray, shape=(n,), optional
			The labels to transform. If passed in this function will return
			both the data and th corresponding labels for the rows that have
			been selected.

		Returns
		-------
		self : SubmodularSelection
		"""

		raise NotImplementedError

	def transform(self, X, y=None):
		"""Transform the data set by selecting the top n_samples samples.

		This method will use the fit ranking to transform the data set by
		returning only the top n_samples elements as defined by the
		ranking. The data will be returned in the order that it was selected
		during the selection process.

		Parameters
		----------
		X : list or numpy.ndarray, shape=(n, d)
			The data set to transform. Must be numeric.

		y : list or numpy.ndarray, shape=(n,), optional
			The labels to transform. If passed in this function will return
			both the data and th corresponding labels for the rows that have
			been selected.

		Returns
		-------
		X_subset : numpy.ndarray, shape=(n_samples, d)
			A subset of the data such that n_samples < n and n_samples is the
			integer provided at initialization.

		y_subset : numpy.ndarray, shape=(n_samples,)
			The labels that match with the indices of the samples if y is
			passed in. Only returned if passed in.
		"""

		if y is not None:
			return X[self.ranking], y[self.ranking]
		return X[self.ranking]

	def fit_transform(self, X, y=None):
		"""Perform selection and return the subset of the data set.

		This method will take in a full data set and return the selected subset
		according to the feature based function. The data will be returned in
		the order that it was selected, with the first row corresponding to
		the best first selection, the second row corresponding to the second
		best selection, etc.

		Parameters
		----------
		X : list or numpy.ndarray, shape=(n, d)
			The data set to transform. Must be numeric.

		y : list or numpy.ndarray, shape=(n,), optional
			The labels to transform. If passed in this function will return
			both the data and th corresponding labels for the rows that have
			been selected.

		Returns
		-------
		X_subset : numpy.ndarray, shape=(n_samples, d)
			A subset of the data such that n_samples < n and n_samples is the
			integer provided at initialization.

		y_subset : numpy.ndarray, shape=(n_samples,)
			The labels that match with the indices of the samples if y is
			passed in. Only returned if passed in.
		"""

		return self.fit(X, y).transform(X, y)
