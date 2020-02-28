# maxCoverage.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 
	
import numpy
from .featureBased import FeatureBasedSelection

class MaxCoverageSelection(FeatureBasedSelection):
	"""A selector based off a coverage function.

	Maximum coverage functions aim to maximize the number of features that 
	have a non-zero element in at least one selected example---there is no 
	marginal benefit to observing a variable in two examples. If each variable 
	is thought to be an item in a set, and the data is a binary matrix where a 
	1 indicates the item is present in the example and 0 indicates it is not, 
	optimizing a maximum coverage function is a solution to the set coverage 
	problem. These functions are useful when the space of variables is massive 
	and each example only sees a small subset of them, which is a common 
	situation when analyzing text data when the variables are words. The 
	maximum coverage function is an instance of a feature-based function 
	when the concave function is minimum.

	.. note::
		All values in your data must be binary for this selection to work.

	The general form of a coverage function is:

	.. math::
		f(X) = \\sum\\limits_{d=1}^{D} \\min \\left( \\sum\\limits_{n=1}^{N} X_{i, d}, 1 \\right) 

	where :math:`f` indicates the function that operates on a subset :math:`X` 
	that has :math:`N` examples and :math:`D` dimensions. Importantly, 
	:math:`X` is the subset and not the ground set, meaning that the time it 
	takes to evaluate this function is proportional only to the size of the 
	selected subset and not the size of the full data set, like it is for
	 graph-based functions.  

	See https://www2.cs.duke.edu/courses/fall17/compsci632/scribing/scribe2.pdf
	where the problem is described as maximum coverage.

	Parameters
	----------
	n_samples : int
		The number of examples to return.

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

	def __init__(self, n_samples, initial_subset=None, optimizer='two-stage', 
		optimizer_kwds={}, n_jobs=1, random_state=None, verbose=False):

		super(MaxCoverageSelection, self).__init__(n_samples=n_samples, 
			concave_func='min', initial_subset=initial_subset, 
			optimizer=optimizer, optimizer_kwds=optimizer_kwds, n_jobs=n_jobs, 
			random_state=random_state, verbose=verbose) 
