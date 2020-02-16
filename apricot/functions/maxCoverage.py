# maxCoverage.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com> 

"""
This file contains code that implements a selector based on the maximum
coverage submodular function.
"""
	
import numpy
from .featureBased import FeatureBasedSelection

class MaxCoverageSelection(FeatureBasedSelection):
	"""A selector based off a coverage function.

	NOTE: All values in your data must be binary for this selection to work.

	This function measures the coverage of the features in a data set. The
	approach simply counts the number of features that take a value of
	1 in at least one example in the selected set. Due to this property, it
	is likely that the function will saturate fairly quickly when selecting
	many examples unless there are also many features.

	This object can be used to solve the set coverage problem, which is to
	identify as small a set of examples as possible that cover the entire
	set of features. One would simply run this approach until the gain is 0,
	at which point all features have been covered.

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
