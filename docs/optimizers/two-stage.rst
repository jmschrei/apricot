.. _optimizers.two-stage

Two-Stage Greedy
================

The two-stage greedy optimizer is a general purpose framework for combining two optimizers by making the first :math:`n` selections out of :math:`k` total selections using one optimizer, and then making the remainder using the other. When the first optimizer is random selection and the second approach is naive/lazy greedy, this becomes partial enumeration. By default, the first algorithm is the naive greedy optimizer and the second algorithm is the lazy greedy. This combination results in the same selection as either optimizer individually but replaces the computationally intensive first few steps for the priority queue, where the algorithm may require scanning through almost the entire queue, with the parallelizable naive greedy algorithm. While, in theory, the lazy greedy algorithm will never perform more function calls than the naive greedy algorithm, there are costs associated both with maintaining a priority queue and with evaluating a single example instead of a batch of examples.

This optimizer, with the naive greedy optimizer first and the lazy greedy optimizer second, is the default optimizer for apricot selectors.

.. code::python

	from apricot import FeatureBasedSelection

	X = numpy.random.randint(10, size=(10000, 100))

	selector = FeatureBasedSelection(100, 'sqrt', optimizer='two-stage')
	selector.fit(X)


API Reference
-------------

.. automodule:: apricot.optimizers.TwoStageGreedy
	:members:
	:inherited-members: