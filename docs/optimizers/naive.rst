.. _optimizers.naive

Naive Greedy
===============

The naive greedy algorithm is the simplest greedy approach for optimizing submodular functions. The approach simply iterates through each example in the ground set that has not already been selected and calculates the gain in function value that would result from adding that example to the selected set. This process is embarassingly parallel and so is extremely amenable both to parallel processing and distributed computing. Further, because it is conceptually simple, it is also simple to implement.

The naive greedy algorithm can be specified for any function by passing in `optimizer='naive'` to the relevant selector object. Here is an example of specifying the naive greedy algorithm for optimizing a feature-based function.

.. code::python

	from apricot import FeatureBasedSelection

	X = numpy.random.randint(10, size=(10000, 100))

	selector = FeatureBasedSelection(100, 'sqrt', optimizer='naive')
	selector.fit(X)


API Reference
-------------

.. automodule:: apricot.optimizers.NaiveGreedy
	:members:
	:inherited-members: