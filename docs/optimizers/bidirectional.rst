.. _optimizers.bidirectional

Bidirectional Greedy
====================

Most submodular optimizers assume that the function is *monotone*, i.e., that the gain from each successive example is positive. However, there are some cases where the key diminishing returns property holds, but the gains are not necessarily positive. In these cases, the naive greedy algorithm is not guaranteed to return a good result. 

The bidirectional greedy algorithm was developed to optimize non-monotone submodular functions. While it has a guarantee that is lower than the naive greedy algorithm has for monotone functions, it generally returns better sets than the greedy algorithm.

.. code::python

	from apricot import FeatureBasedSelection

	X = numpy.random.randint(10, size=(10000, 100))

	selector = FeatureBasedSelection(100, 'sqrt', optimizer='bidirectional')
	selector.fit(X)


API Reference
-------------

.. automodule:: apricot.optimizers.BidirectionalGreedy
	:members:
	:inherited-members: