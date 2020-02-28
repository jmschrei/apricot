.. _optimizers.greedi

GreeDi
======

GreeDi is an optimizer that was designed to work on data sets that are too large to fit into memory. The approach involves first partitioning the data into :math:`m` equally sized chunks without any overlap. Then, :math:`l` elements are selected from each chunk using a standard optimizer like naive or lazy greedy. Finally, these :math:`ml` examples are merged and a standard optimizer selects :math:`k` examples from this set. In this manner, the algorithm sacrifices exactness to ensure that memory limitations are not an issue. 

There are a few considerations to keep in mind when using GreeDi. Naturally, :math:`ml` must both be larger than :math:`k` and also small enough to fit into memory. The larger :math:`l`, the closer the solution is to the exact solution but also the more compute time is required. Conversely, the larger :math:`m` is, the less exact the solution is. When using a graph-based function, increasing :math:`m` can dramatically reduce the amount of computation that needs to be performed, as well as the memory requirements, because the similarity matrix becomes smaller in size. However, feature-based functions are likely to see less of a speed improvement because the cost of evaluating an example is independent of the size of ground set.

.. code::python

	from apricot import FeatureBasedSelection

	X = numpy.random.randint(10, size=(10000, 100))

	selector = FeatureBasedSelection(100, 'sqrt', optimizer='greedi')
	selector.fit(X)


API Reference
-------------

.. automodule:: apricot.optimizers.GreeDi
	:members:
	:inherited-members: