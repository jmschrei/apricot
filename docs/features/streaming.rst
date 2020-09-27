.. _features.streaming:

Streaming Optimization
======================

Streaming optimization involves optimizing a submodular function on a stream of data. The key assumptions of these methods are that the data cannot all fit in memory at one time and, more restrictively, that each example can only be seen once: after seeing each example, the algorithm must decide whether to keep it or not before moving on to the next example. 

Although this problem is difficult, principled approaches for solving it have been proposed that have theoretical guarantees. One such example, `sieve streaming <http://www.cs.cornell.edu/~ashwin85/docs/frp0328-badanidiyuru.pdf>`_, involves making many estimates of what the objective score of the optimal subset would be, and selecting elements based on these elements in parallel. When estimates are found to be too low, the corresponding subsets are discarded, and when certain estimates that were initially too high are found to be plausible, the corresponding subset begins to be populated.

In apricot, streaming optimization can be used with any of the built-in functions by using the `partial_fit` method instead of the `fit` method. Although the algorithm is designed to be applied to one example at a time in a streaming setting, in practice applying the algorithm to batches of data can be much faster while still providing the same answer.

.. note::
	Streaming optimization is implemented for mixtures of functions, but not for sum redundancy or saturated coverage functions.

You can read more about streaming optimization `in this tutorial <https://github.com/jmschrei/apricot/blob/master/tutorials/6.%20Streaming%20Submodular%20Optimization.ipynb>`_. 

.. code-block:: python

	from apricot import FeatureBasedSelection

	X = numpy.random.randint(2, size=(10000, 100), p=[0.99, 0.01])
	sample_cost = numpy.exp(numpy.random.randn(10000))
	
	selector = FeatureBasedSelection(100, 'sqrt')
	selector.partial_fit(X, sample_cost=sample_cost)
