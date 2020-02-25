.. _optimizers.sample

Sample Greedy
=============

The sample greedy algorithm is a simple approach that subsamples the full data set with a user-defined sampling probability and then runs an optimization on that subset. This subsampling can lead to obvious speed improvements because fewer elements as selected, but will generally find a lower quality subset because fewer elements are present. This approach is typically used a baseline for other approaches but can save a lot of time on massive data sets that are known to be highly redundant.

.. code::python

	from apricot import FeatureBasedSelection

	X = numpy.random.randint(10, size=(10000, 100))

	selector = FeatureBasedSelection(100, 'sqrt', optimizer='sample')
	selector.fit(X)


API Reference
-------------

.. automodule:: apricot.optimizers.SampleGreedy
	:members:
	:inherited-members: