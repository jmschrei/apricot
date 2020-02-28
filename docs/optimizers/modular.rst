.. _optimizers.modular

Modular Greedy
==============

The modular greedy optimizer uses the modular upper-bounds for the gain of each example to do selection. Essentially, a defining characteristic of submodular functions is the *diminishing returns* property where the gain of an example decreases with the number of selected examples. In contrast, modular functions have constant gains for examples regardless of the number of selected examples. Thus, approximating the submodular function as a modular function can serve as an upper-bound to the gain for each example during the selection process. This approximation makes the function simple to optimize because one would simply calculate the gain that each example yields before any examples are selected, sort the examples by this gain, and select the top :math:`k` examples. While this approach is fast, this approach is likely best paired with a traditional optimization algorithm after the first few examples are selected.

.. code::python

	from apricot import FeatureBasedSelection

	X = numpy.random.randint(10, size=(10000, 100))

	selector = FeatureBasedSelection(100, 'sqrt', optimizer='modular')
	selector.fit(X)


API Reference
-------------

.. automodule:: apricot.optimizers.ModularGreedy
	:members:
	:inherited-members: