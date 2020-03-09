.. _submodular-opt:

Submodular Optimization
=======================

The field of submodular optimization involves functions of the form :math:`f : 2^{V} \rightarrow \mathcal{R}`. These functions take in a set of elements and output a real value that measures the quality of the subset. These functions are *submodular* when the returned value has the diminishing returns property such that for each :math:`X \subseteq Y`

.. math::
	f(X + {v}) - f(X) \geq f(Y + {v}) - f(Y)

This property means that the gain of adding in a particular element :math:`v` decreases or stays the same each time another element is added to the subset.  

.. note::

	Functions are called *modular* when the returned value ignores the size of the subset and *supermodular* when the returned value increases with subset size.


An intuitive example of the diminishing returns property arises from text data. In this situation, the gain in the function value could be defined to be the number of new words that each example (e.g. a sentence, paragraph, etc.) provides to the subset. Consider a particular example :math:`v`. When an element is added to the subset, the words that are not yet in the subset are either also in :math:`v`, decreasing the gain in adding :math:`v` later, or no new words are also in :math:`v`, keeping the gain the same. The gain from an :math:`v` cannot increase as new elements are added because either it has the same number or fewer new words to contribute to the subset. Here is an example of optimizing a maximum coverage function on text data, as described.

.. code-block:: python

	from apricot import MaxCoverageSelection
	from keras.datasets import reuters

	(X_, _), (_, _) = reuters.load_data(num_words=5000)
	X = numpy.zeros((X_.shape[0], max(map(max, X_))+1))
	for i, x in enumerate(X_):
	    X[i][x] = 1

	model = MaxCoverageSelection(250, optimizer='naive')
	model.fit(X)

It is NP-hard to optimize submodular functions exactly and so a greedy algorithm is usually employed. This algorithm works by selecting examples one at a time based on the gain they would yield if added. In practice, the greedy algorithm finds solutions that are very close to the optimal solution, and `Nemhauser (1978) <http://www.cs.toronto.edu/~eidan/papers/submod-max.pdf>`_ showed that the quality of the subset cannot be worse than :math:`1 - e^{-1}` of the optimal value when selected using a greedy algorithm. More recently, many algorithms have been proposed that allow optimization to scale to massive data sets or be distributed across machines but only find approximations of the greedy solution.
