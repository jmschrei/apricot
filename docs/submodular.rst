.. _submodular-opt:

Submodular Optimization
=======================

The field of submodular optimization involves functions of the form :math:`f : 2^{V} \rightarrow \mathcal{R}`. These functions take in a set of elements and output a real value that measures the quality of that set. Set functions are *submodular* when the returned value has the diminishing returns property such that for each :math:`X \subseteq Y`

.. math::
	f(X + {v}) - f(X) \geq f(Y + {v}) - f(Y)

The diminishing returns property means that the gain of adding in a particular element :math:`v` decreases or stays the same each time another element is added to the subset.  

.. note::

	Set functions are called *modular* when the returned value ignores the size of the subset and *supermodular* when the returned value increases with subset size.

An intuitive example of the diminishing returns property arises in the context of text data. In this situation, each example would be a blob of text, e.g. a sentence, and the set function being employed calculates the number of unique words across all selected sentences. Consider a particular example :math:`v`, an already selected set of examples :math:`X`, and a calculated gain :math:`g_{v}` of adding :math:`v` to :math:`X`. If some other example, :math:`u` is added to :math:`X` instead of :math:`v`, the gain of adding :math:`v` (or any other example) to :math:`X \cup u` must be smaller or equal to the gain of adding :math:`v` to :math:`X`: either :math:`u` has new words that would also be new from :math:`v`, in which case the benefit of adding in :math:`v` would decrease, or it doesn't, in which case the gain would stay the same. 

Here is an example of optimizing a maximum coverage function on text data, as described.

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
