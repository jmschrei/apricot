.. _functions.feature-based:

Feature-based Functions
=======================

Feature-based functions are those that operate on the feature values of examples directly, rather than on a similarity matrix (or graph) derived from those features, as graph-based functions do. Because these functions do not require calculating and storing a :math:`\\mathcal{O}(n^{2})` sized matrix of similarities, they can easily scale to data sets with millions of examples. 

The general form of a feature-based function is:

.. math::
	f(X) = \sum\limits_{d=1}^{D} \phi \left( \sum\limits_{i=1}^{N} X_{i, d} \right)

where :math:`f` indicates the function that uses the concave function :math:`\phi` and is operating on a subset :math:`X` that has :math:`N` examples and :math:`D` dimensions. Importantly, :math:`X` is the subset and not the ground set, meaning that the time it takes to evaluate this function is proportional only to the size of the selected subset and not the size of the full data set, like it is for graph-based functions.  


API Reference
-------------

.. automodule:: apricot.featureBased
	:members:
	:inherited-members:
