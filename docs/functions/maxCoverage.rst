.. _functions.max-coverage:

Maximum Coverage
================

Maximum coverage functions aim to maximize the number of features that have a non-zero element in at least one selected example---there is no marginal benefit to observing a variable in two examples. If each variable is thought to be an item in a set, and the data is a binary matrix where a 1 indicates the item is present in the example and 0 indicates it is not, optimizing a maximum coverage function is a solution to the set coverage problem. These functions are useful when the space of variables is massive and each example only sees a small subset of them, which is a common situation when analyzing text data when the variables are words. The maximum coverage function is an instance of a feature-based function when the concave function is minimum.

The general form of a feature-based function is:

.. math::
	f(X) = \sum\limits_{d=1}^{D} \min \left( \sum\limits_{n=1}^{N} X_{i, d}, 1 \right) 

where :math:`f` indicates the function that operates on a subset :math:`X` that has :math:`N` examples and :math:`D` dimensions. Importantly, :math:`X` is the subset and not the ground set, meaning that the time it takes to evaluate this function is proportional only to the size of the selected subset and not the size of the full data set, like it is for graph-based functions.  


API Reference
-------------

.. automodule:: apricot.functions.maxCoverage
	:members:
	:inherited-members: