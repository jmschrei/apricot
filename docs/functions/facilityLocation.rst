.. _functions.facility-location:

Facility Location
=================

Facility location functions are general purpose submodular functions that, when maximized, choose examples that represent the space of the data well. In many ways, optimizing a facility location function is simply a greedy version of k-medoids, where after the first few examples are selected, the subsequent ones are at the center of clusters. The function, like most graph-based functions, operates on a pairwise similarity matrix, and successively chooses examples that are similar to examples whose current most-similar example is still very dissimilar. Phrased another way, successively chosen examples are representative of underrepresented examples.

The general form of a facility location function is 

.. math::
	f(X, Y) = \sum\limits_{y in Y} \max_{x in X} \phi(x, y)

where :math:`f` indicates the function, :math:`X` is a subset, :math:`Y` is the ground set, and :math:`\phi` is the similarity measure between two examples. Like most graph-based functons, the facility location function requires access to the full ground set.


API Reference
-------------

.. automodule:: apricot.functions.facilityLocation
	:members:
	:inherited-members: