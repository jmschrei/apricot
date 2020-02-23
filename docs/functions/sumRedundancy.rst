.. _functions.sum-redundancy:

Sum Redundancy
==============

Sum redundancy functions.

The general form of a sum redundancy function is 

.. math::
	f(X, Y) = \sum\limits_{y in Y} \max_{x in X} \phi(x, y)

where :math:`f` indicates the function, :math:`X` is a subset, :math:`Y` is the ground set, and :math:`\phi` is the similarity measure between two examples. Like most graph-based functons, the facility location function requires access to the full ground set.


API Reference
-------------

.. automodule:: apricot.sumRedundancy
	:members:
	:inherited-members: