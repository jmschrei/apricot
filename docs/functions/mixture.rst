.. _functions.mixtures

Mixtures
========

A convenient property of submodular functions is that the addition of two of them is still a submodular function. More generally, the linear combination of any number of submodular functions (assuming non-negative weights) is still a submodular function. Because of this, a mixture of submodular functions can be optimized using the same algorithms as an individual submodular function. Mixtures can be useful in situations where there are different important aspects of data that are each submodular.

The general form of a mixture function is 

.. math::
	f(X) = \sum\limits_{i=1}^{M} \alpha_{i} g_{i}(X) 

where :math:`f` indicates the mixture function, :math:`M` is the number of functions in the mixture, :math:`X` is a subset, :math:`\alpha_{i}` is the weight of the :math:`i`-th function and :math:`g_{i}` is the :math:`i`-th function.


API Reference
-------------

.. automodule:: apricot.functions.mixture
	:members:
	:inherited-members: