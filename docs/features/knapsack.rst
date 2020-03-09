.. _features.knapsack:

Knapsack Constraints
====================

A knapsack constraint is an alternate to a cardinality constraint (the default in apricot) where each element has a cost and the selected items cannot exceed a total budget. The name (according to `Krause (2012) <https://las.inf.ethz.ch/files/krause12survey.pdf>`) is a reference to the knapsack problem where one maximizes a modular function subject to a modular cost. In apricot, one is maximizing a *submodular* function subject to a modular cost. There are several ideas that costs might represent but the most intuitive is the simple monetary cost associated with each element. Consider the task of placing sensors around an area given a fixed budget to purchase sensors, where smaller sensors are cheaper than larger sensors. Submodular optimization subject to knapsack constraints is an intuitive way to determine what type of sensor should be placed and where without exceeding a given budget.

Stated more formally, the optimization problem given knapsack constraints is 

.. math::
	\max\limits_{S} f(S) \enspace s.t. \sum\limits_{v \in S} c(v) \leq B

for a cost function :math:`c`, a submodular function :math:`f`, a subset :math:`S`, and a budget :math:`B`. Further, rather than optimizing the gain directly, the gain is divided by the cost of the element, resulting in the "cost-benefit" gain. Essentially, elements with a high gain and a high cost are prioritized lower than those with the same gain but a lower cost. This means that the optimization process is just a 

.. code-block:: python

	from apricot import FeatureBasedSelection

	X = numpy.random.randint(2, size=(10000, 100), p=[0.99, 0.01])
	sample_cost = numpy.abs(numpy.random.randn(10000))
	
	selector = FeatureBasedSelection(100, 'sqrt')
	selector.fit(X_sparse, sample_cost=sample_cost)
