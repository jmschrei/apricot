.. _features.sparse

Sparse Matrices
===============

There is built-in support for sparse matrices for both feature-based and graph-based functions. Using a sparse matrix will always (except when ties exist) give the same result that using a dense matrix populated with 0s would, but it can be significantly faster to explicitly use a sparse matrix representation. For feature-based functions, missing values are assumed to be feature values of 0 and thus do not increase the gain of an example. For graph-based functions, missing values are assumed to be similarities of 0 between examples. It is infrequent in reality to see a similarity of exactly 0 when the similarities are derived from feature values, so this is most likely the case when one is using a pre-defined similarity matrix or an approximation of the dense similarity matrix.

Here is an example of using a feature-based function on a very sparse matrix.

.. code::python

	from apricot import FeatureBasedSelection
	from scipy.sparse import csr_matrix

	X = numpy.random.randint(2, size=(10000, 100), p=[0.99, 0.01])
	X_sparse = csr_matrix(X)

	selector = FeatureBasedSelection(100, 'sqrt')
	selector.fit(X_sparse)

Here is an example of using a graph-based function on a very sparse matrix.

..code::python

	from apricot import FacilityLocationSelection
	from scipy.sparse import csr_matrix

	X = <a dense matrix that has many 0s in it>
	X_sparse = csr_matrix(X)

	selector = FacilityLocationSelection(100, 'precomputed')
	selector.fit(X_sparse)


