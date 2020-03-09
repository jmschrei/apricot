.. _features.sparse:

Sparse Matrices
===============

apricot has built-in support for sparse matrices. This support makes it possible to summarize massive data sets whose dense representations are not feasible to work with, e.g. sparse similarity graphs when using graph-based functions or word counts for text data and feature-based functions. Using a sparse matrix will always (except when ties exist) give the same result that using a dense matrix populated with 0s would, but operating on sparse matrices can be significantly faster process.

Here is an example of using a feature-based function on a very sparse feature matrix. Note that each row can be thought of as an example with binary feature values.

.. code-block:: python

	from apricot import FeatureBasedSelection
	from scipy.sparse import csr_matrix

	X = numpy.random.randint(2, size=(10000, 100), p=[0.99, 0.01])
	X_sparse = csr_matrix(X)
	
	selector = FeatureBasedSelection(100, 'sqrt')
	selector.fit(X_sparse)

Here is an example of using a graph-based function on a very sparse similarity matrix. Note that X is a pre-computed similarity matrix and must be specified as such to the function.

.. code-block:: python

	from apricot import FacilityLocationSelection
	from scipy.sparse import csr_matrix
	
	X = numpy.random.randint(10, size=(1000, 1000), p=[0.99, 0.01])
	X = numpy.abs((X + X.T) / 2)
	X_sparse = csr_matrix(X)
	
	selector = FacilityLocationSelection(100, 'precomputed')
	selector.fit(X_sparse)
