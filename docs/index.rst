.. image:: logos/apricot-logo.png
	:width: 500px

|
 
.. image:: https://travis-ci.org/jmschrei/apricot.svg?branch=master
	:target: https://travis-ci.org/jmschrei/aprico

.. image:: https://readthedocs.org/projects/apricot-select/badge/?version=latest
   :target: http://apricot-select.readthedocs.io/en/latest/?badge=latest

|


apricot
=======

apricot is a Python package that implements submodular optimization for the purpose of summarizing massive data sets into representative subsets. These subsets are widely useful, but perhaps the most relevant usage of these subsets are either to visualize the modalities that exist in massive data sets, or for training accurate machine learning in a fraction of the time and compute power. 


Installation
------------

apricot can be installed using `pip install apricot-select`.




.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   CODE_OF_CONDUCT.rst
   faq.rst
   whats_new.rst

.. toctree::
	:maxdepth: 1
	:hidden:
	:caption: Features

	features/sparse.rst
	features/gpu.rst
	features/numba.rst
	features/

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Functions

   functions/featureBased.rst
   functions/maxCoverage.rst
   functions/facilityLocation.rst
   functions/graphCut.rst
   functions/sumRedundancy.rst
   functions/saturatedCoverage.rst
   functions/mixture.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Optimizers

   optimizers/naive.rst
   optimizers/lazy.rst
   optimizers/two-stage.rst
   optimizers/approx-lazy.rst
   optimizers/stochastic.rst
   optimizers/sample.rst
   optimizers/greedi.rst
   optimizers/modular.rst
   optimizers/bidirectional.rst
