<img src="https://github.com/jmschrei/apricot/blob/master/docs/logos/apricot-logo.png" width=600>

[![Build Status](https://travis-ci.org/jmschrei/apricot.svg?branch=master)](https://travis-ci.org/jmschrei/pomegranate)

Please consider citing the [manuscript](https://arxiv.org/abs/1906.03543) if you use apricot in your academic work!

apricot implements submodular optimization for the purpose of selecting diverse, minimally redundant, subsets of data from massive data sets. While there are many uses for these subsets, their primary usages are likely for visualizing the modalities in the data (such as in the two data sets below) and to train accurate machine learning models with just a fraction of the examples and compute. 

![](img/embeddings.png)

There are many built-in submodular functions and optimizers in apricot. These include:

**Functions**
* Feature-based
* Max (Set) Coverage
* Facility Location
* Graph Cut
* Sum Redundancy
* Saturated Coverage
* Mixtures of Functions

**Optimizers**
* Naive Greedy
* Lazy Greedy
* Approximate Lazy Greedy
* Stochastic Greedy
* Sample Greedy
* GreeDi
* Modular Approximation
* Two-Stage
* Bidirectional Greedy

### Installation

apricot can be installed easily from PyPI with `pip install apricot-select`

### Usage

The main objects in apricot are the selectors. Each selector encapsulates a submodular function and the cached statistics that speed up the optimization process. These selectors have a simple API built in the style of a scikit-learn transformer consisting of a `fit` method where the examples are selected, a `transform` method that applies this selection to the data set, and a `fit_transform` method that does both sequentially. 

Here is an example of reducing a data set of size 5000 down to 100 examples by using a facility location function. 

```python
import numpy
from apricot import FacilityLocationSelection

X = numpy.random.normal(100, 1, size=(5000, 25))
X_subset = FacilityLocationSelection(100).fit_transform(X)
```

#### Feature based functions quickly select subsets for machine learning models

```python
FeatureBasedSelection(n_samples, concave_func='sqrt', optimizer='two-stage', verbose=False)
```

Feature-based functions work well when the features correspond to some notion of quantity or importance, e.g. when the features are number of times a word appears in a document, or the strength of a signal at a sensor. When these functions are maximized, the resulting subsets are comprised of examples that are enriched in value in different sets of features. The intuition behind using a feature-based function is that different modalities in the data (like classes or clusters) will exhibit high values in different sets of features, and so selecting examples enriched for different features means covering these modalities better. 

When using a feature-based function on the 20 newsgroups data set, one can train a logistic regression model using only 100 samples and get the same performance as using all 1,187 potential samples, much better than using random sampling.

![](img/20newsgroups.png)

#### Facility location functions work in a variety of situations

```python
FacilityLocationSelection(n_samples, metric='euclidean', optimizer='two-stage', verbose=False)
```

Facility location functions are more general purpose submodular functions and work whenever a similarity measure can be defined over pairs of examples. When these functions are maximized, elements are selected that represent many elements that are currently underrepresented. However, a limitation of facility location functions (and all other submodular functions that rely on a similarity matrix) is that the full similarity matrix requires quadratic memory with the number of examples and is generally impractical to store.

These exemplars can be used for a variety of tasks, including selecting subsets for training machine learning models, visualization in the place of large data sets, or as centroids in a greedy version of k-means clustering. The animation below shows samples being selected according to facility location and compared to random selection, with facility location first selecting a sample that represents the entire data set, then selecting a sample that is in the largest cluster but near the second largest, and then the centers of local neighborhoods.

![](img/facilitylocation.gif)


#### Initial subsets

The selection process for most optimizers implemented in apricot is greedy, meaning that one example is selected at a time and this is (usually) the best example to include next given those that have already been selected. While a greedy algorithm is not guaranteed to find the best subset of a given size, it was famously shown that this subset cannot have an objective value $1 - e^{-1}$ worse than the optimal subset, and in practice the subsets are usually near-optimal.

apricot exploits the greedy aspect of the algorithms to allow users to define an initial subset to build off of. This can be useful when one already has some elements selected (such as by outside information) and one would like to select elements that are not redundant with these already-selected elements. The initial subsets can be defined in the constructor like so: 

```python
import numpy
from apricot import FacilityLocationSelection

X = numpy.random.normal(20, 1, size=(5000, 25))
X_reordered = FacilityLocationSelection(100, initial_subset=[1, 5, 6, 8, 10]).fit_transform(X)

model = FacilityLocationSelection(1000).fit(X)
X_reordered2 = X[model.ranking]
```

#### scikit-learn integration

Because apricot implements these functions in the style of a scikit-learn transformer, they can be dropped into current workflows. Clustering models that used to be defined as follows:

```python
model = GaussianMixture(10)
model.fit(X)
```

can now be turned into a pipeline without changing the rest of your code like such:

```python
model = Pipeline([('selection', FacilityLocationSelection(25)),
                  ('model', GaussianMixture(10))])
model.fit(X)
```

sklearn does not currently allow transformers to modify the label vector y even though they can modify the data matrix X and so apricot cannot be dropped into pipelines for supervised models. However, if you are not constrained to use a pipeline then you can transform both the data matrix and the label vector together:

```python
X_subset, y_subset = FacilityLocationSelection(100).fit_transform(X, y)
model.fit(X_subset, y_subset)
```

#### When should I use submodular selection?

If the amount of data that you have right now is not burdensome to deal with then it may not be helpful to use submodular selection. However, if training even simple models using the amount of data you have is difficult, you might consider summarizing your data set using apricot. If you're currently running many random selections because of the amount of data you have you may find that a single run of apricot will yield a better subset.

