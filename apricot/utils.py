# utils.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code contains utility functions to support the main functionality of
the code.
"""

import numbers
import numpy
import itertools

from heapq import heappush
from heapq import heappop
from heapq import heapify

from scipy.sparse import csr_matrix

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsTransformer

class PriorityQueue(object):
    """A priority queue implementation.

    This is an implemented of a priority queue using a heap implemtnation. It
    has O(log n) time for adding and popping elements but constant time
    deletion. It is based off of the implementation that is provided at

    https://docs.python.org/2/library/heapq.html

    This implementation assumes that the items with the highest priority
    are the ones with the lowest "weight" that gets passed in. If higher
    weights are supposed to correspond to higher priority, consider reversing
    the sign of the weight.

    Parameters
    ----------
    None

    Attributes
    ----------
    pq : list
        A list containing the heapified priority queue.

    lookup : dict
        A dictionary containing links to each of the elements, allowing
        constant time lookups and deletions.
    """

    def __init__(self, items=None, weights=None):
        self.counter = itertools.count()
        self.lookup = {}
        self.pq = []

        if items is not None and weights is not None:
            for item, weight in zip(items, weights):
                entry = [weight, next(self.counter), item]
                self.pq.append(entry)
                self.lookup[item] = entry

            heapify(self.pq)
    
    def add(self, item, weight):
        """Add an element to the priority queue. Runtime is O(log n).

        This adds in an element to the priority queue. If that element is
        already there it will automatically delete the previous instance
        and replace it with the new instance.

        Parameters
        ----------
        item : object
            The object to be encoded.

        weight : double
            The priority of the item. The lower the weight the higher the
            priority when items get dequeued. If a higher weight is supposed
            to correspond to a higher priority, consider reversing the sign of
            the weight.

        Returns
        -------
        None
        """

        if item in self.lookup:
            self.remove(item)
        
        entry = [weight, next(self.counter), item]
        self.lookup[item] = entry
        heappush(self.pq, entry)

    def remove(self, item):
        """Remove an element from the queue.

        This is not popping the highest priority item, rather it will remove
        an element that is present in the queue. If one attempts to remove an
        item that is not present in the queue the function will error.
        
        Parameters
        ----------
        item : object
            The object to be removed from the queue.

        Returns
        -------
        None
        """

        entry = self.lookup.pop(item)
        entry[-1] = "DELETED"
    
    def pop(self):
        """Pop the highest priority element from the queue. Runtime is O(log n).

        This will remove the highest priority element from the queue. If there
        are no elements left in the queue it will raise an error.

        Parameters
        ----------
        None

        Returns
        -------
        weight : double
            The weight of the element as passed in in the `add` method

        item : object
            The item that was passed in during the `add` method
        """

        while self.pq:
            weight, _, item = heappop(self.pq)
            if item != "DELETED":
                del self.lookup[item]
                return weight, item
        
        raise KeyError("No elements left in the priority queue.")

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    This function will check to see whether the input seed is a valid seed
    for generating random numbers. This is a slightly modified version of
    the code from sklearn.utils.validation.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """

    if seed is None or seed is numpy.random:
        return numpy.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, numpy.integer)):
        return numpy.random.RandomState(seed)
    if isinstance(seed, numpy.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def _calculate_pairwise_distances(X, Y=None, metric='precomputed', n_neighbors=None):
    if metric in ('precomputed', 'ignore'):
        return X

    if n_neighbors is None:
        if metric == 'euclidean':
            X_pairwise = pairwise_distances(X, Y=Y, metric=metric, squared=True)
        elif metric == 'correlation' or metric == 'cosine':
            # An in-place version of:
            # X_pairwise = 1 - (1 - pairwise_distances(X, metric=metric)) ** 2
            
            X_pairwise = pairwise_distances(X, Y=Y, metric=metric)
            X_pairwise = numpy.subtract(1, X_pairwise, out=X_pairwise)
            X_pairwise = numpy.square(X_pairwise, out=X_pairwise)
            X_pairwise = numpy.subtract(1, X_pairwise, out=X_pairwise)
        else:
            X_pairwise = pairwise_distances(X, Y=Y, metric=metric)
    else:
        if metric == 'correlation' or metric == 'cosine':
            # An in-place version of:
            # X = 1 - (1 - pairwise_distances(X, metric=metric)) ** 2

            X = pairwise_distances(X, Y=Y, metric=metric)
            X = numpy.subtract(1, X, out=X)
            X = numpy.square(X, out=X)
            X = numpy.subtract(1, X, out=X)
            metric = 'precomputed'

        if isinstance(n_neighbors, int):
            X_pairwise = KNeighborsTransformer(
                n_neighbors=n_neighbors, metric=metric
                ).fit_transform(X)

        elif isinstance(n_neighbors, KNeighborsTransformer):
            X_pairwise = n_neighbors.fit_transform(X)

    if metric == 'correlation' or metric == 'cosine':
        if isinstance(X_pairwise, csr_matrix):
            X_pairwise.data = numpy.subtract(1, X_pairwise.data, 
                out=X_pairwise.data)
        else:
            X_pairwise = numpy.subtract(1, X_pairwise,
                out=X_pairwise)
    else:
        if isinstance(X_pairwise, csr_matrix):
            X_pairwise.data = numpy.subtract(X_pairwise.max(),
                X_pairwise.data, out=X_pairwise.data)
        else:
            X_pairwise = numpy.subtract(X_pairwise.max(), X_pairwise,
                out=X_pairwise)

    return X_pairwise