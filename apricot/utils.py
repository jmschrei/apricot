# utils.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This code contains utility functions to support the main functionality of
the code.
"""

from heapq import heappush
from heapq import heappop

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

    def __init__(self):
        self.pq = []
        self.lookup = {}
    
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
        
        entry = [weight, item]
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
            weight, item = heappop(self.pq)
            if item != "DELETED":
                del self.lookup[item]
                return weight, item
        
        raise KeyError("No elements left in the priority queue.")
