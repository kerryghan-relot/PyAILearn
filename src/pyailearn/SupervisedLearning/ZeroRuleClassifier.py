"""
This implements the basic Zero-R benchmark classifier for classification problem in machine learning.
One can safely assume that any classification algorithm is of no use if it does not perform better than Zero-R

One-R simply take the majority element of the class you give to it and will always predict it.
"""

from collections import Counter
from collections.abc import Generator, Iterable
from typing import Any

# Since Python 3.9 the use of type annotations like Iterable or Generator from the typing module are deprecated.
# Instead, one should use the collections.abc module


def predict(y: Iterable) -> Any:
    """
    ZeroR Classifier simply returns the majority element of the class.
    This is an efficient way to predict which element appears the most in a given iterable.

    :param y: The class you want to get the majority element of.
    :return: The element of the given iterable that appears the most.
    """
    return Counter(y).most_common()[0][0]


def baseline(y: Iterable) -> tuple[Any, float]:
    """
    Determine the majority element and its percentage of appearance in the given class.

    :param y:  The class you want to classify
    :return: a tupple (e, p) where e is an element of y and p is the percentage of times it appears in the given iterable.
    """
    # Get the total number of elements
    n = len(y)

    # Get the most common element and its occurrence.
    e, p = Counter(y).most_common()[0]

    return e, p/n


def classify(y: Iterable) -> Generator[tuple[Any, float], Any, None]:
    """
    Determine specifically the percentage of apparition of each element and outputs it in sorted order.
    A generator is returned in order preserve space and efficiency.
    This can be useful for determining detailed occurrences of element in the class.

    :param y:  The class you want to classify
    :return: a generator object that gives tupples (e, p) where e is an element of y and p is the percentage of times it appears in the given iterable.
    """
    # Get the total number of elements
    n = len(y)

    # Get the number of element for each class value as a list of tuples in descending order according to the number of occurences
    counts = Counter(y).most_common()

    # Convert the number of occurrences into percentages
    counts = ((val, occ / n) for val, occ in counts)

    return counts