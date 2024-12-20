"""
This implements the basic One-R benchmark classifier for more complex classification problem in machine learning.
One can safely assume that classification algorithms are of no use if they do not perform better than One-R

One-R can be seen as a single node decision tree, choosing the feature that discriminate the dataset the best and solely
using this single feature to make it's prediction.
"""

import time
from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer

# Since Python 3.9 the use of type annotations like Iterable or Generator from the typing module are deprecated.
# Instead, one should use the collections.abc module


def __value_conversion(class_mapping: dict[Any, int], element_mapping: dict[Any, int], class_counts: np.ndarray, value_type: str) -> dict:
    """
    Private helper function to convert a class mapping from one type to another depending on the given value_type.

    :return: The mapping in the desired format
    """
    # In order to do just that, first, revert the mappings
    element_class_key = list(class_mapping.keys())
    best_class_key = list(element_mapping.keys())
    # And eventually convert values according to the given flag.
    if value_type == "native":
        to_native = (lambda val: getattr(val, "tolist", lambda: val)())
        return {to_native(best_class_key[i]): to_native(element_class_key[j]) for i, j in
                enumerate(class_counts.argmax(axis=1))}
    elif value_type == "str":
        return {str(best_class_key[i]): str(element_class_key[j]) for i, j in
                enumerate(class_counts.argmax(axis=1))}
    else:
        return {best_class_key[i]: element_class_key[j] for i, j in
                enumerate(class_counts.argmax(axis=1))}

def __oner(dataset: Iterable, classes: Iterable, return_value_type: str ="default") -> tuple[dict, float]:
    """
    Private helper function to run the OneR algorithm.
    It returns both the prediction (best feature and output mapping) and the actual score of that prediction
    As it is a private helper function, it is assumed that the given data have already been type checked by the caller
    or it is accepted that error may be raised (by the numpy module) if incorrect datatype have been passed.

    :param dataset: The dataset you want to classify
    :param classes: The corresponding classes, this should have the same length as the dataset argument
    :param return_value_type: optional parameter that lets you decide what type the output value should be.\
     Sometimes, the value type can be modified during the process due to the use of numpy.\
     The following values can be chosen: \
     default>will not change anything and you might expect to get back modified value types \
     native>will convert the value back to their corresponding native python type if possible. \
     str>will convert the value to their string representation.
    :return: A tuple containing the prediction and actual score
    """
    #TODO: make it work with lists and dataframes
    M, N = dataset.shape  # M = the number of data (lines) ; N = the number of features (column)
    element_class_mapping = {el: idx for idx, el in enumerate(np.unique(classes))}
    n_class_elements = len(element_class_mapping)
    score = -1
    best_feature = -1
    best_class_counts = None
    best_element_mapping = None

    # First, for every possible feature, compute its Accuracy.
    for feature in range(N):
        # Create a mapping of the features values to a range of indices
        unique_elements_mapping = {el: idx for idx, el in enumerate(np.unique(dataset[:, feature]))}
        # Count the number of unique feature value
        n_unique_elements = len(unique_elements_mapping)
        # Create a 2d numpy array of shape (L, P) where
        #       - L=number of unique feature value for the current feature
        #       - P=number of unique class value in our dataset
        class_counts = np.zeros((n_unique_elements, n_class_elements))

        # Now, for every datapoint in the current feature (and its associated class),
        # let's count how many times each unique feature value correspond to each unique class value
        for datapoint, element_class in zip(dataset[:, feature], classes):
            # Retrieve the proper indices using our mappings
            datapoint_idx = unique_elements_mapping[datapoint]
            class_idx = element_class_mapping[element_class]
            # Add the occurrences in our class count.
            class_counts[datapoint_idx, class_idx] += 1

        # Once we're done with the class_count filling, we have to check if it gives us a better score than what we already have
        new_score = class_counts.max(axis=1).sum() # Take the maximum of each line, and sum that up
        if new_score > score:
            score = new_score
            best_feature = feature
            best_class_counts = class_counts
            best_element_mapping = unique_elements_mapping

    # Now that we computed the score of each feature and kept the best one,
    # we have to return a prediction mapping
    output_mapping = __value_conversion(element_class_mapping, best_element_mapping, best_class_counts, return_value_type)

    result = {"Feature": best_feature,
              "Decision": output_mapping}

    return result, score/M

def predict(dataset: Iterable, classes: Iterable, return_value_type: str ="default") -> dict[str, int | dict]:
    """
    OneR classifier choses the best feature to make a decision and will ignore the other
    It selects the feature that allows us to best split the dataset.

    :param dataset: The dataset you want to classify
    :param classes: The corresponding classes, this should have the same length as the dataset argument
    :param return_value_type: optional parameter that lets you decide what type the output value should be.\
     Sometimes, the value type can be modified during the process due to the use of numpy.\
     The following values can be chosen: \
     default>will not change anything and you might expect to get back modified value types \
     native>will convert the value back to their corresponding native python type if possible. \
     str>will convert the value to their string representation.
    :return: A dictionary of the prediction. The first element to the dictionary is the index of the feature to select\
     (referred by the key "Feature") and the second element is the mapping of the values to the predicted class\
     (referred by the key "Decision").
    """

    if return_value_type not in ["default", "native", "str"]:
        raise ValueError("Return_type must be 'default', 'native' or 'str'.")

    return __oner(dataset, classes, return_value_type)[0]

def baseline(dataset: Iterable, classes: Iterable) -> float:
    """
    Determine the best feature to split the data and returns the baseline score as a percentage.

    :param dataset: The dataset you want to classify
    :param classes: The corresponding classes, this should have the same length as the dataset argument
    :return: Returns the score of the single feature prediction as a percentage (0.0 to 1.0)
    """
    return __oner(dataset, classes)[1]





if __name__ == "__main__":
    #TODO: remove this main and do the testing inside a proper test script

    start = time.perf_counter()
    iris = load_iris()
    end = time.perf_counter()
    total_duration = end - start
    print("Total time:", total_duration)

    print(type(iris.data))
    print(type(iris.data.tolist()))
    #pp(iris.data.tolist())

    start = time.perf_counter()
    X = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile').fit_transform(iris.data)#.tolist()
    end = time.perf_counter()
    total_duration = end - start
    print("Total time:", total_duration)
    y = iris.target

    print(type(X))

    start = time.perf_counter()
    prediction = predict(X, y, "native")
    end = time.perf_counter()
    total_duration = end - start
    print("Total time:", total_duration)

    start = time.perf_counter()
    score = baseline(X, y)
    end = time.perf_counter()
    total_duration = end - start
    print("Total time:", total_duration)

    print(f"\nChosen classifier: {prediction["Feature"]} ({prediction["Feature"] + 1}th column)\n"
          f"Predicted class: {prediction["Decision"]}")

    print(f"\nActual score of the OneR classifier as a baseline: {score:.2f}%")
