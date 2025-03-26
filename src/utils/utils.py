from src.utils.plots import *
from collections import Counter


def majority_vote(predictions, rule="hard"):
    """
    Performs majority voting on a list of predictions.

    Args:
        predictions: A list of lists, where each inner list contains predictions (classes) from one model.
        rule: "hard" for hard voting, "soft" for soft voting (default).

    Returns:
        A list of size equal to the number of inner lists, where each element is the majority class for the corresponding prediction across all models.
    """
    majority_classes = []
    for i in range(
            len(predictions[0])):

        if rule == 'hard':
            class_counts = Counter([prediction[i]
                                   for prediction in predictions])
            majority_class = class_counts.most_common(1)[0][0]
        elif rule == 'soft':
            majority_class = [0] * len(predictions[0][0])

            for model_predictions in predictions:
                majority_class = [
                    a + b for a,
                    b in zip(
                        majority_class,
                        model_predictions[i])]

            if sum(majority_class) > 0:
                majority_class = [p / sum(majority_class)
                                  for p in majority_class]

        else:
            raise ValueError("Invalid rule. Choose 'hard' or 'soft'.")
        majority_classes.append(majority_class)

    return majority_classes


def get_index_and_proba(data):
    """
    Finds the index and value of the highest element in each sublist.

    Args:
        data: A list of lists, where each inner list contains numerical values.

    Returns:
        A tuple containing two lists:
            - indices: A list containing the index of the highest element in each sublist.
            - values: A list containing the corresponding highest elements.
    """
    indices = []
    values = []

    for _, sublist in enumerate(data):
        # Find the index of the maximum value
        max_index = sublist.index(max(sublist))
        # Append the index and corresponding value
        indices.append(max_index)
        values.append(max(sublist))

    return indices, values
