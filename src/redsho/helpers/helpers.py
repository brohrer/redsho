import random
from typing import Any

import numpy as np

type Cond = dict[str, Any]
type CondGrid = dict[str, list[Any]]
type CondList = list[Cond]


def get_random_condition(condition_grid: CondGrid) -> Cond:
    """
    Pick a random condition from the space of all possible conditions.
    There's no guarantee that this condition hasn't been generated before
    or that it hasn't been evaluated already. If this is important,
    make sure you add that logic elsewhere.
    """
    rand_condition: Cond = {}
    for param, value in condition_grid.items():
        rand_condition[param] = random.choice(value)
    return rand_condition


def low_biased_choice(
    n_choices: int,
    errors: list[float],
    greediness: float,
) -> list[int]:
    """
    Assign a selection_weight to option,
    based on the error score (or loss)
    associated with it. We want to choose low error options
    but they don't have to be the lowest.
    Rather, they are chosen randomly giving strong preference to options
    with lower errors.
    """
    if len(errors) <= 0:
        return []
    if n_choices <= 0:
        return []
    if n_choices >= len(errors):
        return list(np.arange(len(errors), dtype=int))

    eps: float = 1e-17  # to avoid division by zero
    selection_weights: Any = (  # actually a NumPy n-d Array
        (np.max(errors) - np.array(errors))
        / (np.max(errors) - np.min(errors) + eps)
    ) ** greediness
    # Turn the weights into a proper probability distribution, summing to one.
    selection_probabilities: Any = selection_weights / np.sum(selection_weights)

    i_choices: list[int] = np.random.choice(
        np.arange(len(errors), dtype=int),
        n_choices,
        replace=False,
        p=selection_probabilities,
    ).tolist()

    return i_choices
