import random

from redsho.helpers.helpers import (
    Cond,
    CondGrid,
    get_random_condition,
    low_biased_choice,
)

default_greediness: float = 2.0


def test_low_biased_choice_single() -> None:
    n_choices: int = 1
    errors: list[float] = [random.random() for _ in range(8)]
    choices = low_biased_choice(n_choices, errors, default_greediness)
    assert isinstance(choices, list)
    assert len(choices) == n_choices
    assert isinstance(choices[0], int)
    assert choices[0] < len(errors)


def test_low_biased_choice_multiple() -> None:
    n_choices: int = 5
    errors: list[float] = list(range(11))
    choices = low_biased_choice(n_choices, errors, default_greediness)
    assert isinstance(choices, list)
    assert len(choices) == n_choices
    assert isinstance(choices[0], int)
    for choice in choices:
        assert choice < len(errors)


def test_low_biased_no_choice() -> None:
    n_choices: int = 0
    errors: list[float] = [random.random() for _ in range(8)]
    choices = low_biased_choice(n_choices, errors, default_greediness)
    assert isinstance(choices, list)
    assert len(choices) == n_choices


def test_low_biased_choice_greedy() -> None:
    n_choices: int = 1
    errors: list[float] = [random.random() for _ in range(8)]
    high_greediness: float = 1000.0
    choices = low_biased_choice(n_choices, errors, high_greediness)
    assert errors[choices[0]] == min(errors)


def test_low_biased_choice_clear_best() -> None:
    n_choices: int = 1
    errors: list[float] = [1.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    choices = low_biased_choice(n_choices, errors, default_greediness)
    assert errors[choices[0]] == 1.0


def test_get_random_condition() -> None:
    condition_grid: CondGrid = {"c1": [4, 7, 9], "c2": [0.1, 0.3, 0.6, 0.8]}
    rand_condition: Cond = get_random_condition(condition_grid)
    assert len(rand_condition) == 2
    assert rand_condition["c1"] in condition_grid["c1"]
    assert rand_condition["c2"] in condition_grid["c2"]
