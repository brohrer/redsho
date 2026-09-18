import copy

import pytest

from redsho.helpers.helpers import (
    CondList,
)
from redsho.optimizer import (
    choose_children,
    choose_more_conditions,
    choose_parents,
    generate_conditions,
)

from common import (
    add,
    default_condition_grid,
    default_conditions,
    default_param,
    default_parent,
    empty_param,
    full_param,
    small_condition_grid,
    small_condition_list_full,
)


def test_generate_conditions_small() -> None:
    conditions: CondList = []
    for i in generate_conditions(conditions, small_condition_grid, int(1e4)):
        pass
    assert len(conditions) == 4


def test_generate_conditions_default() -> None:
    conditions: CondList = []
    for i in generate_conditions(conditions, default_condition_grid, int(1e4)):
        pass
    assert len(conditions) == 180


def test_generate_conditions_cap() -> None:
    conditions: CondList = []
    for _ in generate_conditions(conditions, default_condition_grid, 20):
        pass
    assert len(conditions) <= 23


def test_choose_more_conditions() -> None:
    conditions = copy.deepcopy(default_conditions)
    choose_more_conditions(
        conditions,
        default_condition_grid,
    )
    assert len(conditions) > len(default_conditions)


def test_choose_more_conditions_target() -> None:
    conditions = copy.deepcopy(default_conditions)
    choose_more_conditions(
        conditions,
        default_condition_grid,
        n_conditions_to_add=1,
    )
    assert len(conditions) == len(default_conditions) + 1


def test_choose_more_conditions_full() -> None:
    conditions = copy.deepcopy(small_condition_list_full)
    with pytest.raises(StopIteration):
        choose_more_conditions(
            conditions,
            small_condition_grid,
        )


def test_choose_parents() -> None:

    n_parents_to_try: int = len(default_condition_grid)

    parents: CondList = choose_parents(default_conditions, n_parents_to_try)
    assert isinstance(parents, list)
    assert len(parents) == n_parents_to_try
    assert isinstance(parents[0], dict)
    for parent in parents:
        assert parent in default_conditions


def test_choose_parents_low_supply() -> None:

    n_parents_to_try: int = 11

    parents: CondList = choose_parents(default_conditions, n_parents_to_try)
    assert isinstance(parents, list)
    assert len(parents) == 5
    assert isinstance(parents[0], dict)
    for parent in parents:
        assert parent in default_conditions


def test_choose_parents_no_supply() -> None:

    n_parents_to_try: int = len(default_condition_grid)
    # Create a copy of conditions that has not been evaluated.
    # It has no "error" key in its dict.
    unevaluated_conditions = copy.deepcopy(default_conditions)
    for cond in unevaluated_conditions:
        try:
            del cond["error"]
        except KeyError:
            pass

    parents: CondList = choose_parents(
        unevaluated_conditions,
        n_parents_to_try,
    )
    assert isinstance(parents, list)
    assert len(parents) == 1
    assert isinstance(parents[0], dict)
    assert len(parents[0]) == 0


def test_choose_parents_greedy() -> None:

    n_parents_to_try: int = 1

    parents: CondList = choose_parents(
        default_conditions,
        n_parents_to_try,
        greediness=1e6,
    )
    assert isinstance(parents, list)
    assert len(parents) == n_parents_to_try
    parent = parents[0]
    assert isinstance(parent, dict)
    assert parent["p0"] == 5
    assert parent["p1"] == 0.08
    assert parent["p2"] == "eleven"
    assert parent["p3"] == add
    assert parent["error"] == -0.6


def test_choose_children() -> None:
    success: bool = choose_children(
        default_conditions,
        default_condition_grid,
        default_parent,
        default_param,
    )
    assert success


def test_choose_children_empty_direction() -> None:
    success: bool = choose_children(
        default_conditions,
        default_condition_grid,
        default_parent,
        empty_param,
    )
    assert success


def test_choose_children_full_direction() -> None:
    success: bool = choose_children(
        default_conditions,
        default_condition_grid,
        default_parent,
        full_param,
    )
    assert not success
