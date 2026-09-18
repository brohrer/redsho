from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from redsho.helpers.equals import (
    contains_condition,
    eq,
    eq_dict,
    eq_float,
    eq_func,
    eq_generic,
    eq_iterable,
    eq_ndarray,
    is_condition,
    param_type,
)


def test_contains_condition_all_ints():
    conditions = [
        {"a": 3, "b": 3, "c": 3, "d": 1},
        {"a": 2, "b": 2, "c": 1, "d": 4},
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"a": 1, "b": 1, "c": 2, "d": 2},
        {"a": 4, "b": 4, "c": 5, "d": 2},
    ]
    condition_in = {"a": 1, "b": 2, "c": 3, "d": 4}
    condition_not_in = {"a": 4, "b": 1, "c": 2, "d": 4}
    assert contains_condition(conditions, condition_in)
    assert not contains_condition(conditions, condition_not_in)


def test_contains_condition_all_floats():
    conditions = [
        {"a": 3, "b": 3, "c": 3, "d": 1},
        {"a": 2, "b": 2, "c": 1, "d": 4},
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"a": 1, "b": 1, "c": 2, "d": 2},
        {"a": 4, "b": 4, "c": 5, "d": 2},
    ]
    condition_in = {"a": 1, "b": 2, "c": 3, "d": 4}
    condition_not_in = {"a": 4, "b": 1, "c": 2, "d": 4}
    assert contains_condition(conditions, condition_in)
    assert not contains_condition(conditions, condition_not_in)


def test_contains_condition_all_str():
    conditions = [
        {"a": "pineapple", "b": "banana", "c": "cherry"},
        {"a": "apple", "b": "ananas", "c": "cherry"},
        {"a": "apple", "b": "banana", "c": "sherry"},
        {"a": "apple", "b": "banana", "c": "cherry"},
    ]
    condition_in = {"a": "apple", "b": "banana", "c": "cherry"}
    condition_not_in = {"a": "apple", "b": "banana", "c": "pondicherry"}
    assert contains_condition(conditions, condition_in)
    assert not contains_condition(conditions, condition_not_in)


def test_contains_condition_mixed():
    conditions = [
        {"depth": (3, 7), "eta": 0.6, "objective": "logistic", "method": max},
        {"depth": (7, 3), "eta": 1.2, "objective": "logistic", "method": max},
        {"depth": (3, 7), "eta": 1.2, "objective": "linear", "method": max},
        {"depth": (3, 7), "eta": 1.2, "objective": "logistic", "method": max},
        {"depth": (3, 7), "eta": 1.2, "objective": "logistic", "method": min},
        {"depth": (3, 7), "eta": 1.2, "objective": "identity", "method": max},
        {"depth": (3, 6), "eta": 1.2, "objective": "logistic", "method": None},
    ]
    condition_in = {
        "depth": (3, 7),
        "eta": 1.2,
        "objective": "logistic",
        "method": max,
    }
    condition_not_in = {
        "depth": (3, 7),
        "eta": 0.3,
        "objective": "logistic",
        "method": max,
    }
    assert contains_condition(conditions, condition_in)
    assert not contains_condition(conditions, condition_not_in)


def test_contains_condition_mixed_with_errors():
    conditions = [
        {"depth": (3, 7), "eta": 0.6, "objective": "logistic", "error": 98.3},
        {"depth": (7, 3), "eta": 1.2, "objective": "logistic", "error": -2387},
        {"depth": (3, 7), "eta": 1.2, "objective": "linear", "error": 0.05},
        {"depth": (3, 7), "eta": 1.2, "objective": "logistic"},
        {"depth": (3, 7), "eta": 1.2, "objective": "logistic", "error": -34},
        {"depth": (3, 7), "eta": 1.2, "objective": "identity", "error": -0.005},
        {"depth": (3, 6), "eta": 1.2, "objective": "logistic", "error": 1.34},
    ]
    condition_in_with_error = {
        "depth": (3, 7),
        "eta": 1.2,
        "objective": "logistic",
        "error": 2.0000001,
    }
    condition_in_without_error = {
        "depth": (3, 7),
        "eta": 1.2,
        "objective": "logistic",
    }
    condition_not_in = {
        "depth": (3, 7),
        "eta": 0.3,
        "objective": "logistic",
        "method": max,
        "error": 23.54,
    }
    assert contains_condition(conditions, condition_in_with_error)
    assert contains_condition(conditions, condition_in_without_error)
    assert not contains_condition(conditions, condition_not_in)


@pytest.mark.parametrize(
    "a,expected",
    [
        (["this", 678, 42.42, True], False),
        (("arg", 98, True), False),
        ({"a": 6, "b": 100.01, "c": "burken"}, True),
        (
            {"depth": (3, 7), "eta": 1.2, "objective": "linear", "error": 0.05},
            True,
        ),
        ({"depth": (3, 7), "eta": 1.2, "objective": "logistic"}, True),
        (
            {
                "depth": (3, 7),
                "eta": 1.2,
                "objective": "logistic",
                "error": -34,
            },
            True,
        ),
        (1e86, False),
        (2.000000000001, False),
        (1.0, False),
        (-5e2, False),
        (6439.3298, False),
        (1e12 + 1.0, False),
        ([1, 2, 3], False),
        ([1, 2, 3, 4], False),
        ([1.1, 2.5, -10.5], False),
        (["a", "b"], False),
    ],
)
def test_is_condition(a: Any, expected: str) -> None:
    assert is_condition(a) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (9.0, 9.0, True),
        (-0.190, -0.190, True),
        (1.98754, 1.98754, True),
        (1e86, 1e86, True),
        (2.0, 2.000000000001, True),
        (1.0, 2.0, False),
        (-5e2, 5e2, False),
        (6439.3298, 6539.3298, False),
        (1e12, 1e12 + 1.0, True),
        ([1, 2, 3], [1, 2, 3], True),
        ([1, 2, 3, 4], [1, 2, 3], False),
        ([1, 2, 3], [1, 2, 3, 4], False),
        ([1.1, 2.5, -10.5], [1.1, 2.5, -10.5], True),
        ([1.1, 2.5, -10.5], [1.1, 2.5, -10.2], False),
        (["a", "b"], ["a", "b"], True),
        (["a", "b"], ["a", "b", "c"], False),
        ([True, False, True], [True, False, True], True),
        ([True], [False], False),
        ([], [], True),
        (["this", 678, 42.42, True], ["this", 678, 42.42, True], True),
        (["this", 678, 42.42, True], ["that", 678, 42.42, True], False),
        (["this", 678, 42.42, True], ["this", 679, 42.42, True], False),
        (["this", 678, 42.42, True], ["this", 678, 42.52, True], False),
        (["this", 678, 42.42, True], ["this", 678, 42.42, False], False),
        (("arg", 98, True), ("arg", 98, True), True),  # Compare tuples
        (("arg", 98, True), ("arg", 89, True), False),
        ((98, "arg", True), ("arg", 98, True), False),
        (
            {"a": 6, "b": 100.01, "c": "burken"},
            {"a": 6, "b": 100.01, "c": "burken"},
            True,
        ),
        (
            {"a": 6, "b": 100.01, "c": "burken"},
            {"a": 6, "b": 100.01, "c": "buckle"},
            False,
        ),
        ({"a": "done", "b": 200.01}, {"a": "200.1", "b": "done"}, False),
        ({}, {}, True),
        (np.array([9.2, 55.9, 0.12]), np.array([9.2, 55.9, 0.12]), True),
        (np.array([9.2, 0.12, 55.9]), np.array([9.2, 55.9, 0.12]), False),
        (2, 2, True),
        (-5, -5, True),
        (0, 0, True),
        (int(1e6), int(1e6), True),
        (2, 32, False),
        (-555, 35, False),
        (0, int(1e6), False),
        (int(1e12), int(1e12) + 1, False),
        ("lofty", "lofty", True),
        ("particular", "PARTICULAR", False),
        ("burrrlap", "bur__lap", False),
        ("MIN", "MIN", True),
        ("983", "983", True),
        ("L5.4", "L5-4", False),
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        ({3, 4, 5}, {3, 4, 5}, True),
        ({3, 4, 5}, {5, 4, 3}, True),
        ({3, 4, 5}, {3, 4, 5, 8}, False),
    ],
)
def test_eq(a: Any, b: Any, expected: bool) -> None:
    assert eq(a, b) == expected


@pytest.mark.parametrize(
    "a,expected",
    [
        (3, "int"),
        (-84, "int"),
        (int(0.000), "int"),
        (890.890, "float"),
        (-0.00005, "float"),
        (1e89, "float"),
        ("plabber", "str"),
        ("9283", "str"),
        ("max", "str"),
        (True, "bool"),
        (False, "bool"),
        ([7, "f", 9.2], "list"),
        (["a"], "list"),
        ([eq, min, max], "list"),
        (("a", "z", 0, "top"), "tuple"),
        ((7,), "tuple"),
        (([4, 6], [7, 3], [8, 3]), "tuple"),
        ({"i": 1, "j": 2, "k": 3}, "dict"),
        ({"ibog": (0, "c"), "blergh": None}, "dict"),
        ({(5, 3): "k", 73: [8, 2, 7]}, "dict"),
        (np.array([6, 7, 3, 7]), "ndarray"),
        (np.array([5.8, 7.1, 8.0, 5.2]), "ndarray"),
        (np.array(["d", "e", "f"]), "ndarray"),
        (eq, "func"),
        (min, "func"),
        (sorted, "func"),
        ({4, 8, 2}, "unknown"),
        (None, "unknown"),
    ],
)
def test_param_type(a: Any, expected: str) -> None:
    assert param_type(a) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (9.0, 9.0, True),
        (-0.190, -0.190, True),
        (1.98754, 1.98754, True),
        (1e86, 1e86, True),
        (2.0, 2.000000000001, True),
        (1.0, 2.0, False),
        (-5e2, 5e2, False),
        (6439.3298, 6539.3298, False),
        (1e12, 1e12 + 1.0, True),
    ],
)
def test_eq_float(a: float, b: float, expected: bool) -> None:
    assert eq_float(a, b) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1, 2, 3], [1, 2, 3], True),
        ([1, 2, 3, 4], [1, 2, 3], False),
        ([1, 2, 3], [1, 2, 3, 4], False),
        ([1.1, 2.5, -10.5], [1.1, 2.5, -10.5], True),
        ([1.1, 2.5, -10.5], [1.1, 2.5, -10.2], False),
        (["a", "b"], ["a", "b"], True),
        (["a", "b"], ["a", "b", "c"], False),
        ([True, False, True], [True, False, True], True),
        ([True], [False], False),
        ([], [], True),
        (["this", 678, 42.42, True], ["this", 678, 42.42, True], True),
        (["this", 678, 42.42, True], ["that", 678, 42.42, True], False),
        (["this", 678, 42.42, True], ["this", 679, 42.42, True], False),
        (["this", 678, 42.42, True], ["this", 678, 42.52, True], False),
        (["this", 678, 42.42, True], ["this", 678, 42.42, False], False),
        (("arg", 98, True), ("arg", 98, True), True),  # Compare tuples
        (("arg", 98, True), ("arg", 89, True), False),
        ((98, "arg", True), ("arg", 98, True), False),
    ],
)
def test_eq_iterable(a: Sequence, b: Sequence, expected: bool) -> None:
    assert eq_iterable(a, b) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (
            {"a": 6, "b": 100.01, "c": "burken"},
            {"a": 6, "b": 100.01, "c": "burken"},
            True,
        ),
        (
            {"a": 6, "b": 100.01, "c": "burken"},
            {"a": 6, "b": 100.01, "c": "buckle"},
            False,
        ),
        ({"a": "done", "b": 200.01}, {"a": "200.1", "b": "done"}, False),
        ({}, {}, True),
    ],
)
def test_eq_dict(a: dict[Any, Any], b: dict[Any, Any], expected: bool) -> None:
    assert eq_dict(a, b) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (np.array([9.2, 55.9, 0.12]), np.array([9.2, 55.9, 0.12]), True),
        (np.array([9.2, 0.12, 55.9]), np.array([9.2, 55.9, 0.12]), False),
    ],
)
def test_eq_nparray(a: Any, b: Any, expected: bool) -> None:
    assert eq_ndarray(a, b) == expected


def test_eq_func_same() -> None:
    assert eq_func(test_eq_iterable, test_eq_iterable)


def test_eq_func_diff() -> None:
    assert not eq_func(test_eq_iterable, test_eq_dict)


def test_eq_func_same_name() -> None:
    a = eq_dict
    b = eq_dict
    assert eq_func(a, b)


def test_eq_func_builtin() -> None:
    a = min
    b = min
    assert eq_func(a, b)


def test_eq_func_lambda() -> None:
    a = lambda x: x * 3.14
    b = lambda x: x * 3.14
    assert eq_func(a, b)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        # integers
        (2, 2, True),
        (-5, -5, True),
        (0, 0, True),
        (int(1e6), int(1e6), True),
        (2, 32, False),
        (-555, 35, False),
        (0, int(1e6), False),
        (int(1e12), int(1e12) + 1, False),
        # strings
        ("lofty", "lofty", True),
        ("particular", "PARTICULAR", False),
        ("burrrlap", "bur__lap", False),
        ("MIN", "MIN", True),
        ("983", "983", True),
        ("L5.4", "L5-4", False),
        # booleans
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        # sets
        ({3, 4, 5}, {3, 4, 5}, True),
        ({3, 4, 5}, {5, 4, 3}, True),
        ({3, 4, 5}, {3, 4, 5, 8}, False),
    ],
)
def test_eq_generic(a: Any, b: Any, expected: bool) -> None:
    assert eq_generic(a, b) == expected
