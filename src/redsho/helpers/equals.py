import copy
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from redsho.helpers.helpers import (
    Cond,
    CondList,
)


def is_condition(arg: Any) -> bool:
    if not isinstance(arg, dict):
        return False
    for k in arg.keys():
        if not isinstance(k, str):
            return False
    return True


def contains_condition(conditions: CondList, condition: Cond) -> bool:
    # Check that there's at least one condition in the list
    for condition_from_list in conditions:
        clean_condition_from_list: Cond = copy.deepcopy(condition_from_list)
        clean_condition: Cond = copy.deepcopy(condition)

        # If the condition has an error attatched, hide it so that
        # the comparison will just be between the parameters.
        try:
            del clean_condition_from_list["error"]
        except KeyError:
            pass

        try:
            del clean_condition["error"]
        except KeyError:
            pass

        if eq(clean_condition_from_list, clean_condition):
            return True
    return False


def eq(a: Any, b: Any):
    eq_method: dict[str, Callable] = {
        "int": eq_generic,
        "float": eq_float,
        "str": eq_generic,
        "bool": eq_generic,
        "list": eq_iterable,
        "tuple": eq_iterable,
        "dict": eq_dict,
        "func": eq_func,
        "ndarray": eq_ndarray,
        "unknown": eq_generic,
    }

    type_a: str = param_type(a)
    type_b: str = param_type(b)
    if type_a != type_b:
        return False

    return eq_method[type_a](a, b)


def param_type(a: Any):
    # Figure out the type of each value
    if isinstance(a, bool):
        return "bool"
    elif isinstance(a, int):
        return "int"
    elif isinstance(a, float):
        return "float"
    elif isinstance(a, str):
        return "str"
    elif isinstance(a, list):
        return "list"
    elif isinstance(a, tuple):
        return "tuple"
    elif isinstance(a, dict):
        return "dict"
    elif isinstance(a, np.ndarray):
        return "ndarray"
    elif callable(a):
        return "func"
    else:
        return "unknown"


def eq_float(a: float, b: float) -> bool:
    epsilon: float = max(1e-20, abs(a) * 1e-4)
    return abs(a - b) < epsilon


def eq_iterable(a: Sequence[Any], b: Sequence[Any]) -> bool:
    if len(a) != len(b):
        return False

    for i in range(len(a)):
        if not eq(a[i], b[i]):
            return False
    return True


def eq_dict(a: dict[str, Any], b: dict[str, Any]) -> bool:
    # Check that they have the same number of keys
    if len(a) != len(b):
        return False

    for key, val in a.items():
        if key not in b:
            return False
        if not eq(val, b[key]):
            return False
    return True


def eq_ndarray(a: Any, b: Any) -> bool:
    return np.array_equal(a, b)


def eq_func(a: Callable, b: Callable) -> bool:
    # If the two functions have the same bytecode, consider them the same
    return a.__name__ == b.__name__


def eq_generic(a: Any, b: Any) -> bool:
    # A fallback check if none of the others seem to fit
    return a == b
