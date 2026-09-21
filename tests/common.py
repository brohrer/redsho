import os
from collections.abc import Callable

from redsho.helpers.helpers import (
    Cond,
    CondGrid,
    CondList,
)


def add(a: float, b: float) -> float:
    return a + b


def multiply(a: float, b: float) -> float:
    return a * b


default_condition_grid: CondGrid = {
    "p0": [2, 5, 8],
    "p1": [0.08, 0.42, 1.414, 2.738, 6.28],
    "p2": ["six", "eleven", "eighteen", "ninety-nine"],
    "p3": [add, multiply, pow],
}
default_conditions: CondList = [
    {"p0": 2, "p1": 2.738, "p2": "eleven", "p3": add, "error": 3.3},
    {"p0": 5, "p1": 0.08, "p2": "eleven", "p3": add, "error": -0.6},
    {"p0": 8, "p1": 2.738, "p2": "six", "p3": add},
    {"p0": 8, "p1": 2.738, "p2": "eleven", "p3": add, "error": 0.1},
    {"p0": 8, "p1": 2.738, "p2": "eleven", "p3": multiply, "error": 13.0},
    {"p0": 8, "p1": 2.738, "p2": "eleven", "p3": pow, "error": 0.0},
]

default_parent: Cond = {"p0": 8, "p1": 2.738, "p2": "eleven", "p3": add}
default_param: str = "p2"
empty_param: str = "p1"
full_param: str = "p3"


small_condition_grid: CondGrid = {
    "p0": [2, 5],
    "p1": [0.08, 0.42],
}
small_condition_list_full: CondList = [
    {"p0": 2, "p1": 0.08, "error": 3.3},
    {"p0": 2, "p1": 0.42, "error": -7.1},
    {"p0": 5, "p1": 0.08},
    {"p0": 5, "p1": 0.42, "error": 12.9},
]


default_report_dir: str = "temp_test_reports"
default_report_filename: str = "optimizer_results.csv"
default_report_plot_filename: str = "optimizer_results.png"
default_report_path: str = os.path.join(
    default_report_dir, default_report_filename
)
default_report_plot_path: str = os.path.join(
    default_report_dir, default_report_plot_filename
)


def default_evaluate(
    *,  # ensures everything after will be a required keyword argument
    p0: int,
    p1: float,
    p2: str,
    p3: Callable,
) -> float:

    p0f: float = float(p0)
    p2a: float = 0

    if p2 == "six":
        p2a = 6.0
    elif p2 == "eleven":
        p2a = 11.0
    elif p2 == "eighteen":
        p2a = 18.0
    elif p2 == "ninety-nine":
        p2a = 99.0
    else:
        raise ValueError("p2 value not valid")

    error: float = p3(p0f, p1) % p2a
    return error
