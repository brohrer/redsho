"""
Implementation of the Redsho algorithm for finding the best performing
combination of parameters for an arbitrary evaluation function.
Built to help find good hyperparameter sets for machine learning models.
"""

import copy
import os
import random
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from redsho.helpers.equals import contains_condition
from redsho.helpers.helpers import (
    Cond,
    CondGrid,
    CondList,
    get_random_condition,
    low_biased_choice,
)
from redsho.helpers.reports import (
    condition_list_to_csv,
    progress_report,
)


def optimize(
    condition_grid: CondGrid,
    evaluate: Callable[[Cond], float],
    n_iter: int = int(1e3),
    n_processors: int = 1,
    report_dir: str = "reports",
    report_filename: str = "optimizer_results.csv",
    report_plot_filename: str = "optimizer_results.png",
    verbose: bool = True,
) -> tuple[float, Cond]:
    """
    Calling `optimize()` searches through possible combinations of
    the hyperparameters and tries to find the best-performing
    (lowest error) combination.

    `condition_grid`: a description the entire condition space, expressed
        as a dict where each key is a condition_name and
        each value is a list of valid condition values.

    `evaluate`: function that takes keyword arguments,
        one for each key in the `conditions` dict.

    `n_iter`: the number of conditions to evaluate before giving up.
        By default this is a large number and
        will likely result in an exhaustive grid search. Bring it down
        lower if you don't feel like waiting that long.

    `n_processors`: the number of processors to recruit for running
       multiple condition evaluations in parallel.
       `n_processors = 1` (default) will keep everything running serially
       on one processor. Limited to the total number of CPU cores on the
       machine, minus one, in order to avoid freezing things up.

    """
    # Make sure the inputs are valid
    assert isinstance(condition_grid, dict), (
        "First argument must be a condition grid, "
        + "a dict with items of the form\n"
        + "    { <parameter_name>: <list of allowable parameter values>, }"
    )
    assert len(condition_grid) > 1, (
        "First argument must have at least one item. "
        + "There needs to be at least one parameter to optimize over."
    )
    for k, v in condition_grid.items():
        assert isinstance(k, str), "Parameter names must be strings."
        assert isinstance(v, list), (
            "Allowable parameter values must be in a list."
        )
        assert len(v) > 0, (
            "Each parameter must have at least one allowable value."
        )

    # Check that `evaluate` is a function
    assert callable(evaluate), (
        "The second argument must be a function. "
        + "It takes a set of parameters and returns a float."
    )

    n_iter = int(n_iter)
    assert isinstance(n_iter, int), "n_iter must be an int."
    n_processors = int(n_processors)
    assert isinstance(n_processors, int), "n_processors must be an int."

    report_path: str = os.path.join(report_dir, report_filename)
    report_plot_path: str = os.path.join(report_dir, report_plot_filename)

    # Ensure that the report directory exists
    os.makedirs(report_dir, exist_ok=True)

    if verbose:
        print()
        print("This might take a while.")
        print("    You can check on the best-so-far solution at any time")
        print(f"    in {report_plot_path}")
        print("    The full results log is maintained")
        print(f"    in {report_path}")
        print()

    best_error, best_condition = optimization_loop(
        condition_grid,
        evaluate,
        n_iter,
        report_path,
        report_plot_path,
        verbose,
    )

    return best_error, best_condition


def optimization_loop(
    condition_grid: CondGrid,
    evaluate: Callable[[Cond], float],
    n_iter: int,
    report_path: str,
    report_plot_path: str,
    verbose: bool,
) -> tuple[float, Cond]:
    """
    Cycle through candidate options and keep track of the best error
    seen so far.

    Keep going until `generate_conditions()` runs out of candidates or
    until the user forcibly stops the run.
    """
    best_error: float = 1e10
    best_condition: Cond = {}

    # `conditions` will be the data structure for keeping track of which
    # conditions have been chosen, which have been evaluated, and what
    # their error values are. It is a growing list of individual conditions
    # `i_next_up` is the index of the condition that needs to be evaluated.
    # Once a condition has been evaluated, it gets a new key added, `error`,
    # with its attendant float-valued result.
    #
    # `conditions` gets intentionally passed around by reference so that
    # every function can write to it and make changes. This is a little
    # messy, and adds some cognitive burden when trying to chase it through
    # the code, but I felt like it was simpler overall than wrapping
    # these all in a class and more aestheically pleasing than clumsily
    # declaring it global.
    conditions: CondList = []

    for i_condition in generate_conditions(conditions, condition_grid, n_iter):
        condition: Cond = conditions[i_condition]
        if verbose:
            print("    Evaluating condition", condition)
        error: float = evaluate(condition)
        condition["error"] = error

        # Keep track of the best-so-far answer.
        if error < best_error:
            best_error = error
            best_condition = condition
        if verbose:
            progress_report(conditions, report_plot_path)

        condition_list_to_csv(conditions, report_path)

    progress_report(conditions, report_plot_path)
    return best_error, best_condition


def generate_conditions(
    conditions: CondList,
    condition_grid: CondGrid,
    n_iter: int,
) -> Iterator[int]:
    """
    Core logic for the optimizer.
    Decide which combination of values to try next.
    Selection is random, but weighted heavily toward the most successful
    conditions seen so far.
    New conditions are generated until either
        - `n_iter`, the maximum number of iterations is reached
        - all possibile combinations of valid hyperparameters have been tried
        - or the user forcibly stops the run.

    One hyperparameter (direction) is explored at a time.
    The order in which the hyperparameters are explored (the direction set)
    is random.
    """
    # Before starting in, how many random points to check.
    # Randomly seeding a few of these across the hyperparameter space
    # helps prevent getting stuck in a "bad luck" initial condition.
    n_initial_random_conditions: int = len(condition_grid)
    for _ in range(n_initial_random_conditions):
        new_condition = get_random_condition(condition_grid)
        if new_condition not in conditions:
            conditions.append(new_condition)

    for i_condition in range(n_iter):
        # If the hopper has run out of candidates, collect a few more.
        if len(conditions) <= i_condition:
            try:
                choose_more_conditions(conditions, condition_grid)
            except StopIteration:
                break

        yield i_condition


def choose_more_conditions(
    conditions: CondList,
    condition_grid: CondGrid,
    n_conditions_to_add: int = 3,
) -> None:
    """
    Once the conditions list runs out of new conditions to evaluate
    this method repopulates it.

    `n_conditions_to_add` is the number of new conditions that the algorithm
    aims to add to the list. It might fall short of this, and that's OK,
    but it won't exceed this. This is a hyperparameter that helps
    control how Redsho works, but it's not exposed. It's not expected
    to affect the result in an important way.
    """

    # How many parent conditions to try finding children for (and fail)
    # before declaring the parameter space sufficiently explored.
    # Because an exhaustive list of conditions is never generated,
    # there's no way for the algorithm to be certain when it has explored
    # it completely. This number of attempts is intended to establish
    # with reasonable confidence that the space is reasonably well explored.
    n_parents_to_try: int = len(condition_grid)

    parents: CondList = choose_parents(conditions, n_parents_to_try)

    # Handle the case where things are still getting started
    # and there aren't enough evaluated points to choose parents.
    if len(parents[0]) == 0:
        for _ in range(n_parents_to_try):
            new_condition = get_random_condition(condition_grid)
            if new_condition not in conditions:
                conditions.append(new_condition)
        return

    for parent in parents:
        condition_names: list[str] = list(condition_grid.keys())
        np.random.shuffle(condition_names)
        for condition_name in condition_names:
            success: bool = choose_children(
                conditions,
                condition_grid,
                parent,
                condition_name,
                n_children_max=n_conditions_to_add,
            )
            if success:
                return
    # If it gets this far it means that there are no lines radiating
    # from any of the parents that haven't yet been explored.
    # The algorithm is done.
    raise StopIteration


def choose_parents(
    conditions: CondList,
    n_parents: int,
    greediness: float = 2.0,
) -> CondList:
    """
    Select which previously evaluated parents to use as seed conditions for
    choosing new children. As in any evolutionary algorithm,
    high-performing parents are the most desirable seeds. But to keep
    from getting trapped in local patterns or weirdly well-performing
    conditions, introduce some randomness into the process. Parents
    are chosen by a performance-weighted random selection.

    `greediness`: a parameter that controls how heavily the algorithm
        leans toward choosing the highest performing condition. At
        `greediness = -inf`, the process is a roll of the dice
        amongst all the options, regardless
        of performance. At `greediness = +inf` it is winner-take-all.
        At `greediness = 0`, selection is weighted proportionally, based on
        how low the error is relative to the others in the list.
        This parameter isn't currently exposed at the top level,
        but if it proves useful, I'll change that.
    """
    parents: CondList = [{}]

    # Conditions still being evaluated won't have errors associated
    # with them yet. Find the conditions that have.
    evaluated_conditions: CondList = [
        cond for cond in conditions if "error" in cond
    ]

    # If no potential parents to consider yet, return an empty list.
    if len(evaluated_conditions) == 0:
        return parents

    # If the pool of potential parents is small, return them all.
    if len(evaluated_conditions) < n_parents:
        return evaluated_conditions

    # If the pool of potential parents is large enough, choose some
    # of the most promising.
    error_list: list[float] = [cond["error"] for cond in evaluated_conditions]

    errors: Any = np.array(error_list)

    i_parents = low_biased_choice(n_parents, errors, greediness=greediness)
    parents = [evaluated_conditions[i] for i in i_parents]

    return parents


def choose_children(
    conditions: CondList,
    condition_grid: CondGrid,
    parent: Cond,
    param: str,
    n_children_max: int = 3,
) -> bool:
    """
    Create a set of child conditions, a parent condition.
    This expands the starting condition `parent` along
    a single parameter `param`.

    `n_children_max`: the target number of child conditions to collect.
    The algorithm will attempt to add this many previously unevaluated
    conditions to the list. It will add no more, but may add fewer.

    `returns`: was the adding process successful? Was at least one new child
    added to the list?
    """
    success: bool = False
    vals: CondList = list(condition_grid[param])

    # The parent condition doesn't need to be considered.
    vals.remove(parent[param])
    # Randomly select which child candidates to test.
    random.shuffle(vals)

    n_children: int = 0
    for val in vals:
        new_cond: Cond = copy.deepcopy(parent)
        new_cond[param] = val
        if not contains_condition(conditions, new_cond):
            conditions.append(new_cond)
            success = True
            n_children += 1
        if n_children >= n_children_max:
            break

    return success
