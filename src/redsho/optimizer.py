import copy
import os
from multiprocessing import (
    Pool,
    pool.AsyncResult as AsyncResult,

from typing import Callable

import numpy as np

import redsho.reporting as rep

# The type describing a condition, a collection of hyperparameters
# and their values. Values can be really be any type at all.
# A condition set includes a list of valid values for each hyperparameter.
# The full set of conditions can be created combinatorially, with each
# possible combination of valid values.
# Whereas a condition list is a straightforward list of individual conditions.
type Cond = dict[str, Any]
type CondSet = dict[str, list[Any]]
type CondList = list[Cond]


def optimize(
    conditions: CondSet,
    evaluate: Callable[[Cond], float],
    n_iter: int = int(1e8),
    parallel: bool = False,
    n_proc: int = os.cpu_count() - 1,
    report_dir: str = "reports",
    report_filename: str =  "optimizer_results.csv",
    report_plot_filename: str = "optimizer_results.png",
    verbose: bool = True,
) -> tuple[float, Cond]:
    """
    `conditions`: dict where
        each key is a condition_name and
        each value is a list of valid condition values
    Values can be any type.

    `evaluate`: function that takes keyword arguments,
    one for each key in the `conditions` dict.

    `n_iter`: the number of hyperparameter combinations (conditions)
    to try before giving up. By default this is a lare number and
    will likely result in an exhaustive grid search. Bring it down
    lower if you don't feel like waiting that long.

    `parallel`: flag for whether to run multiple optimization runs at once.
    Same as `optimize(), but faster because it runs several variants
    at the same time on different processors.

    The top-level implementation of Redsho.
    It's main purpose is to support
    the `optimize()` function, which does all the work.

    Calling `optimize()` searches through possible combinations of
    the hyperparameters and tries to find the best-performing
    (lowest error) combination.
    It does not assume that the error landscape is smooth or continuous.
    It takes more samples to find an optimimal combination than fancier
    methods that do makd these assumptions, like Bayesian or
    Gradient-based hyperparameter optimization.
    But it usually takes fewer samples than random or exhaustive grid search.
    (See https://en.wikipedia.org/wiki/Hyperparameter_optimization )

    Robust: it doesn't assume smoothness or continutiy

    Evolutionary: it randomly explores new options based on the most
        successful of its previous tries.

    Direction Set: It alternates through its hyperparameters, exploring in
    one "direction" (hyperparameter) at a time.
    """
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

    if parallel:
        best_error, best_condition = _optimization_loop_parallel(
            conditions,
            evaluate,
            n_iter,
            n_proc,
            report_path,
            report_plot_path,
            verbose,
        ):
    else:
        best_error, best_condition = _opimization_loop(
            conditions,
            evaluate,
            n_iter,
            report_path,
            report_plot_path,
            verbose,
        )

    return best_error, best_condition


def __optimization_loop(
    conditions: CondSet,
    evaluate: Callable[[Cond], float],
    n_iter: int,
    report_path: str,
    report_plot_path: str,
    verbose: bool,
) -> tuple[float, Cond]:
    """
    Cycle through candidate options and keep track of the best error
    seen so far.

    Keep going until `_generate_conditions()` runs out of candidates or
    until the user forcibly stops the run.
    """
    best_error: int = int(1e10)
    best_condition: Cond | None = None
    condition_history: CondList = []
    for condition: Cond in _generate_conditions(conditions, n_iter):
        if verbose:
            print("    Evaluating condition", condition)
        error: float = evaluate(**condition)
        condition["error"] = error
        condition_history.append(condition)
        rep.results_dict_list_to_csv(condition_history, report_path)

        # Keep track of the best-so-far answer.
        if error < best_error:
            best_error = error
            best_condition = condition
        if verbose:
            results_so_far = rep.results_csv_to_dict_list(report_path)
            rep.progress_report(results_so_far, report_plot_path)

    results_so_far = rep.results_csv_to_dict_list(report_path)
    rep.progress_report(results_so_far, report_plot_path)
    return best_error, best_condition


def _optimization_loop_parallel(
    conditions: CondSet,
    evaluate: Callable[[Cond], float],
    n_iter: int,
    n_proc: int,
    report_path: str,
    report_plot_path: str,
    verbose: bool,
) -> tuple[float, Cond]:
    """
    Farm out the evaluation of individual conditions to `multiprocessing.Pool`
    processes. Using multiprocessing instead of threading allows it to
    make use of several processors at once, rather than just sharing
    the one. It effectively allows it to take over your whole computer.

    This will speed up optimization if it is CPU-limited, but not if
    it is memory-limited. It's worth monitoring your computer's
    resources while running this.
    """
    best_error: int = 1e10
    best_condition: Cond | None = None
    condition_history: CondList = []
    results: list[Cond, AsyncResult] = []
    with Pool(processes=n_proc) as pool:
        for condition: Cond in _generate_conditions(conditions, n_iter):
            if verbose:
                print("    Evaluating condition", condition)

            error_placeholder: AsyncResult = pool.apply_async(
                evaluate, (), condition
            )
            results.append((condition, error_placeholder))

            if len(results) > n_proc:
                (
                    res_condition: Cond,
                    res_placeholder: AsyncResult,
                ) = results.pop(0)
                error: float = res_placeholder.get()
                res_condition["error"] = error
                condition_history.append(res_condition)
                rep.results_dict_list_to_csv(condition_history, report_path)

                if error < best_error:
                    best_error = error
                    best_condition = condition
                if verbose:
                    results_so_far = rep.results_csv_to_dict_list(
                        report_filename
                    )
                    rep.progress_report(results_so_far, report_plot_path)
    results_so_far = rep.results_csv_to_dict_list(report_path)
    rep.progress_report(results_so_far, report_plot_path)
    return best_error, best_condition


def _generate_conditions(
    conditions: CondSet,
    n_iter: int,
) -> Iterator[Cond]:
    """
    Core logic for the optimizer.
    Decide which combination of values to try next.
    Selection is random, but weighted heavily toward the most successful
    conditions seen so far.
    New conditions are generated until either
        - `n_iter`, the maximum number of iterations is reached
        - all possibile combinations of valid hyperparameters have been tried
        - or the user forcible stops the run.

    One hyperparameter (direction) is explored at a time.
    The order in which the hyperparameters are explored (the direction set)
    is random.
    """
    condition_names: list[str] = list(conditions.keys())
    np.random.shuffle(condition_names)

    conditions_evaluated: list[Cond] = []
    conditions_with_scores: list[Cond] = []
    children_to_evaluate: list[Cond] = []

    # Before starting in, how many random points to check.
    # Randomly seeding a few of these across the hyperparameter space
    # helps prevent getting stuck in a "bad luck" initial condition.
    n_initial_random_conditions: int = 2 * len(condition_names)
    for _ in range(n_initial_random_conditions):
        children_to_evaluate.append(get_random_condition(conditions))

    # How many parent conditions to try finding children for (and fail)
    # before declaring the parameter space sufficiently explored.
    # Because an exhaustive list of conditions is never generated,
    # there's no way for the algorithm to be certain when it has explored
    # it completely. This number of attempts is intended to establish
    # with reasonable confidence that the space is reasonably well explored.
    n_parents_to_try: int = 2 * len(condition_names)

    for _ in range(n_iter):
        # It the hopper has run out of candidates, collect a few more.
        if len(children_to_evaluate) == 0:
            try:
                _choose_more_children_to_evaluate(children_to_evaluate)
            except StopIteration:
                return

        child: Cond = children_to_evaluate.pop()

        Can I fold conditions_evaluated and conditions_with scores into one variable?
        # Keeping a copy of child condition means that it remains unmodified.
        # It's useful for checking whether a condition has been tested
        # already.
        conditions_evaluated.append(copy.deepcopy(child))

        # Keeping the original object is helpful too. We know that
        # it will have the evaluation error appended to it.
        # We can use it for determining which point to expand.
        conditions_with_scores.append(child)

        yield child

def _choose_more_children_to_evaluate(children_to_evaluate):
    """
    Once the `children_to_evaluate` queue is empty, this method
    repopulates it.

    The fraction of a hyperparameter's values that are considered
    as potential children
    controlled by `expansion_fraction`.
    When enumerating the values of a parameter,
    approximately what fraction of all of its
    values to consider for children.
    expansion_fraction = 0.3
    """
    parents = _choose_parents(n_parents_to_try)

    # Handle the case where things are still getting started
    # and there aren't enough evaluated points to choose parents.
    if parents is None:
        for _ in range(n_parents_to_try):
            children_to_evaluate.append(
                get_random_condition(conditions)
            )
        return

    for parent in parents:
        condition_names.insert(0, condition_names.pop())
        for condition_name in condition_names:
            children_to_evaluate += _choose_children_from_param(
                parent, condition_name
            )
            if len(children_to_evaluate) > 0:
                return
    # If it gets this far it means that there are no lines radiating
    # from this point that haven't yet been explored.
    # The algorithm is done.
    raise StopIteration


def _choose_parents(n_parents: int) -> list[Cond]:
    error_list: list[float] = []

    # Pull out all the errors that have been generated so far.
    # Conditions still being evaluated won't have errors associated
    # with them yet. Ignore these and move on.
    candidates: list[Cond] = [cond for cond in conditions_with_scores if "error" in cond]

    # No parents to consider just yet.
    if len(candidates) < 2:
        return None

    error_list: list[float] = [cond["error"] for cond in candidates]

    errors: Any = np.array(error_list)
    parents: list[Cond] = []

    # Assign a selection_weight to each condition,
    # based on the error score (or loss)
    # associated with it. We want to choose a low error condition
    # as a parent, but it doesn't have to be the lowest. We'll
    # randomly choose one, giving strong preference to conditions
    # with lower errors.
    # The lowest error will have a selection_weight of 1.
    # The highest error
    # will have a selection_weight of 0. An error halfway in between will
    # have a selection_weight of .5**2 = .25
    eps: float = 1e-17  # to avoid division by zero
    selection_weights: Any = (
        (np.max(errors) - errors + eps)
        / (np.max(errors) - np.min(errors) + eps)
    ) ** 2

    # Normalized, stacked selection weights, such that choosing
    # a random number between 0 and 1 comes up associated with
    # exactly one option
    norm_weight_stack: Any = (
        np.cumsum(selection_weights) /
        np.sum(selection_weights)
    )

    for _ in range(n_parents):
        # Find which option in the stack matches the random number
        i_cond: int = np.where(norm_weight_stack > np.random.uniform())[0][0]
        parents.append(
            copy.deepcopy(conditions_with_scores[int(i_cond)])
        )

    # TODO this seems unnecessary. Assume deterministic evaluation for now.
    # Strip the score information from the chosen parents.
    for parent in parents:
        try:
            del parent["error"]
        except KeyError:
            pass
    return parents


def _choose_children_from_param(
    parent,
    param,
    conditions,
    conditions_evaluated,
    expansion_fraction: float = 0.3,
):
    """
    Create a set of conditions for each
    value. This expands the starting condition `cond` along
    a single parameter `param`.
    """
    new_children_to_evaluate = []
    vals = conditions[param]
    for val in vals:
        new_cond = copy.deepcopy(parent)
        new_cond[param] = val
        if new_cond not in conditions_evaluated:
            new_children_to_evaluate.append(new_cond)

        # Randomly select just a few of the candidate children.
        np.random.shuffle(new_children_to_evaluate)
        n_children_max = int(
            np.ceil(len(list(vals)) * expansion_fraction)
        )
        if len(new_children_to_evaluate) > n_children_max:
            new_children_to_evaluate = new_children_to_evaluate[
                :n_children_max
            ]
    return new_children_to_evaluate
