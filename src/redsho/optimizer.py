import copy
import os
from multiprocessing import Pool

import numpy as np

import redsho.toolbox as tb


class Redsho:
    """
    An evolutionary direction set method.
    """

    def __init__(
        self,
        n_iter=1e10,
        report_dir="reports",
        report_filename="opt_results.csv",
        report_plot_filename="opt_results.png",
        verbose=True,
    ):
        self.best_error = 1e10
        self.best_condition = None
        self.n_iter = int(n_iter)
        self.verbose = verbose

        self.report_dir = report_dir
        self.report_filename = os.path.join(self.report_dir, report_filename)
        self.report_plot_filename = os.path.join(
            self.report_dir, report_plot_filename
        )

        # Ensure that the report directory exists
        os.makedirs(self.report_dir, exist_ok=True)

        if self.verbose:
            print()
            print("This might take a while.")
            print("    You can check on the best-so-far solution at any time")
            print(f"    in {self.report_plot_filename}")
            print("    The full results log is maintained")
            print(f"    in {self.report_filename}")
            print()

    def optimize(self, evaluate, conditions):
        condition_history = []
        for condition in self._condition_generator(conditions):
            if self.verbose:
                print("    Evaluating condition", condition)
            error = evaluate(**condition)
            condition["error"] = error
            condition_history.append(condition)
            tb.results_dict_list_to_csv(condition_history, self.report_filename)

            if error < self.best_error:
                self.best_error = error
                self.best_condition = condition
            if self.verbose:
                results_so_far = tb.results_csv_to_dict_list(
                    self.report_filename
                )
                tb.progress_report(results_so_far, self.report_plot_filename)
        results_so_far = tb.results_csv_to_dict_list(self.report_filename)
        tb.progress_report(results_so_far, self.report_plot_filename)
        return self.best_error, self.best_condition, self.report_filename

    def _condition_generator(self, conditions):
        condition_names = list(conditions.keys())
        np.random.shuffle(condition_names)

        # When enumerating the values of a parameter,
        # approximately what fraction of all of its
        # values to consider for children.
        expansion_fraction = 0.3

        conditions_evaluated = []
        conditions_with_scores = []
        children_to_evaluate = []

        # Before starting in, how many random points to check.
        # Having a few of these helps prevent getting stuck in a
        # "bad luck" initial condition.
        n_initial_random_conditions = 2 * len(condition_names)
        for _ in range(n_initial_random_conditions):
            children_to_evaluate.append(tb.get_random_condition(conditions))

        # How many parent conditions to try finding children for (and fail)
        # before declaring the parameter space sufficiently explored.
        n_parents_to_try = 3

        def _choose_more_children_to_evaluate(children_to_evaluate):
            """
            Once the `children_to_evaluate` queue is empty, this method
            repopulates it.
            """
            parents = _choose_parents(n_parents_to_try)

            # Handle the case where things are still getting started
            # and there aren't enough evaluated points to choose parents.
            if parents is None:
                for _ in range(n_parents_to_try):
                    children_to_evaluate.append(
                        tb.get_random_condition(conditions)
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

        def _choose_parents(n_parents):
            error_list = []

            # Pull out all the errors that have been generated so far.
            # Conditions still being evaluated won't have errors associated
            # with them yet. Ignore these and move on.
            for cond in conditions_with_scores:
                try:
                    error_list.append(cond["error"])
                except KeyError:
                    pass

            # No parents to consider just yet.
            if len(error_list) < 2 or n_parents < 1:
                return None

            errors = np.array(error_list)
            parents = []

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
            eps = 1e-17  # to avoid division by zero
            selection_weights = (
                (np.max(errors) - errors)
                / (np.max(errors) - np.min(errors) + eps)
            ) ** 2

            for _ in range(n_parents):
                i_selection_weight = np.where(
                    selection_weights > np.random.uniform()
                )[0][0]
                chosen_selection_weight = selection_weights[i_selection_weight]

                i_cond = np.where(selection_weights == chosen_selection_weight)[
                    0
                ][0]

                if i_cond.size > 1:
                    i_cond = np.random.choice(i_cond)
                parents.append(
                    copy.deepcopy(conditions_with_scores[int(i_cond)])
                )

            # Strip the score information from the chosen parents.
            for parent in parents:
                try:
                    del parent["error"]
                except KeyError:
                    pass
            return parents

        def _choose_children_from_param(parent, param):
            """
            For a given parameter, create a set of conditions for each
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

        for _ in range(self.n_iter):
            if len(children_to_evaluate) == 0:
                try:
                    _choose_more_children_to_evaluate(children_to_evaluate)
                except StopIteration:
                    return

            child = children_to_evaluate.pop()
            # Keeping a copy of child means that it remains unmodified.
            # It's useful for checking whether a condition has been tested
            # already.
            conditions_evaluated.append(copy.deepcopy(child))
            # Keeping the original object is helpful too. We know that
            # it will have the evaluation error appended to it.
            # We can use it for determining which point to expand.
            conditions_with_scores.append(child)
            yield child


class ParallelRedsho(Redsho):
    """
    An evolutionary direction set method.
    Faster because it runs several variants at once.
    """

    def optimize(self, evaluate, conditions):
        condition_history = []
        results = []
        nproc = os.cpu_count() - 1
        with Pool(processes=nproc) as pool:
            for condition in self._condition_generator(conditions):
                if self.verbose:
                    print("    Evaluating condition", condition)

                error_placeholder = pool.apply_async(evaluate, (), condition)
                results.append((condition, error_placeholder))

                if len(results) > nproc:
                    res_condition, res_placeholder = results.pop(0)
                    error = res_placeholder.get()
                    res_condition["error"] = error
                    condition_history.append(res_condition)
                    tb.results_dict_list_to_csv(
                        condition_history, self.report_filename
                    )

                    if error < self.best_error:
                        self.best_error = error
                        self.best_condition = condition
                    if self.verbose:
                        results_so_far = tb.results_csv_to_dict_list(
                            self.report_filename
                        )
                        tb.progress_report(
                            results_so_far, self.report_plot_filename
                        )
        results_so_far = tb.results_csv_to_dict_list(self.report_filename)
        tb.progress_report(results_so_far, self.report_plot_filename)
        return self.best_error, self.best_condition, self.report_filename
