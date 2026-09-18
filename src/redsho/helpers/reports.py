import csv

import matplotlib.pyplot as plt
import numpy as np

from redsho.helpers.helpers import (
    Cond,
    CondList,
)

plt.switch_backend("agg")


def condition_list_to_csv(conditions: CondList, csv_filename: str) -> None:
    """
    Lifted from Matthew Flaschen on StackOverflow
    https://stackoverflow.com/questions/3086973/how-do-i-convert-this-list-of-dictionaries-to-a-csv-file
    """
    with open(csv_filename, "wt") as csv_file:
        dict_writer = csv.DictWriter(csv_file, conditions[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(conditions)


def csv_to_condition_list(csv_filename: str) -> CondList:
    with open(csv_filename, "rt") as csv_file:
        dict_reader = csv.DictReader(csv_file)
        return list(dict_reader)


def progress_report(conditions: CondList, filename: str) -> None:
    """
    Show how the best-so-far error value has evolved.
    """
    best_so_far_error: float = 1e10
    best_so_far_errors: list[float] = []
    best_so_far_params: Cond = {}
    for condition in conditions:
        try:
            error: float = float(condition["error"])
        except KeyError:
            continue
        if not np.isnan(error):
            if error < best_so_far_error:
                best_so_far_error = error
                best_so_far_params = condition
            best_so_far_errors.append(best_so_far_error)

    fig = plt.Figure()
    ax = fig.gca()
    ax.plot(np.arange(len(best_so_far_errors)) + 1, best_so_far_errors)
    ax.set_xlabel("Parameter combinations tested")
    ax.set_ylabel("Best error so far")
    param_msg = ""
    for key, value in best_so_far_params.items():
        param_msg += f"{key}: {value}\n"
    ax.text(
        len(best_so_far_errors) - 1,
        np.max(best_so_far_errors),
        param_msg,
        horizontalalignment="right",
        verticalalignment="top",
    )
    fig.savefig(filename, dpi=300)
