import os
import shutil

from common import (
    default_condition_grid,
    default_evaluate,
    default_report_dir,
    default_report_filename,
    default_report_path,
    default_report_plot_filename,
    default_report_plot_path,
)

from redsho.helpers.equals import is_condition
from redsho.optimizer import (
    optimization_loop,
    optimize,
)


def test_optimize() -> None:
    lowest_error, winning_condition = optimize(
        default_condition_grid,
        default_evaluate,
        report_dir=default_report_dir,
        report_filename=default_report_filename,
        report_plot_filename=default_report_plot_filename,
    )
    assert isinstance(lowest_error, float)
    assert is_condition(winning_condition)

    cleanup()


def test_optimization_loop() -> None:
    os.makedirs(default_report_dir, exist_ok=True)
    best_err, best_cond = optimization_loop(
        default_condition_grid,
        default_evaluate,
        n_iter=int(1e2),
        report_path=default_report_path,
        report_plot_path=default_report_plot_path,
        verbose=True,
    )
    assert isinstance(best_err, float)
    assert is_condition(best_cond)

    cleanup()


def cleanup():
    try:
        shutil.rmtree(default_report_dir)
    except OSError:
        print("test had difficulty deleting test reports")
