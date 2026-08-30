import os
import shutil

import numpy as np
from pytest import fixture

from redsho.optimizer import ParallelRedsho, Redsho

N_ITER_DEFAULT = 1e3
REPORT_DIR_NAME = "temp_test_reports"


@fixture
def optimizer():
    opt = Redsho(
        n_iter=N_ITER_DEFAULT,
        report_dir=REPORT_DIR_NAME,
        verbose=False,
    )

    yield opt

    try:
        shutil.rmtree(REPORT_DIR_NAME)
    except OSError:
        print("test had difficulty deleting test reports")


@fixture
def parallel_optimizer():
    opt = ParallelRedsho(
        n_iter=N_ITER_DEFAULT,
        report_dir=REPORT_DIR_NAME,
        verbose=False,
    )

    yield opt

    try:
        shutil.rmtree(REPORT_DIR_NAME)
    except OSError:
        print("test had difficulty deleting test reports")


def evaluate(x=0, y=0):
    """
    The objective function is a 2D variant of the sinc function.
    """
    x0 = 1
    y0 = 1.5
    return -np.sinc(x - x0) * np.sinc(y - y0)


def test_redsho_instantiation(optimizer):
    assert optimizer.best_error == 1e10
    assert optimizer.best_condition is None
    assert optimizer.n_iter == N_ITER_DEFAULT
    assert optimizer.report_dir == REPORT_DIR_NAME
    assert os.path.isdir(optimizer.report_dir)


def test_redsho_optimization(optimizer):
    conditions = {
        "x": list(np.linspace(0, np.pi, 10)),
        "y": list(np.linspace(0, np.pi, 10)),
    }

    best_error, best_condition, _ = optimizer.optimize(evaluate, conditions)

    assert isinstance(best_error, np.float64)
    assert isinstance(best_condition["x"], np.float64)
    assert isinstance(best_condition["y"], np.float64)
    assert -1 <= best_error <= 1
    assert 0 <= best_condition["x"] <= np.pi
    assert 0 <= best_condition["y"] <= np.pi


def test_redsho_parallel_optimization(parallel_optimizer):
    conditions = {
        "x": list(np.linspace(0, np.pi, 10)),
        "y": list(np.linspace(0, np.pi, 10)),
    }

    best_error, best_condition, _ = parallel_optimizer.optimize(
        evaluate, conditions
    )

    assert isinstance(best_error, np.float64)
    assert isinstance(best_condition["x"], np.float64)
    assert isinstance(best_condition["y"], np.float64)
    assert -1 <= best_error <= 1
    assert 0 <= best_condition["x"] <= np.pi
    assert 0 <= best_condition["y"] <= np.pi
