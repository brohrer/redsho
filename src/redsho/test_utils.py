from redsho.utils import get_random_condition, grid_expand


def test_grid_expand():
    conditions = {"c1": [4, 7, 9], "c2": [0.1, 0.3, 0.6, 0.8]}
    expanded = grid_expand(conditions)
    assert len(expanded) == len(conditions["c1"]) * len(conditions["c2"])
    assert len(expanded[3]) == 2
    assert expanded[6]["c1"] in conditions["c1"]
    assert expanded[11]["c2"] in conditions["c2"]


def test_get_random_condition():
    conditions = {"c1": [4, 7, 9], "c2": [0.1, 0.3, 0.6, 0.8]}
    rand_condition = get_random_condition(conditions)
    assert len(rand_condition) == 2
    assert rand_condition["c1"] in conditions["c1"]
    assert rand_condition["c2"] in conditions["c2"]

