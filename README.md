# redsho
Robust Evolutionary Direction Set Hyperparameter Optimizer

Redsho is a discrete optimizer, which means that
it works with parameters that can
only take on a certain set of values. This works well for evaluating
machine learning models with a collection of hyperparameters that
influence their behavior, such as gradient boosted decision trees
and all manner of neural networks. It's also useful for evaluting complex
systems where entire algorithmic blocks can be swapped in for each other.
It's intended especially for larger models and systems since they take
so long to train and test, and since the hyperparameters can have non-intuitive,
strongly non-linear, and interactive effects.

Redsho is an evolutionary search algorithm variant
inspired by direction set methods like 
[Powell's method](https://en.wikipedia.org/wiki/Powell%27s_method), so much
so that it was originally called Evolutionary Powell's method.
Here is [a detailed description of how it works](https://brohrer.github.io/evopowell.html).

![Animated demo of Redsho in operation on a 2D variant of the sinc function
](https://github.com/brohrer/redsho-examples/blob/main/sinc/splash.gif?raw=true)

This demo illustrates how redsho

1. Peppers the parameter space with a few starting points
2. Chooses neighboring points in varying directions to test
3. Leans toward neighbors of the highest performing points, but
4. Also explores more broadly

## Installation

It's on PyPI, so install as part of a uv environment (recommended)

```bash
uv add redsho
```

or via pip

```bash
pip install redsho
```

## Usage

Start with an evaluation function that takes some hyperparameters as
keyword arguments.

```python3
def evaluate(*, a, b, c):
    return a * b**3 % c
```

(the leading `*` tells Python that everything that follows is a required
keyword argument)

Build a collection of values to try for each parameter.

```python3
values = {
    "a": [4, 7, 9],
    "b": [2, 5, 6],
    "c": [5, 8, 11],
}
```

Call the optimizer.

```python3
from redsho.optimizer import optimize

lowest_error, best_parameters = optimize(evaluate, values)
```

where `error` is the lowest error achieved, `best_parameters` is the collection
of parameter values that achieved it.

## Examples

There are some standalone examples of how redsho can be used in a
separate repo called
[redsho-examples](https://codeberg.org/brohrer/redsho-examples/src/branch/main/README.md)
, including finding the maximum of a *sinc* function.

## Developing

If you want to extend or tweak the approach, then
[DEVELOPING.md](https://codeberg.org/brohrer/redsho/src/branch/main/DEVELOPING.md)
is for you.


### Some terminology
- REDSHO: robust evolutionary direction set hyperparameter optimizer
- robust: it doesn't assume smoothness or continutiy
- evolutionary: it randomly explores new options based on the most
    successful of its previous tries.
- direction set: it alternates through its hyperparameters, exploring
    by varying one parameter at a time
- hyperparameter: in this context, any variable that has an influence
    on the result of the evaluation function. Also referred to as
    a direction or a dimension.
- hyperparameter space: if each hyperparameter is a direction, then
    taken together, n hyperparameters form an n-dimensional space
- condition: a full collection of hyperparameter names and one valid value
    for each. Each condition is a point in the hyperparameter space.
- condition grid: the set of all conditions in the hyperparameter space.
    Forms an irregularly-spaced n-dimensional grid.
- evaluation function: pretty much anything that can take in a set of
    parameters and return a number that evaluates its performance.
    It can be a mathematical expression, a traditional machine learning
    model, an arbitrary chunk of Python code, whatever.
- error: the result, after a condition has been evaluated.
    Also called the "loss" is the context of machine learning.
    Lower is always better. (If you have a "higher is better" evaluator
    just slap a negative sign on it.) The algorithm
    is trying to find the condition with the lowest error.
- error landscape: picturing a two-dimensional hyperparameter space,
    the error can be imagined as the height of a mountainous landscape
    in an extra dimension of that space. This can be generalized to
    higher dimensions
    (harder to picture in your head, but the math doesn't mind).
    The goal of the algorithm is to find the bottom of the lowest valley.
- parent: a condition chosen as a point from which to select the next
    generation of conditions to evaluate. The fact that this is
    a direction set method means that one of the parent's parameters
    will be varied in to find candidates.
- child: conditions chosen based on a parent are its children.

### Assumptions
- assumes the evaluator is deterministic, that it will give the
    same answer every time for the same set of hyperparameters. If this
    is not the case for your application you can approximate a stochastic
    solution by running it several times and looking for a grouping in
    the winning conditions.
- assumes discrete valued parameters. Even if parameters are continuous,
    the way you feed them in forces you to choose a handful of specific
    values to try.

### Non-assumptions
- does not assume that the error landscape is smooth or continuous.
    As a result, redsho often takes more iterations to find
    an optimimal combination than fancier methods that that assume
    smoothness, like Bayesian or Gradient-based hyperparameter optimization.
    But it usually takes fewer samples than random or exhaustive grid search.
    (See https://en.wikipedia.org/wiki/Hyperparameter_optimization )
    And, on average, it settles into a "pretty good" solution rather quickly.
- does not assume that hyperparameter values are numerical.
    This opens up Redsho to handling categorical arguments, such as
    booleans or strings. It can also be used with string arguments
    for a Python function, or even whole functions, classes, or subsystems.
    Any valid Python object can be used as a hyperparameter value.
    This is helpful when evaluating models that have options such as
    `method` which can be assigned any one of several strings or
    arguments that accept functions, similar to how to
    SciPy's `optimize.minimize` does.
