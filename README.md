# redsho

Robust Evolutionary Direction Set Hyperparameter Optimizer

Redsho is a discrete optimizer, which means that
it works with parameters that can
only take on a certain set of values. This works well for evaluating neural
networks and other large and complex models, since they take so long to
train and test, and since the hyperparameters can have non-intuitive,
strongly non-linear, and interactive effects.


![Animated demo of Redsho in operation](https://github.com/brohrer/redsho/blob/main/landing_page_demo.gif?raw=true)

Redsho (pictured in action above), an evolutionary search algorithm variant
inspired by direction set methods like 
[Powell's method](https://en.wikipedia.org/wiki/Powell%27s_method), so much
so that it was originally called Evolutionary Powell's method.
Here is [a detailed description of how it works](https://brohrer.github.io/evopowell.html).

## Installation

It's on PyPI, so install with

```bash
pip install redsho
```

or as part of a uv environment

```bash
uv add redsho
```

If you want to experiment with tweaking the algorithm, clone the repository
to your local machine and install it from there.

```bash
git clone https://codeberg.org/brohrer/redsho.git
python3 -m pip install -e redsho
```

## Run the demo

In a python script

```python3
import redsho.demo
```

## Parallelization

Redsho can seamlessly take advantage of multiple processor systems.
Instead of 

```python
from redsho.optimizer import Redsho
```

try

```python
from redsho.parallel_optimizer import ParallelRedsho
```

It automatically recruits all your processors (but one) to do its bidding.
