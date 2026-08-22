# redsho

Robust Evolutionary Direction Set Hyperparameter Optimizer

It's a discrete optimizer, which means that it works with parameters that can
only take on a certain set of values. This works well for evaluating neural
networks and other large and complex models, since it takes so long to
train and test them, and since the hyperparameters can have non-intuitive,
strongly non-linear, and interactive effects.


![REDSHO animated](/redsho/landing_page_demo.gif)

REDSHO (pictured in action above), an evolutionary search algorithm variant
inspired by direction set methods like 
[Powell's method](https://en.wikipedia.org/wiki/Powell%27s_method), so much
so that it was originally called Evolutionary Powell's method .
Here is [a detailed description of how it works](https://brohrer.github.io/evopowell.html).
As far as I know this method is novel and has not previously been published.
Please let me know if you've seen something like it before.

## Installation

Clone the repository to your local machine and install it from there.

```bash
git clone https://codeberg.org/brohrer/redsho.git
python3 -m pip install -e redsho
```

## Run the demo

```bash
python3
```
```python3
>>> import redsho.demo
```

## Parallelization

REDSHO can seamlessly take advantage of multiple processor systems.
Instead of 

```python
import redsho.redsho
```

try

```python
import redsho.redsho_parallel
```

It automatically recuits all your processors but one to do its bidding.
