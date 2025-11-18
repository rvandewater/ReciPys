<div align="center">
  <img src="https://github.com/rvandewater/ReciPies/blob/development/docs/figures/recipies_logo.svg?raw=true"
alt="recipies logo" height="300">
</div>

# ReciPies 🥧

<p align="center"><em>A declarative pipeline for reproducible ML preprocessing</em></p>

[![CI](https://github.com/rvandewater/ReciPies/actions/workflows/ci.yml/badge.svg)](https://github.com/rvandewater/ReciPies/actions/workflows/ci.yml)
![Platform](https://img.shields.io/badge/platform-linux--64%20%7C%20win--64%20%7C%20osx--64-lightgrey)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI version shields.io](https://img.shields.io/pypi/v/recipies.svg)](https://pypi.python.org/pypi/recipies/)
[![Python Version](https://img.shields.io/pypi/pyversions/recipies.svg)](https://pypi.python.org/pypi/recipies/)
[![Downloads](https://pepy.tech/badge/recipies)](https://pepy.tech/project/recipies)
[![arXiv](https://img.shields.io/badge/arXiv-2306.05109-b31b1b.svg)](http://arxiv.org/abs/2306.05109)
[![codecov](https://codecov.io/gh/rvandewater/ReciPies/graph/badge.svg?token=5L5KUN8I3F)](https://codecov.io/gh/rvandewater/ReciPies)

Modern machine learning (ML) workflows live or die by their data‑preprocessing steps, yet in Python—a language with a
rich ecosystem for data science and ML—these steps are often scattered across ad‑hoc scripts or opaque Scikit-Learn
(sklearn) snippets that are hard to read, audit, or reuse. `ReciPies` provides a concise, human‑readable, and fully
reproducible way to declare, execute, and share preprocessing pipelines, adhering to Configuration as Code principles.
It lets users describe transformations as a recipe made of ordered *steps* (e.g., imputing, encoding, normalizing)
applied to variables identified by semantic roles (predictor, outcome, ID, time stamp, etc.). Recipes can be *prepped*
(trained) once, *baked* many times, and cleanly separated between training and new data—preventing data leakage by
construction. Under the hood, `ReciPies` targets both Pandas and Polars backends for performance and flexibility, and
it is easily extensible: users can register custom steps with minimal boilerplate. Each recipe is serializable to
JSON/YAML for provenance tracking, collaboration, and publication, and integrates smoothly with downstream modeling
libraries. Packaging preprocessing as clear, declarative objects, `ReciPies` lowers the cognitive load of feature
engineering, improves reproducibility, and makes methodological choices explicit, benefiting individual researchers,
engineering teams, and peer reviewers alike.

The backend can either be [Polars](https://github.com/pola-rs/polars) or [Pandas](https://github.com/pandas-dev/pandas) dataframes.
The operation of this package is inspired by the R-package [recipes](https://recipes.tidymodels.org/). Please check the [documentation](rvandewater.github.io/ReciPies/) for more details.

## Installation

### Using `pip`

You can install ReciPies from pip using:

```
pip install recipies
```

### Using `uv`

You can install ReciPies using `uv` (the unified package manager) with the following command:

```bash
uv add recipies
```

> Note that the package is called `recipies` on pip.

### Developer / Editable install

```bash
# with conda (optional)
conda env update -f environment.yml
conda activate ReciPies
# with pip
pip install -e .
# with uv venv
uv venv && source .venv/bin/activate

```

## Getting Start

Here's a simple example of using ReciPies:

```python
# Import necessary libraries
import polars as pl
import numpy as np
from datetime import datetime, MINYEAR
from recipies import Ingredients, Recipe
from recipies.selector import all_numeric_predictors, all_predictors
from recipies.step import StepSklearn, StepHistorical, Accumulator, StepImputeFill
from sklearn.impute import MissingIndicator

# Set up random state for reproducible results
rand_state = np.random.RandomState(42)

# Create time columns for two different groups
timecolumn = pl.concat(
    [
        pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 5), "1h", eager=True),
        pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 3), "1h", eager=True),
    ]
)

# Create sample DataFrame
df = pl.DataFrame(
    {
        "id": [1] * 6 + [2] * 4,
        "time": timecolumn,
        "y": rand_state.normal(size=(10,)),
        "x1": rand_state.normal(loc=10, scale=5, size=(10,)),
        "x2": rand_state.binomial(n=1, p=0.3, size=(10,)),
        "x3": pl.Series(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"], dtype=pl.Categorical),
        "x4": pl.Series(["x", "y", "y", "x", "y", "y", "x", "x", "y", "x"], dtype=pl.Categorical),
    }
)

# Introduce some missing values
df = df.with_columns(pl.when(pl.int_range(pl.len()).is_in([1, 2, 4, 7])).then(None).otherwise(pl.col("x1")).alias("x1"))

df2 = df.clone()

# Create Ingredients and Recipe
ing = Ingredients(df)
rec = Recipe(ing, outcomes=["y"], predictors=["x1", "x2", "x3", "x4"], groups=["id"], sequences=["time"])

rec.add_step(StepSklearn(MissingIndicator(features="all"), sel=all_predictors()))
rec.add_step(StepImputeFill(sel=all_predictors(), strategy="forward"))
rec.add_step(StepHistorical(sel=all_predictors(), fun=Accumulator.MEAN, suffix="mean_hist"))

# Apply the recipe to the ingredients
df = rec.prep()

# Apply the recipe to a new DataFrame (e.g., test set)
df2 = rec.bake(df2)
```

## Core Concepts

Below is a schematic overview of ReciPies' architecture. We 1) load a Pandas or Polars (training) dataframe, then 2) wrap it in an
Ingredients object that maintains column role information (i.e., what does this column do in this dataset).
Next, we 3) define a Recipe consisting of multiple Steps that operate on selected columns.
Finally, we 4) prep the Recipe on the training data and 5) bake it on new data. We can then 6) run our ML pipeline on
train and test data.

<div>
  <img src="docs/figures/recipies_flow.svg" alt="recipies flowchart" height="800">
</div>
The main building blocks of ReciPies are:

- **Ingredients**: A wrapper around DataFrames that maintains column role information, ensuring data semantics are preserved during transformations.
- **Recipe**: A collection of processing steps that can be applied to Ingredients objects to create reproducible data pipelines.
- **Step**: Individual data transformation operations that understand column roles and can work with both Polars and Pandas backends.
- **Selector**: Utilities for selecting columns based on their roles or other criteria.

## Backend Support

ReciPies supports both Polars and Pandas backends:

- **Polars**: High-performance DataFrame library with lazy evaluation
- **Pandas**: Traditional DataFrame library with extensive ecosystem support

The package automatically detects the backend and provides a consistent API regardless of the underlying DataFrame implementation.

## Examples

Check out the `examples/` directory for Jupyter notebooks demonstrating various use cases of ReciPies.
Check out the `benchmarks/` directory for performance comparisons between Polars and Pandas backends.

## Contributing

Contributions are welcome! Please see our contributing guidelines and open an issue or submit a pull request on the [GitHub repository](https://github.com/rvandewater/ReciPies).

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/rvandewater/ReciPies/blob/main/LICENSE) file for details.

## How to cite

If you use this code in your research, please cite the following publication which uses ReciPys extensively to create a
customisable preprocessing pipeline (a standalone paper is in preparation):

```
@inproceedings{vandewaterYetAnotherICUBenchmark2024,
  title = {Yet Another ICU Benchmark: A Flexible Multi-Center Framework for Clinical ML},
  shorttitle = {Yet Another ICU Benchmark},
  booktitle = {The Twelfth International Conference on Learning Representations},
  author = {van de Water, Robin and Schmidt, Hendrik Nils Aurel and Elbers, Paul and Thoral, Patrick and Arnrich, Bert and Rockenschaub, Patrick},
  year = {2024},
  month = oct,
  urldate = {2024-02-19},
  langid = {english},
}
```

This paper can also be found on arxiv: [arxiv](https://arxiv.org/pdf/2306.05109.pdf).
