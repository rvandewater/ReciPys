<div class="centered hero-logo">
  <img src="https://github.com/rvandewater/ReciPies/blob/development/docs/figures/recipies_logo.svg?raw=true" alt="recipies logo">
</div>

<p align="center"><em>A declarative pipeline for reproducible ML preprocessing</em></p>

[![CI](https://github.com/rvandewater/ReciPies/actions/workflows/ci.yml/badge.svg)](https://github.com/rvandewater/ReciPies/actions/workflows/ci.yml)
![Platform](https://img.shields.io/badge/platform-linux--64%20|%20win--64%20|%20osx--64-lightgrey)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI version shields.io](https://img.shields.io/pypi/v/recipies.svg)](https://pypi.python.org/pypi/recipies/)
[![Python Version](https://img.shields.io/pypi/pyversions/recipies.svg)](https://pypi.python.org/pypi/recipies/)
[![Downloads](https://pepy.tech/badge/recipies)](https://pepy.tech/project/recipies)
[![arXiv](https://img.shields.io/badge/arXiv-2306.05109-b31b1b.svg)](http://arxiv.org/abs/2306.05109)

ReciPies is a Python package for feature engineering and data preprocessing with a focus on medical and clinical data.
It provides a unified interface for working with both Polars and Pandas DataFrames while maintaining column role
information throughout data transformations.

## Summary

- Declarative, reproducible data preprocessing
- Human-readable and transparent pipelines
- No trade-off between readability, performance, or flexibility
- Backend flexibility: works with Polars and Pandas
- Reduces cognitive overhead in feature engineering

## Installation

```bash
pip install recipies
```

For development:

```bash
git clone https://github.com/rvandewater/ReciPies.git
cd ReciPies
pip install -e '.[dev]'
```

## Quick Start

```python
import polars as pl
from recipies import Ingredients, Recipe
from recipies.selector import all_numeric_predictors, all_predictors
from recipies.step import StepSklearn, StepHistorical, Accumulator, StepImputeFill
from sklearn.impute import MissingIndicator

df_train = pl.read_parquet("path_to_your_data.parquet")
ing = Ingredients(df_train)
rec = Recipe(ing, outcomes=["y"], predictors=["x1", "x2"], groups=["id"], sequences=["time"]) 
rec.add_step(StepSklearn(MissingIndicator(features="all"), sel=all_predictors()))
rec.add_step(StepImputeFill(sel=all_predictors(), strategy="forward"))
rec.add_step(StepHistorical(sel=all_predictors(), fun=Accumulator.MEAN, suffix="mean_hist"))
```
Now prep your data (train the steps and convert the data):
``` python
df_train_preprocessed = rec.prep()
```
Now use the prepped recipe to also process your test data without information leakage:
``` python
df_test = pl.read_parquet("path_to_your_data.parquet")
df_test_preprocessed = rec.bake(df_test)
```
Now you're ready to train your ML model!

## Core Concepts
Below is a schematic overview of ReciPies' architecture. We 1) load a Pandas or Polars (training) dataframe, then 2) wrap it in an
Ingredients object that maintains column role information (i.e., what does this column do in this dataset). 
Next, we 3) define a Recipe consisting of multiple Steps that operate on selected columns.
Finally, we 4) prep the Recipe on the training data and 5) bake it on new data. We can then 6) run our ML pipeline on 
train and test data.
![ReciPies Flow](figures/recipies_flow.svg)

- Ingredients: Wrapper maintaining column role information
- Recipe: Collection of processing steps applied to ingredients
- Step: Individual transformation operations
- Selector: Utilities for selecting columns by roles/criteria

## Links

- [GitHub](https://github.com/rvandewater/ReciPies)
- [PyPI](https://pypi.org/project/recipies/)
