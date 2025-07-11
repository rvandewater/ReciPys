![logo](https://github.com/rvandewater/ReciPies/blob/development/docs/figures/recipies_logo.png?raw=true)

# 🥧ReciPies🐍

[![CI](https://github.com/rvandewater/ReciPies/actions/workflows/ci.yml/badge.svg)](https://github.com/rvandewater/ReciPies/actions/workflows/ci.yml)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Platform](https://img.shields.io/badge/platform-linux--64%20|%20win--64%20|%20osx--64-lightgrey)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI version shields.io](https://img.shields.io/pypi/v/recipies.svg)](https://pypi.python.org/pypi/recipies/)
[![arXiv](https://img.shields.io/badge/arXiv-2306.05109-b31b1b.svg)](http://arxiv.org/abs/2306.05109)

The ReciPies package is a preprocessing framework operating on [Polars](https://github.com/pola-rs/polars)
and [Pandas](https://github.com/pandas-dev/pandas) dataframes. The backend can be chosen by the user.
The operation of this package is inspired by the R-package [recipes](https://recipes.tidymodels.org/).
This package allows the user to apply a number of extensible operations for imputation, feature generation/extraction,
scaling, and encoding.
It operates on modified Dataframe objects from the established data science package Pandas.

## Installation

You can install ReciPies from pip using:

```
pip install recipies
```

> Note that the package is called `recipies`  on pip.
>
You can install ReciPies from source to ensure you have the latest version:

```
conda env update -f environment.yml
conda activate ReciPies
pip install -e .
```

> Note that the last command installs the package called `recipies`.

## Usage

To define preprocessing operations, one has to supply _roles_ to the different columns of the Dataframe.
This allows the user to create groups of columns which have a particular function.
Then, we provide several "steps" that can be applied to the datasets, among which: Historical accumulation,
Resampling the time resolution, A number of imputation methods, and a wrapper for any
[Scikit-learn](https://github.com/scikit-learn/scikit-learn) preprocessing step.
We believe to have covered any basic preprocessing needs for prepared datasets.
Any missing step can be added by following the step interface.

# 📄Paper

If you use this code in your research, please cite the following publication (a standalone paper is in preparation):

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

This paper can also be found on arxiv: https://arxiv.org/pdf/2306.05109.pdf




