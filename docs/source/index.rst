.. image:: /../figures/recipies_logo.svg
  :alt: ReciPies Logo
  :align: center

ReciPies
=======

.. image:: https://img.shields.io/pypi/v/recipies
  :target: https://pypi.org/project/recipies/
  :alt: PyPI Version

.. image:: https://codecov.io/gh/rvandewater/ReciPies/graph/badge.svg?token=YOUR_CODECOV_TOKEN
  :target: https://codecov.io/gh/rvandewater/ReciPies
  :alt: Code Coverage

.. image:: https://github.com/rvandewater/ReciPies/actions/workflows/tests.yaml/badge.svg
  :target: https://github.com/rvandewater/ReciPies/actions/workflows/tests.yaml
  :alt: Tests

.. image:: https://img.shields.io/pypi/pyversions/recipies
  :target: https://pypi.org/project/recipies/
  :alt: Python Versions

.. image:: https://img.shields.io/github/license/rvandewater/ReciPies
  :target: https://github.com/rvandewater/ReciPies/blob/main/LICENSE
  :alt: License



ReciPies is a Python package for feature engineering and data preprocessing with a focus on medical and clinical data.
It provides a unified interface for working with both Polars and Pandas DataFrames while maintaining column role
information throughout data transformations.

Features
--------

- **Dual Backend Support**: Seamlessly work with both Polars and Pandas DataFrames
- **Column Role Management**: Track and maintain semantic roles of columns (e.g., patient_id, timestamp, features)
- **Medical Data Focus**: Specialized tools for clinical and medical data preprocessing
- **Pipeline Architecture**: Build reproducible data processing pipelines with Steps and Recipes
- **Type Safety**: Strong typing support for better code reliability
- **Performance**: Leverage the speed of Polars while maintaining Pandas compatibility

Installation
------------

Install ReciPies using pip:

.. code-block:: bash

  pip install recipies

For development installation:

.. code-block:: bash

  git clone https://github.com/rvandewater/ReciPies.git
  cd ReciPies
  pip install -e '.[dev]'

Quick Start
-----------

Here's a simple example of using ReciPies:

   .. code-block:: python

      import polars as pl
      import numpy as np
      from datetime import datetime, MINYEAR
      import recipies as rp
      from recipies import Ingredients, Recipe

      # Set up random state for reproducible results
      rand_state = np.random.RandomState(42)

      # Create time columns for two different groups
      timecolumn = pl.concat([
          pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 5), "1h", eager=True),
          pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 3), "1h", eager=True)
      ])

      # Create sample DataFrame
      df = pl.DataFrame({
          "id": [1] * 6 + [2] * 4,
          "time": timecolumn,
          "y": rand_state.normal(size=(10,)),
          "x1": rand_state.normal(loc=10, scale=5, size=(10,)),
          "x2": rand_state.binomial(n=1, p=0.3, size=(10,)),
          "x3": pl.Series(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"], dtype=pl.Categorical),
          "x4": pl.Series(["x", "y", "y", "x", "y", "y", "x", "x", "y", "x"], dtype=pl.Categorical),
      })

      # Introduce some missing values
      df = df.with_columns(
          pl.when(pl.int_range(pl.len()).is_in([1, 2, 4, 7]))
          .then(None)
          .otherwise(pl.col("x1"))
          .alias("x1")
      )

      # Create Ingredients and Recipe
      ing = Ingredients(df)
      rec = Recipe(
          ing,
          outcomes=["y"],
          predictors=["x1", "x2", "x3", "x4"],
          groups=["id"],
          sequences=["time"]
      )
Core Concepts
-------------

**Ingredients**
  A wrapper around DataFrames that maintains column role information, ensuring data semantics are preserved during transformations.

**Recipe**
  A collection of processing steps that can be applied to Ingredients objects to create reproducible data pipelines.

**Step**
  Individual data transformation operations that understand column roles and can work with both Polars and Pandas backends.

**Selector**
  Utilities for selecting columns based on their roles or other criteria.

Backend Support
---------------

ReciPies supports both Polars and Pandas backends:

- **Polars**: High-performance DataFrame library with lazy evaluation
- **Pandas**: Traditional DataFrame library with extensive ecosystem support

The package automatically detects the backend and provides a consistent API regardless of the underlying DataFrame implementation.

Examples
--------

Check out the `examples/` directory for Jupyter notebooks demonstrating:

- Basic usage and concepts
- Medical data preprocessing workflows
- Performance benchmarking between backends
- Advanced pipeline construction

Contributing
------------

Contributions are welcome! Please see our contributing guidelines and open an issue or submit a pull request on the `GitHub repository <https://github.com/rvandewater/ReciPies>`_.

License
-------

This project is licensed under the MIT License. See the `LICENSE <https://github.com/rvandewater/ReciPies/blob/main/LICENSE>`_ file for details.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`