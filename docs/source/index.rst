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

.. image:: figures/recipies_logo.png
  :alt: ReciPies Logo
  :align: center

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
      pip install -e .

   Quick Start
   -----------

   Here's a simple example of using ReciPies:

   .. code-block:: python

      import polars as pl
      from recipies import Ingredients, Recipe
      from recipies.step import Step

      # Create sample data
      data = pl.DataFrame({
          "patient_id": [1, 1, 2, 2],
          "timestamp": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
          "heart_rate": [72, 75, 68, 70],
          "blood_pressure": [120, 125, 110, 115]
      })

      # Define column roles
      roles = {
          "patient_id": "patient_id",
          "timestamp": "timestamp",
          "heart_rate": "feature",
          "blood_pressure": "feature"
      }

      # Create Ingredients object
      ingredients = Ingredients(data, roles=roles)

      # Build a recipe with processing steps
      recipe = Recipe()
      recipe.add_step(Step("normalize_features"))

      # Apply the recipe
      processed_data = recipe.apply(ingredients)

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