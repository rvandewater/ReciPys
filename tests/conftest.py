import pytest
import numpy as np
import polars as pl
from recipys.recipe import Recipe
from recipys.ingredients import Ingredients
from datetime import datetime, timedelta
@pl.api.register_dataframe_namespace("pd")
class PolarsPd:
    def __init__(self, df: pl.DataFrame):
        self.df = df.to_pandas()

@pytest.fixture()
def example_df():
    rand_state = np.random.RandomState(42)
    interval = timedelta(hours=1)

    df = pl.DataFrame(
        {
            "id": [1] * 6 + [2] * 4,
            "time": pl.Series(np.concatenate([(np.arange(0,6) * interval), (np.arange(0,4) * interval)])),
            # "time": timedelta(hours=range(10)),
            "y": rand_state.normal(size=(10,)),
            "x1": rand_state.normal(loc=10, scale=5, size=(10,)),
            "x2": rand_state.binomial(n=1, p=0.3, size=(10,)),
            "x3": pl.Series(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"]),
            "x4": pl.Series(["x", "y", "y", "x", "y", "y", "x", "x", "y", "x"]), #, dtype="category"
        }
    )
    pandas_df = df.to_pandas()
    print(df)
    return df


@pytest.fixture
def example_ingredients(example_df):
    return Ingredients(example_df)


@pytest.fixture()
def example_recipe(example_df):
    return Recipe(example_df, ["y"], ["x1", "x2", "x3", "x4"], ["id"])  # FIXME: add squence when merged


@pytest.fixture()
def example_recipe_w_nan(example_df):
    example_df.loc[[1, 2, 4, 7], "x1"] = np.nan
    return Recipe(example_df, ["y"], ["x1", "x2", "x3", "x4"], ["id"])  # FIXME: add squence when merged
