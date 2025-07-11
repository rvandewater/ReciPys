import pytest
import numpy as np
import polars as pl
from src.recipies.recipe import Recipe
from src.recipies.ingredients import Ingredients
import pandas as pd
from src.recipies.constants import Backend


@pl.api.register_dataframe_namespace("pd")
class PolarsPd:
    def __init__(self, df: pl.DataFrame):
        self.df = df.to_pandas()


@pytest.fixture
def example_pl_df():
    rand_state = np.random.RandomState(42)
    timecolumn = pl.from_pandas(pd.to_timedelta(np.concatenate((np.arange(6), np.arange(4))), unit="h"))
    # timecolumn = pl.concat([pl.datetime_range(datetime(MINYEAR, 1, 1,0), datetime(MINYEAR, 1, 1,5), "1h", eager=True),
    #           pl.datetime_range(datetime(MINYEAR, 1, 1,0), datetime(MINYEAR, 1, 1,3), "1h", eager=True)])

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
    return df


@pytest.fixture
def example_pd_df():
    rand_state = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "id": [1] * 6 + [2] * 4,
            "time": pd.to_timedelta(np.concatenate((np.arange(6), np.arange(4))), unit="h"),
            "y": rand_state.normal(size=(10,)),
            "x1": rand_state.normal(loc=10, scale=5, size=(10,)),
            "x2": rand_state.binomial(n=1, p=0.3, size=(10,)),
            "x3": pd.Series(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"], dtype="category"),
            "x4": pd.Series(["x", "y", "y", "x", "y", "y", "x", "x", "y", "x"], dtype="category"),
        }
    )
    return df


@pytest.fixture(params=["example_pl_df", "example_pd_df"])
def example_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def example_pl_ingredients(example_pl_df):
    return Ingredients(example_pl_df)


@pytest.fixture
def example_pd_ingredients(example_pd_df):
    return Ingredients(example_pd_df, backend=Backend.PANDAS)


@pytest.fixture(params=["example_pl_ingredients", "example_pd_ingredients"])
def example_ingredients(request):
    return request.getfixturevalue(request.param)


# @pytest.fixture
# def example_ingredients(example_pl_df):
#     return Ingredients(example_pl_df, backend=Backend.PANDAS)


@pytest.fixture()
def example_pl_recipe(example_pl_df):
    return Recipe(example_pl_df, ["y"], ["x1", "x2", "x3", "x4"], ["id"], ["time"])


@pytest.fixture()
def example_pd_recipe(example_pd_df):
    return Recipe(example_pd_df, ["y"], ["x1", "x2", "x3", "x4"], ["id"], ["time"])


@pytest.fixture()
def example_recipe_w_nan(example_df):
    example_df.loc[[1, 2, 4, 7], "x1"] = np.nan
    return Recipe(example_df, ["y"], ["x1", "x2", "x3", "x4"], ["id"], ["time"])
