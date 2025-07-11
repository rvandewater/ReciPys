import polars as pl

from src.recipies.constants import Backend
from src.recipies.recipe import Recipe


def test_empty_prep_return_df(example_pl_df):
    rec = Recipe(example_pl_df)
    assert isinstance(rec.prep(), pl.DataFrame)


def test_empty_bake_return_df(example_pl_df):
    rec = Recipe(example_pl_df)
    assert isinstance(rec.bake(), pl.DataFrame)


def test_prep_bake_same_result(example_pl_df, example_pl_recipe):
    example2 = example_pl_df.clone()
    output1 = example_pl_recipe.prep(example_pl_df)
    output2 = example_pl_recipe.bake(example2)
    assert output1.equals(output2)


def test_init_roles(example_pl_df):
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2", "x3"], ["id"], ["time"])
    assert rec.data.roles["y"] == ["outcome"]
    assert rec.data.roles["x1"] == ["predictor"]
    assert rec.data.roles["x2"] == ["predictor"]
    assert rec.data.roles["x3"] == ["predictor"]
    assert rec.data.roles["time"] == ["sequence"]
    assert rec.data.roles["id"] == ["group"]


def test_inferring_backend_recipe(example_df):
    ing = Recipe(example_df)
    if isinstance(example_df, pl.DataFrame):
        assert ing.get_backend() == Backend.POLARS
    else:
        assert ing.get_backend() == Backend.PANDAS


def test_explicit_backend_recipe(example_df):
    ing = Recipe(example_df, backend=Backend.PANDAS)
    assert ing.get_backend() == Backend.PANDAS
    ing = Recipe(example_df, backend=Backend.POLARS)
    assert ing.get_backend() == Backend.POLARS


def test_backend_ingredients_recipe(example_ingredients):
    rec = Recipe(example_ingredients)
    assert rec.get_backend() == example_ingredients.get_backend()
