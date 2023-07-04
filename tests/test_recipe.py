import polars as pl
from recipys.recipe import Recipe


def test_empty_prep_return_df(example_pl_df):
    rec = Recipe(example_pl_df)
    # assert type(rec.prep()) == pl.DataFrame
    assert type(rec.prep()) == pl.DataFrame


def test_empty_bake_return_df(example_pl_df):
    rec = Recipe(example_pl_df)
    assert type(rec.bake()) == pl.DataFrame


def test_init_roles(example_pl_df):
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2", "x3"], ["id"])  # FIXME: add squence when merged
    assert rec.data.roles["y"] == ["outcome"]
    assert rec.data.roles["x1"] == ["predictor"]
    assert rec.data.roles["x2"] == ["predictor"]
    assert rec.data.roles["x3"] == ["predictor"]
    assert rec.data.roles["id"] == ["group"]
