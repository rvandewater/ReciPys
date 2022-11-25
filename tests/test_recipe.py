import pandas as pd
from recipys.recipe import Recipe


def test_empty_prep_return_df(example_df):
    rec = Recipe(example_df)
    assert type(rec.prep()) == pd.DataFrame


def test_empty_bake_return_df(example_df):
    rec = Recipe(example_df)
    assert type(rec.bake()) == pd.DataFrame


def test_init_roles(example_df):
    rec = Recipe(example_df, ["y"], ["x1", "x2", "x3"], ["id"])  # FIXME: add squence when merged
    assert rec.data.roles["y"] == ["outcome"]
    assert rec.data.roles["x1"] == ["predictor"]
    assert rec.data.roles["x2"] == ["predictor"]
    assert rec.data.roles["x3"] == ["predictor"]
    assert rec.data.roles["id"] == ["group"]
