import polars as pl
from recipys.recipe import Recipe


def test_empty_prep_return_df(example_pl_df):
    rec = Recipe(example_pl_df)
    assert type(rec.prep()) == pl.DataFrame


def test_empty_bake_return_df(example_pl_df):
    rec = Recipe(example_pl_df)
    assert type(rec.bake()) == pl.DataFrame

def test_prep_bake_same_result(example_pl_df, example_recipe):
    example2 = example_pl_df.clone()
    output1 = example_recipe.prep(example_pl_df)
    output2 = example_recipe.bake(example2)
    assert output1.equals(output2)

def test_init_roles(example_pl_df):
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2", "x3"], ["id"], ["time"])
    assert rec.data.roles["y"] == ["outcome"]
    assert rec.data.roles["x1"] == ["predictor"]
    assert rec.data.roles["x2"] == ["predictor"]
    assert rec.data.roles["x3"] == ["predictor"]
    assert rec.data.roles["time"] == ["sequence"]
    assert rec.data.roles["id"] == ["group"]
