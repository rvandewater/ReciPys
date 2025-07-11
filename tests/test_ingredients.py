import pytest

from src.recipies.ingredients import Ingredients
from src.recipies.constants import Backend
import polars as pl


def test_pl_init_role(example_df):
    Ingredients(example_df, roles={"y": "outcome"})
    assert True


def test_pd_init_role(example_df):
    Ingredients(example_df, roles={"y": "outcome"})
    assert True


def test_pl_init_role_wrong_type(example_df):
    with pytest.raises(TypeError) as e_info:
        Ingredients(example_df, roles=["outcome"])
    assert e_info.match(f"Expected dict object for roles, got {[].__class__}")


def test_pd_init_role_wrong_type(example_df):
    with pytest.raises(TypeError) as e_info:
        Ingredients(example_df, roles=["outcome"], backend=Backend.PANDAS)
    assert e_info.match(f"Expected dict object for roles, got {[].__class__}")


def test_pl_init_role_typo(example_df):
    with pytest.raises(ValueError) as e_info:
        Ingredients(example_df, roles={"z": "outcome"})
    assert e_info.match(r"^Roles contains variable names that are not in the data.")


def test_pd_init_role_typo(example_df):
    with pytest.raises(ValueError) as e_info:
        Ingredients(example_df, roles={"z": "outcome"}, backend=Backend.PANDAS)
    assert e_info.match(r"^Roles contains variable names that are not in the data.")


def test_pl_init_role_copy(example_df):
    roles = {"y": ["outcome"]}
    ing = Ingredients(example_df, roles=roles)
    roles["x1"] = ["predictor"]
    assert ing.roles == {"y": ["outcome"]}


def test_pd_init_role_copy(example_df):
    roles = {"y": ["outcome"]}
    ing = Ingredients(example_df, roles=roles, backend=Backend.PANDAS)
    roles["x1"] = ["predictor"]
    assert ing.roles == {"y": ["outcome"]}


def test_pl_init_role_noncopy(example_df):
    roles = {"y": ["outcome"]}
    ing = Ingredients(example_df, copy=False, roles=roles)
    roles["x1"] = ["predictor"]
    assert ing.roles == {"x1": ["predictor"], "y": ["outcome"]}


def test_pd_init_role_noncopy(example_df):
    roles = {"y": ["outcome"]}
    ing = Ingredients(example_df, copy=False, roles=roles, backend=Backend.PANDAS)
    roles["x1"] = ["predictor"]
    assert ing.roles == {"x1": ["predictor"], "y": ["outcome"]}


def test_pl_reinit_copy(example_df):
    ing = Ingredients(example_df, roles={"y": ["outcome"]})
    reing = Ingredients(ing)
    reing.add_role("y", "predictor")
    assert ing.roles != reing.roles


def test_pd_reinit_copy(example_df):
    ing = Ingredients(example_df, roles={"y": ["outcome"]}, backend=Backend.PANDAS)
    reing = Ingredients(ing)
    reing.add_role("y", "predictor")
    assert ing.roles != reing.roles


def test_pl_reinit_noncopy(example_df):
    ing = Ingredients(example_df, roles={"y": ["outcome"]})
    reing = Ingredients(ing, copy=False)
    reing.add_role("y", "predictor")
    assert ing.roles == reing.roles


def test_pd_reinit_noncopy(example_df):
    ing = Ingredients(example_df, roles={"y": ["outcome"]}, backend=Backend.PANDAS)
    reing = Ingredients(ing, copy=False)
    reing.add_role("y", "predictor")
    assert ing.roles == reing.roles


def test_add_role(example_ingredients):
    example_ingredients.update_role("y", "first role")
    example_ingredients.add_role("y", "another role")
    assert example_ingredients.roles["y"] == ["first role", "another role"]


def test_add_role_na(example_ingredients):
    with pytest.raises(RuntimeError) as e_info:
        example_ingredients.add_role("y", "first role")
    assert e_info.match("has no roles yet")


def test_update_role_na(example_ingredients):
    example_ingredients.update_role("y", "first role")
    assert example_ingredients.roles["y"] == ["first role"]


def test_update_role_na_but_old_role(example_ingredients):
    with pytest.raises(ValueError) as e_info:
        example_ingredients.update_role("y", "first role", "imaginary role")
    assert e_info.match("does not have a role yet")


def test_update_role_implicit(example_ingredients):
    example_ingredients.update_role("y", "first role")
    example_ingredients.update_role("y", "updated role")
    assert example_ingredients.roles["y"] == ["updated role"]


def test_update_role_implicit_multiple(example_ingredients):
    example_ingredients.update_role("y", "first role")
    example_ingredients.add_role("y", "second role")
    with pytest.raises(ValueError) as e_info:
        example_ingredients.update_role("y", "updated role")
    assert e_info.match("has more than one current role")


def test_update_role_explicit(example_ingredients):
    example_ingredients.update_role("y", "first role")
    example_ingredients.update_role("y", "updated role", "first role")
    assert example_ingredients.roles["y"] == ["updated role"]


def test_update_role_explicit_multiple(example_ingredients):
    example_ingredients.update_role("y", "first role")
    example_ingredients.add_role("y", "second role")
    example_ingredients.update_role("y", "updated role", "first role")
    assert example_ingredients.roles["y"] == ["second role", "updated role"]


def test_update_role_typo(example_ingredients):
    example_ingredients.update_role("y", "first role")
    with pytest.raises(ValueError) as e_info:
        example_ingredients.update_role("y", "updated role", "firs role")
    assert e_info.match("not among current roles")


def test_inferring_backend_ingredients(example_df):
    ing = Ingredients(example_df)
    if isinstance(example_df, pl.DataFrame):
        assert ing.get_backend() == Backend.POLARS
    else:
        assert ing.get_backend() == Backend.PANDAS


def test_explicit_backend_ingredients(example_df):
    ing = Ingredients(example_df, backend=Backend.PANDAS)
    assert ing.get_backend() == Backend.PANDAS
    ing = Ingredients(example_df, backend=Backend.POLARS)
    assert ing.get_backend() == Backend.POLARS
