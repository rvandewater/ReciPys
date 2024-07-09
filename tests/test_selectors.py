import pytest
import polars as pl

from recipys.selector import (
    Selector,
    all_outcomes,
    all_of,
    regex_names,
    select_groups,
    select_sequence,
    starts_with,
    ends_with,
    contains,
    has_role,
    has_type,
    all_numeric_predictors,
    all_predictors,
    intersection,
    enlist_str,
)


def test_no_description():
    with pytest.raises(TypeError) as e_info:
        Selector()
    assert e_info.match("missing 1 required positional argument")


def test_not_ingredients(example_pl_df):
    with pytest.raises(TypeError) as e_info:
        Selector("test step")(example_pl_df)
    assert e_info.match("Expected Ingredients")


def test_intersection():
    assert intersection(["a", "b"], ["b", "c"]) == ["b"]


def test_enlist_str():
    assert enlist_str("string") == ["string"]


def test_enlist_str_list():
    assert enlist_str(["string1", "string2"]) == ["string1", "string2"]


def test_enlist_str_None():
    assert enlist_str(None) is None


def test_enlist_str_other():
    with pytest.raises(TypeError) as e_info:
        enlist_str({"k": "string"})
    assert e_info.match("Expected str or list of str")


def test_enlist_str_other_list():
    with pytest.raises(TypeError) as e_info:
        enlist_str(["outer", {"k": "inner"}])
    assert e_info.match("Only lists of str are allowed.")


def test_all_of(example_ingredients):
    sel = all_of(["y", "x1"])
    assert sel(example_ingredients) == ["y", "x1"]


def test_regex_names(example_ingredients):
    sel = regex_names(r"^x\d")
    assert sel(example_ingredients) == ["x1", "x2", "x3", "x4"]


def test_starts_with(example_ingredients):
    sel = starts_with("x")
    assert sel(example_ingredients) == ["x1", "x2", "x3", "x4"]


def test_ends_with(example_ingredients):
    sel = ends_with("1")
    assert sel(example_ingredients) == ["x1"]


def test_contains(example_ingredients):
    sel = contains("i")
    assert sel(example_ingredients) == ["id", "time"]


def test_has_role(example_ingredients):
    example_ingredients.update_role("x1", "predictor")
    example_ingredients.update_role("x2", "predictor")
    sel = has_role("predictor")
    assert sel(example_ingredients) == ["x1", "x2"]


def test_has_type(example_ingredients):
    # sel = has_type("Float64")
    # sel = has_type(pl.Float64)
    sel = has_type("Float64")
    assert sel(example_ingredients) == ["y", "x1"]

# def test_has_type_pl(example_ingredients):
#     sel = has_type(pl.Float64)
#     assert sel(example_ingredients) == ["y", "x1"]
def test_all_predictors(example_ingredients):
    example_ingredients.update_role("x1", "predictor")
    example_ingredients.update_role("x2", "predictor")
    sel = all_predictors()
    assert sel(example_ingredients) == ["x1", "x2"]


def test_all_numeric_predictors(example_ingredients):
    example_ingredients.update_role("x1", "predictor")
    example_ingredients.update_role("x2", "predictor")
    sel = all_numeric_predictors()
    assert sel(example_ingredients) == ["x1", "x2"]


def test_all_outcomes(example_ingredients):
    example_ingredients.update_role("y", "outcome")
    sel = all_outcomes()
    assert sel(example_ingredients) == ["y"]


def test_select_groups(example_ingredients):
    example_ingredients.update_role("id", "group")
    assert select_groups(example_ingredients) == ["id"]


def test_select_sequence(example_ingredients):
    example_ingredients.update_role("time", "sequence")
    assert select_sequence(example_ingredients) == ["time"]
