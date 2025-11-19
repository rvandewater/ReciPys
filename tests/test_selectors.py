import pytest
import polars as pl
from src.recipies.constants import Backend
import re
from src.recipies.selector import (
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
    enlist_dt,
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
    if example_ingredients.get_backend() == Backend.POLARS:
        sel = has_type("Float64")
    else:
        sel = has_type("float64")
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
    sel = all_numeric_predictors(backend=example_ingredients.get_backend())
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


def test_selector_len(example_ingredients):
    selector = Selector("Test Selector", names=["x1", "x2"])
    selector(example_ingredients)  # Call the selector with Ingredients
    assert len(selector) == 2


def test_selector_set_names():
    selector = Selector("Test Selector")
    selector.set_names(["col1", "col2"])
    assert selector.names == ["col1", "col2"]


def test_selector_set_roles():
    selector = Selector("Test Selector")
    selector.set_roles(["role1", "role2"])
    assert selector.roles == ["role1", "role2"]


def test_selector_set_types():
    selector = Selector("Test Selector")
    selector.set_types(["type1", "type2"])
    assert selector.types == ["type1", "type2"]


def test_selector_set_pattern():
    selector = Selector("Test Selector")
    pattern = re.compile("col.*")
    selector.set_pattern(pattern)
    assert selector.pattern == pattern


def test_selector_call(example_ingredients):
    selector = Selector("Test Selector", names=["x1", "x2"])
    selected = selector(example_ingredients)
    assert selected == ["x1", "x2"]


def test_selector_call_with_roles(example_ingredients):
    selector = Selector("Test Selector", roles=["predictor"])
    example_ingredients.update_role(["x1", "x2", "x3", "x4"], "predictor")
    selected = selector(example_ingredients)
    assert selected == ["x1", "x2", "x3", "x4"]  # Assuming these have the "predictor" role


def test_selector_call_with_types(example_ingredients):
    selector = Selector(
        "Test Selector", types=["float64"] if example_ingredients.get_backend() == Backend.PANDAS else ["Float64"]
    )
    selected = selector(example_ingredients)
    assert selected == ["y", "x1"]


def test_selector_call_with_names(example_ingredients):
    selector = Selector("Test Selector", names=["x1", "x3"])
    selected = selector(example_ingredients)
    assert selected == ["x1", "x3"]


def test_selector_call_with_pattern(example_ingredients):
    selector = Selector("Test Selector")
    selector.set_pattern(re.compile(r"^x[1-3]$"))
    selected = selector(example_ingredients)
    assert selected == ["x1", "x2", "x3"]


def test_selector_repr():
    selector = Selector("Test Selector")
    assert repr(selector) == "Test Selector"


def test_enlist_dt():
    # Test wrapping a single DataType
    dt = pl.Float64  # Updated to use pl.Float64
    assert enlist_dt(dt) == [dt]

    # Test passing a list of DataTypes
    dt_list = [pl.Float64, pl.Int64]  # Updated to use pl.Float64 and pl.Int64
    assert enlist_dt(dt_list) == dt_list

    # Test invalid input
    with pytest.raises(TypeError):
        enlist_dt("invalid")


def test_selector_iter(example_ingredients):
    # Test __iter__ method
    sel = all_of(["x1", "x2"])
    selected = sel(example_ingredients)
    assert list(sel) == selected  # Ensure iteration works
    assert list(sel) == ["x1", "x2"]


def test_selector_getitem(example_ingredients):
    # Test __getitem__ method
    sel = all_of(["x1", "x2"])
    sel(example_ingredients)  # Call the selector with Ingredients
    assert sel[0] == "x1"  # Ensure indexing works
    assert sel[1] == "x2"
    with pytest.raises(IndexError):
        _ = sel[2]  # Ensure out-of-range indexing raises an error


def test_selector_iter_error():
    # Test __iter__ error when _last_selection is not set
    sel = Selector(description="Test Selector")
    with pytest.raises(AttributeError, match="Selector must be called with Ingredients before"):
        list(sel)


def test_selector_len_error():
    # Test __len__ error when _last_selection is not set
    sel = Selector(description="Test Selector")
    with pytest.raises(AttributeError, match="Selector must be called with Ingredients before getting length."):
        len(sel)


def test_selector_getitem_error():
    # Test __getitem__ error when _last_selection is not set
    sel = Selector(description="Test Selector")
    with pytest.raises(AttributeError, match="Selector must be called with Ingredients before indexing."):
        sel[0]


def test_selector_call_type_error():
    # Test __call__ error when input is not an Ingredients object
    sel = Selector(description="Test Selector")
    with pytest.raises(TypeError, match="Expected Ingredients, got <class 'str'>"):
        sel("invalid_input")


def test_enlist_str_type_error():
    # Test enlist_str with invalid input
    with pytest.raises(TypeError, match="Expected str or list of str, got <class 'int'>"):
        from src.recipies.selector import enlist_str

        enlist_str(123)


def test_enlist_dt_type_error():
    # Test enlist_dt with invalid input
    with pytest.raises(TypeError, match="Expected a pl datatype, got <class 'int'>"):
        from src.recipies.selector import enlist_dt

        enlist_dt(123)


def test_intersection_type_error():
    # Test intersection with invalid input
    with pytest.raises(TypeError, match="'int' object is not iterable"):
        from src.recipies.selector import intersection

        intersection(123, ["x1", "x2"])


def test_selector_roles_invalid_role(example_ingredients):
    # Test has_role with a role that doesn't exist
    sel = has_role(["nonexistent_role"])
    selected = sel(example_ingredients)
    assert selected == []


def test_selector_types_invalid_type(example_ingredients):
    # Test has_type with a type that doesn't exist
    sel = has_type(["nonexistent_type"])
    selected = sel(example_ingredients)
    assert selected == []


def test_selector_pattern_no_match(example_ingredients):
    # Test regex_names with a pattern that doesn't match any column
    sel = regex_names(r"z.*")
    selected = sel(example_ingredients)
    assert selected == []
