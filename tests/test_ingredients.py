import pytest
import pandas as pd
import polars as pl

from src.recipies.ingredients import Ingredients
from src.recipies.constants import Backend


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
        example_ingredients.update_role("y", "updated role", "first ever role")
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


def test_ingredients_backend_inference():
    # Test backend inference for Polars DataFrame
    pl_df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    ingr = Ingredients(pl_df)
    assert ingr.backend == Backend.POLARS

    # Test backend inference for Pandas DataFrame
    pd_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    ingr = Ingredients(pd_df)
    assert ingr.backend == Backend.PANDAS

    # Test backend inference for Ingredients object
    ingr2 = Ingredients(ingr)
    assert ingr2.backend == Backend.PANDAS

    # Test invalid backend inference
    with pytest.raises(ValueError):
        Ingredients("invalid_data")


def test_ingredients_roles_copy():
    # Test roles copying
    pd_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    roles = {"col1": ["role1"], "col2": ["role2"]}  # Ensure roles match column names
    ingr = Ingredients(pd_df, roles=roles, check_roles=False)  # Disable role checking for this test
    ingr_copy = Ingredients(ingr)
    assert ingr_copy.roles == roles
    assert ingr_copy.data.equals(ingr.data)


def test_ingredients_init_invalid_roles(example_pl_df):
    # Test roles with invalid column names
    with pytest.raises(ValueError, match="Roles contains variable names that are not in the data"):
        Ingredients(example_pl_df, roles={"invalid_column": ["role"]})


def test_ingredients_init_backend_inference(example_pl_df):
    # Test backend inference for polars
    ingredients = Ingredients(example_pl_df)
    assert ingredients.get_backend() == Backend.POLARS

    # Test backend inference for pandas
    pandas_df = example_pl_df.to_pandas()
    ingredients = Ingredients(pandas_df)
    assert ingredients.get_backend() == Backend.PANDAS


def test_ingredients_add_role(example_pl_df):
    # Test adding a role to a column with existing roles
    ingredients = Ingredients(example_pl_df, roles={"x1": ["predictor"]})
    ingredients.add_role("x1", "new_role")
    assert "new_role" in ingredients.roles["x1"]

    # Test adding a role to a column without roles (should raise an error)
    with pytest.raises(RuntimeError, match="has no roles yet, use update_role instead"):
        ingredients.add_role("x2", "new_role")


def test_ingredients_update_role(example_pl_df):
    # Test updating a role for a column
    ingredients = Ingredients(example_pl_df, roles={"x1": ["predictor"]})
    ingredients.update_role("x1", "new_predictor", "predictor")
    assert "new_predictor" in ingredients.roles["x1"]
    assert "predictor" not in ingredients.roles["x1"]

    # Test updating a role without specifying old_role
    ingredients.update_role("x1", "another_role")
    assert ingredients.roles["x1"] == ["another_role"]

    # Test invalid old_role
    with pytest.raises(
        ValueError, match="Attempted to set role of x1 from invalid_role to new_role but invalid_role not among current roles"
    ):
        ingredients.update_role("x1", "new_role", "invalid_role")


def test_ingredients_select_dtypes(example_pl_df):
    # Test selecting columns by data types
    ingredients = Ingredients(example_pl_df)
    selected = ingredients.select_dtypes(include=["float64"] if ingredients.get_backend() == Backend.PANDAS else ["Float64"])
    assert "x1" in selected or "y" in selected


def test_ingredients_get_dtypes(example_pl_df):
    # Test retrieving data types
    ingredients = Ingredients(example_pl_df)
    dtypes = ingredients.get_dtypes()
    assert len(dtypes) == len(example_pl_df.columns)


def test_ingredients_get_str_dtypes(example_pl_df):
    # Test retrieving data types as strings
    ingredients = Ingredients(example_pl_df)
    str_dtypes = ingredients.get_str_dtypes()
    assert all(isinstance(dtype, str) for dtype in str_dtypes.values())


def test_ingredients_groupby(example_pl_df):
    # Test grouping by columns for polars backend
    ingredients = Ingredients(example_pl_df)
    grouped = ingredients.groupby("x1")
    assert grouped is not None

    # Test grouping by columns for pandas backend
    pandas_df = example_pl_df.to_pandas()
    ingredients = Ingredients(pandas_df)
    grouped = ingredients.groupby("x1")
    assert grouped is not None


def test_ingredients_getitem(example_pl_df):
    # Test item access for polars backend
    ingredients = Ingredients(example_pl_df)
    column = ingredients["x1"]
    assert isinstance(column, pl.Series)

    # Test item access for pandas backend
    pandas_df = example_pl_df.to_pandas()
    ingredients = Ingredients(pandas_df)
    column = ingredients["x1"]
    assert isinstance(column, pd.Series)


def test_ingredients_setitem(example_pl_df):
    # Test item assignment for polars backend
    ingredients = Ingredients(example_pl_df)
    ingredients.set_df(ingredients.data.with_columns(pl.Series("new_col", list(range(len(ingredients.data))))))
    assert "new_col" in ingredients.columns

    # Test item assignment for pandas backend
    pandas_df = example_pl_df.to_pandas()
    ingredients = Ingredients(pandas_df)
    ingredients["new_col"] = list(range(len(ingredients.data)))
    assert "new_col" in ingredients.columns


def test_select_dtypes_pandas(example_pd_ingredients):
    # Test select_dtypes for Pandas backend
    selected = example_pd_ingredients.select_dtypes(include=["int64"])
    assert selected == ["id", "x2"]

    selected = example_pd_ingredients.select_dtypes(include=["float64"])
    assert selected == ["y", "x1"]

    selected = example_pd_ingredients.select_dtypes(include=["bool"])
    assert selected == []

    selected = example_pd_ingredients.select_dtypes(include=["object"])
    assert selected == []


def test_select_dtypes_polars(example_pl_ingredients):
    # Test select_dtypes for Polars backend
    selected = example_pl_ingredients.select_dtypes(include=["Int64"])
    assert selected == ["id", "x2"]

    selected = example_pl_ingredients.select_dtypes(include=["Float64"])
    assert selected == ["y", "x1"]

    selected = example_pl_ingredients.select_dtypes(include=["Utf8"])
    assert selected == []

    selected = example_pl_ingredients.select_dtypes(include=["Boolean"])
    assert selected == []


def test_select_dtypes_empty_include(example_pd_ingredients):
    # Test with an empty include list
    selected = example_pd_ingredients.select_dtypes(include=[])
    assert selected == []


def test_select_dtypes_no_match(example_pd_ingredients):
    # Test with a data type that doesn't exist in the DataFrame
    selected = example_pd_ingredients.select_dtypes(include=["nonexistent_dtype"])
    assert selected == []


def test_select_dtypes_single_column(example_pd_ingredients):
    # Test with a single column matching the data type
    selected = example_pd_ingredients.select_dtypes(include=["float64"])
    assert selected == ["y", "x1"]


def test_select_dtypes_multiple_matches(example_pd_ingredients):
    # Test with multiple columns matching the data type
    selected = example_pd_ingredients.select_dtypes(include=["int64"])
    assert selected == ["id", "x2"]


def test_select_dtypes_polars_empty_include(example_pl_ingredients):
    # Test with an empty include list for Polars
    selected = example_pl_ingredients.select_dtypes(include=[])
    assert selected == []


def test_select_dtypes_polars_no_match(example_pl_ingredients):
    # Test with a data type that doesn't exist in the Polars DataFrame
    selected = example_pl_ingredients.select_dtypes(include=["nonexistent_dtype"])
    assert selected == []


def test_select_dtypes_polars_single_column(example_pl_ingredients):
    # Test with a single column matching the data type for Polars
    selected = example_pl_ingredients.select_dtypes(include=["Float64"])
    assert selected == ["y", "x1"]


def test_select_dtypes_polars_multiple_matches(example_pl_ingredients):
    # Test with multiple columns matching the data type for Polars
    selected = example_pl_ingredients.select_dtypes(include=["Int64"])
    assert selected == ["id", "x2"]


def test_to_df_pandas(example_pd_ingredients):
    df = example_pd_ingredients.to_df(output_format=Backend.PANDAS)
    assert isinstance(df, pd.DataFrame)


def test_to_df_polars(example_pl_ingredients):
    df = example_pl_ingredients.to_df(output_format=Backend.POLARS)
    assert isinstance(df, pl.DataFrame)


def test_check_column_invalid(example_pd_ingredients):
    with pytest.raises(ValueError, match="Expected string"):
        example_pd_ingredients._check_column(123)

    with pytest.raises(ValueError, match="does not exist in this Data object"):
        example_pd_ingredients._check_column("nonexistent_column")
