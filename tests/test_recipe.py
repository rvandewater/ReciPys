import polars as pl
from src.recipies.constants import Backend
from src.recipies.recipe import Recipe
from src.recipies.step import StepImputeFill
from src.recipies.selector import all_predictors
from collections import Counter
from itertools import chain


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


def test_repr(example_pl_df):
    # Create a Recipe object with roles and steps
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2", "x3"], ["id"], ["time"])

    # Add a dummy step for testing
    class DummyStep:
        def __str__(self):
            return "DummyStep()"

    rec.steps.append(DummyStep())

    # Call the __repr__ method
    repr_output = repr(rec)

    # Check that the output contains the expected sections
    assert "Recipe" in repr_output
    assert "Inputs:" in repr_output
    assert "Operations:" in repr_output

    # Check that the roles and their counts are correctly displayed
    num_roles = Counter(chain.from_iterable(rec.data.roles.values()))
    for role, count in num_roles.items():
        assert f"{role}" in repr_output
        assert f"{count}" in repr_output

    # Check that the step is included in the operations section
    assert "DummyStep()" in repr_output


def test_cache_method(example_pl_df):
    # Create a Recipe object with data
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2", "x3"], ["id"], ["time"])

    # Ensure the data is initially present
    assert rec.data is not None

    # Call the cache method
    rec.cache()

    # Ensure the data is deleted after caching
    assert hasattr(rec, "data") is False


def test_roles_after_cache(example_pl_df):
    # Create a Recipe object with roles
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2", "x3"], ["id"], ["time"])

    # Ensure roles are accessible before caching
    assert hasattr(rec, "roles")
    print(rec.roles)

    # Check if the roles are in any of the lists of the roles dictionary
    assert any("group" in role_list for role_list in rec.roles.values())
    assert any("sequence" in role_list for role_list in rec.roles.values())
    assert any("predictor" in role_list for role_list in rec.roles.values())

    # Call the cache method
    rec.cache()

    # Ensure roles are still accessible after caching
    assert hasattr(rec, "roles")
    assert any("group" in role_list for role_list in rec.roles.values())
    assert any("sequence" in role_list for role_list in rec.roles.values())
    assert any("predictor" in role_list for role_list in rec.roles.values())


def test_add_step(example_pl_df):
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2", "x3"], ["id"], ["time"])

    class DummyStep:
        def __str__(self):
            return "DummyStep()"

    step = DummyStep()
    rec.add_step(step)

    # Ensure the step is added
    assert step in rec.steps


def test_update_roles(example_pl_df):
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2"], ["id"], ["time"])
    rec.update_roles("x1", new_role="feature")
    assert "feature" in rec.roles["x1"]


def test_bake(example_pl_df):
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2"], ["id"], ["time"])
    rec.add_step(StepImputeFill(sel=all_predictors(), strategy="forward"))
    baked_df = rec.bake()
    assert baked_df is not None


def test_apply_fit_transform(example_pl_df):
    rec = Recipe(example_pl_df, ["y"], ["x1", "x2"], ["id"], ["time"])
    rec.add_step(StepImputeFill(sel=all_predictors(), strategy="forward"))
    transformed_df = rec._apply_fit_transform()
    assert transformed_df is not None
