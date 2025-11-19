from datetime import datetime, MINYEAR

import pandas as pd
import pytest
import polars as pl
import numpy as np
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    SplineTransformer,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator

from src.recipies.ingredients import Ingredients
from src.recipies.recipe import Recipe
from src.recipies.selector import all_numeric_predictors, has_type, has_role, all_of
from src.recipies.step import (
    StepSklearn,
    StepHistorical,
    Accumulator,
    StepImputeFill,
    StepScale,
    StepResampling,
    StepImputeFastZeroFill,
    StepImputeFastForwardFill,
    StepFunction,
    Step,
)
from src.recipies.constants import Backend


@pytest.fixture()
def example_recipe(example_ingredients):
    return Recipe(example_ingredients, ["y"], ["x1", "x2", "x3", "x4"], ["id"], ["time"])


@pytest.fixture()
def example_recipe_w_nan(example_ingredients):
    example_ingredients[[2, 4, 6], "x2"] = np.nan
    return Recipe(example_ingredients, ["y"], ["x1", "x2", "x3", "x4"], ["id"], ["time"])


def test_no_group_for_group_step(example_ingredients):
    rec = Recipe(example_ingredients, ["y"], ["x1", "x2"])
    rec.add_step(StepImputeFill(value=0))
    rec.prep()


class TestStepResampling:
    def test_step_grouped(self, example_df):
        # Using group role
        pre_sampling_len = example_df.shape[0]
        if isinstance(example_df, pl.DataFrame):
            backend = Backend.POLARS
            timecolumn = pl.concat(
                [
                    pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 5), "1h", eager=True),
                    pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 3), "1h", eager=True),
                ]
            )
            example_df = example_df.with_columns(time=timecolumn)
        else:
            backend = Backend.PANDAS
        rec = Recipe(example_df, ["y"], ["x1", "x2"], ["id"], ["time"], backend=backend)
        resampling_dict = {all_numeric_predictors(backend): Accumulator.MEAN}
        rec.add_step(StepResampling("2h", accumulator_dict=resampling_dict))
        df = rec.bake()
        assert df.shape[0] == pre_sampling_len / 2

    def test_step_wo_selectors(self, example_df):
        # Using group role and without supplying any selectors
        pre_sampling_len = example_df.shape[0]
        if isinstance(example_df, pl.DataFrame):
            backend = Backend.POLARS
            timecolumn = pl.concat(
                [
                    pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 5), "1h", eager=True),
                    pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 3), "1h", eager=True),
                ]
            )
            example_df = example_df.with_columns(time=timecolumn)
        else:
            backend = Backend.PANDAS
        rec = Recipe(example_df, ["y"], ["x1", "x2"], ["id"], ["time"], backend=backend)
        rec.add_step(StepResampling("2h"))
        df = rec.bake()
        assert df.shape[0] == pre_sampling_len / 2

    # Todo: check if desired behaviour to have no group role
    def test_step_ungrouped(self, example_df):
        # Without using group role
        if isinstance(example_df, pl.DataFrame):
            timecolumn = pl.concat(
                [
                    pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 5), "1h", eager=True),
                    pl.datetime_range(datetime(MINYEAR, 1, 1, 0), datetime(MINYEAR, 1, 1, 3), "1h", eager=True),
                ]
            )
            example_df = example_df.with_columns(time=timecolumn)
            example_df = example_df.drop("id")
            example_df = example_df.unique(subset="time")
        else:
            # Pandas
            example_df.drop("id", axis=1, inplace=True)
            example_df = example_df.drop_duplicates(subset="time", inplace=False, keep="first")
        pre_sampling_len = example_df.shape[0]
        rec = Recipe(example_df, ["y"], ["x1", "x2"])
        rec.update_roles("time", "sequence")
        resampling_dict = {all_numeric_predictors(): Accumulator.MEAN}
        rec.add_step(StepResampling("2h", accumulator_dict=resampling_dict))
        df = rec.bake()
        assert df.shape[0] == (pre_sampling_len / 2)


class TestStepHistorical:
    def test_step(self, example_df):
        rec = Recipe(Ingredients(example_df), ["y"], ["x1", "x2"], ["id"])
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.MIN, suffix="_min"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.MAX, suffix="_max"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.MEAN, suffix="_mean"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.MEDIAN, suffix="_median"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.COUNT, suffix="_count"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.VAR, suffix="_var"))
        df = rec.bake()
        if rec.get_backend() == Backend.POLARS:
            assert df["x1_min"][-1] == df.filter(pl.col("id") == 2).select(pl.col("x1")).min().item()
            assert df["x1_max"][-1] == df.filter(pl.col("id") == 2).select(pl.col("x1")).max().item()
            # assert df["x1_mean"][-1] == df.filter(pl.col("id") == 2).select(pl.col("x1")).mean().item()
            # With approximate equality
            assert df["x1_mean"][-1] == pytest.approx(df.filter(pl.col("id") == 2).select(pl.col("x1")).mean().item())
            assert df["x1_median"][-1] == df.filter(pl.col("id") == 2).select(pl.col("x1")).median().item()
            assert df["x1_count"][-1] == df.filter(pl.col("id") == 2).select(pl.col("x1")).count().item()
            # somehow we get a rounding difference between these two values
            assert (
                df["x1_var"].round(2)[-1]
                == df.filter(pl.col("id") == 2).select(pl.col("x1")).var().to_series().round(2).item()
            )
        else:
            assert df["x1_min"].iloc[-1] == df["x1"].loc[df["id"] == 2].min()
            assert df["x2_max"].iloc[-1] == df["x2"].loc[df["id"] == 2].max()
            assert df["x2_mean"].iloc[-1] == df["x2"].loc[df["id"] == 2].mean()
            assert df["x1_median"].iloc[-1] == df["x1"].loc[df["id"] == 2].median()
            assert df["x1_count"].iloc[-1] == df["x1"].loc[df["id"] == 2].count()
            assert df["x2_var"].iloc[-1] == df["x2"].loc[df["id"] == 2].var()


class TestImputeSteps:
    def test_impute_fill(self, example_recipe_w_nan):
        example_recipe_w_nan.add_step(StepImputeFill(strategy="forward"))
        backend = example_recipe_w_nan.get_backend()
        res = example_recipe_w_nan.prep()
        nan_list = [0, 1, 1, 0, 0, 0, np.nan, 0, 0, 1]
        exp = (
            pl.Series("x2", nan_list, dtype=pl.Int32, strict=False)
            if backend == Backend.POLARS
            else pd.Series(nan_list, dtype="float64")
        )
        assert res["x2"].equals(exp)
        example_recipe_w_nan.add_step(StepImputeFill(sel=all_numeric_predictors(backend), value=0))
        res = example_recipe_w_nan.prep()
        imputed_list = [0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
        exp = (
            pl.Series("x2", imputed_list, pl.Int32, strict=False)
            if backend == Backend.POLARS
            else pd.Series(imputed_list, dtype="float64")
        )
        assert res["x2"].equals(exp)

    def test_fast_zero_fill(self, example_recipe_w_nan):
        backend = example_recipe_w_nan.get_backend()
        example_recipe_w_nan.add_step(StepImputeFastZeroFill(sel=all_numeric_predictors(backend)))
        if backend == Backend.POLARS:
            with pytest.raises(ValueError) as e_info:
                res = example_recipe_w_nan.prep()
            assert e_info.match("Backend.POLARS not supported by this step.")
        else:
            res = example_recipe_w_nan.prep()
            imputed_list = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
            exp = pd.Series(imputed_list, dtype="float64")
            assert res["x2"].equals(exp)

    def test_fast_forward_fill(self, example_recipe_w_nan):
        backend = example_recipe_w_nan.get_backend()
        example_recipe_w_nan.add_step(StepImputeFastForwardFill(sel=all_numeric_predictors(backend)))
        if backend == Backend.POLARS:
            with pytest.raises(ValueError) as e_info:
                res = example_recipe_w_nan.prep()
            assert e_info.match("Backend.POLARS not supported by this step.")
        else:
            res = example_recipe_w_nan.prep()
            imputed_list = [0, 1, 1, 0, 0, 0, np.nan, 0, 0, 1]
            exp = pd.Series(imputed_list, dtype="float64")
            assert res["x2"].equals(exp)


class TestScaleStep:
    def test_scale_step_default(self, example_recipe_w_nan):
        example_recipe_w_nan.add_step(StepScale(all_numeric_predictors(backend=example_recipe_w_nan.get_backend())))
        res = example_recipe_w_nan.prep()
        assert abs(res["x1"].mean()) < 0.00001
        assert abs(res["x2"].mean()) < 0.00001

    def test_scale_step_w_args(self, example_recipe):
        example_recipe.add_step(
            StepScale(all_numeric_predictors(backend=example_recipe.get_backend()), with_mean=False, with_std=False)
        )
        res = example_recipe.prep()
        assert abs(res["x1"].mean()) > 1
        assert abs(res["x1"].var()) > 1.5

    def test_scale_step_w_sel(self, example_recipe):
        example_recipe.add_step(StepScale(sel=all_of(["x2"])))
        res = example_recipe.prep()
        assert abs(res["x2"].mean()) < 0.00001
        assert abs(res["x1"].mean()) > 1


class TestSklearnStep:
    @pytest.fixture()
    def example_recipe_w_categorical_label(self, example_df):
        if isinstance(example_df, pl.DataFrame):
            example_df = example_df.with_columns(
                y=pl.Series(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"], dtype=pl.Categorical)
            )
        else:
            example_df["y"] = pd.Categorical(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"])
        return Recipe(Ingredients(example_df), ["y"], ["x1", "x2", "x3", "x4"], ["id"], ["time"])

    def test_simple_imputer(self, example_recipe_w_nan):
        backend = example_recipe_w_nan.get_backend()
        example_recipe_w_nan.add_step(StepSklearn(SimpleImputer(strategy="constant", fill_value=0)))
        df = example_recipe_w_nan.prep()
        assert (
            (df[[2, 4, 6], "x2"].to_numpy() == np.full(3, 0)).all()
            if backend == backend.POLARS
            else (df.loc[[2, 4, 6], "x2"] == 0).all()
        )

    def test_knn_imputer(self, example_recipe_w_nan):
        backend = example_recipe_w_nan.get_backend()
        example_recipe_w_nan.add_step(StepSklearn(KNNImputer(), sel=all_numeric_predictors(backend)))
        df = example_recipe_w_nan.prep()
        assert (
            (~np.isnan(df[[2, 4, 6], "x2"].to_numpy())).all()
            if backend == backend.POLARS
            else (~np.isnan(df.loc[[2, 4, 6], "x2"])).all()
        )

    def test_iterative_imputer(self, example_recipe_w_nan):
        backend = example_recipe_w_nan.get_backend()
        example_recipe_w_nan.add_step(StepSklearn(IterativeImputer(), sel=all_numeric_predictors(backend)))
        df = example_recipe_w_nan.prep()
        assert (
            (~np.isnan(df[[2, 4, 6], "x2"].to_numpy())).all()
            if backend == backend.POLARS
            else (~np.isnan(df.loc[[2, 4, 6], "x2"])).all()
        )

    def test_missing_indicator(self, example_recipe_w_nan):
        backend = example_recipe_w_nan.get_backend()
        example_recipe_w_nan.add_step(
            StepSklearn(MissingIndicator(features="all"), sel=all_numeric_predictors(backend), in_place=False)
        )
        df = example_recipe_w_nan.prep()
        assert (
            (df[[2, 4, 6], "MissingIndicator_x2"].to_numpy()).all()
            if backend == backend.POLARS
            else (df.loc[[2, 4, 6], "MissingIndicator_x2"]).all()
        )

    def test_standard_scaler(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(StepSklearn(StandardScaler(), sel=all_numeric_predictors(backend)))
        df = example_recipe.prep()
        assert abs(df["x1"].mean()) < 0.00001
        assert abs(df["x2"].mean()) < 0.00001

    def test_min_max_scaler(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(StepSklearn(MinMaxScaler(), sel=all_numeric_predictors(backend)))
        df = example_recipe.prep()
        assert ((0 <= df["x1"]) & (df["x1"] <= 1)).all()
        assert ((0 <= df["x2"]) & (df["x2"] <= 1)).all()

    def test_max_abs_scaler(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(StepSklearn(MaxAbsScaler(), sel=all_numeric_predictors(backend)))
        df = example_recipe.prep()
        assert ((-1 <= df["x1"]) & (df["x1"] <= 1)).all()
        assert ((-1 <= df["x2"]) & (df["x2"] <= 1)).all()

    def test_robust_scaler(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(StepSklearn(RobustScaler(), sel=all_numeric_predictors(backend)))
        df = example_recipe.prep()
        assert df["x1"].median() < 10e-12
        assert df["x2"].median() < 10e-12

    def test_binarizer(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(StepSklearn(Binarizer(), sel=all_numeric_predictors(backend=backend)))
        df = example_recipe.prep()
        assert (df["x1"].is_in([0, 1])).all() if backend == Backend.POLARS else (df["x1"].isin([0, 1])).all()
        assert (df["x2"].is_in([0, 1])).all() if backend == Backend.POLARS else (df["x2"].isin([0, 1])).all()

    def test_normalizer(self, example_recipe):
        example_recipe.add_step(StepSklearn(Normalizer(), sel=all_numeric_predictors(backend=example_recipe.get_backend())))
        df = example_recipe.prep()
        assert ((0 <= df["x1"]) & (df["x1"] <= 1)).all()
        assert ((0 <= df["x2"]) & (df["x2"] <= 1)).all()

    def test_k_bins_binarizer(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(
            StepSklearn(
                KBinsDiscretizer(n_bins=2, strategy="uniform", encode="ordinal"),
                sel=all_numeric_predictors(backend=example_recipe.get_backend()),
                in_place=False,
            )
        )
        df = example_recipe.prep()
        assert (
            (df["KBinsDiscretizer_x1"].is_in([0, 1])).all()
            if backend == Backend.POLARS
            else (df["KBinsDiscretizer_x1"].isin([0, 1])).all()
        )
        assert (
            (df["KBinsDiscretizer_x2"].is_in([0, 1])).all()
            if backend == Backend.POLARS
            else (df["KBinsDiscretizer_x2"].isin([0, 1])).all()
        )

    def test_quantile_transformer(self, example_recipe):
        example_recipe.add_step(
            StepSklearn(QuantileTransformer(n_quantiles=10), sel=all_numeric_predictors(example_recipe.get_backend()))
        )
        df = example_recipe.prep()
        assert ((0 <= df["x1"]) & (df["x1"] <= 1)).all()
        assert ((0 <= df["x2"]) & (df["x2"] <= 1)).all()

    def test_ordinal_encoder(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(
            StepSklearn(
                OrdinalEncoder(),
                sel=has_type([str(pl.Categorical(ordering="physical")) if backend == backend.POLARS else "category"]),
                in_place=False,
            )
        )
        df = example_recipe.prep()
        # FIXME assert correct number of new columns
        assert ((0 <= df["OrdinalEncoder_x3"]) & (df["OrdinalEncoder_x4"] <= 2)).all()

    def test_onehot_encoder(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(
            StepSklearn(
                OneHotEncoder(sparse_output=False),
                sel=has_type([str(pl.Categorical(ordering="physical")) if backend == backend.POLARS else "category"]),
                in_place=False,
            )
        )
        df = example_recipe.prep()
        if backend == backend.POLARS:
            assert (df["OneHotEncoder_1"].is_in([0, 1])).all()
            assert (df["OneHotEncoder_2"].is_in([0, 1])).all()
            assert (df["OneHotEncoder_3"].is_in([0, 1])).all()
            assert (df["OneHotEncoder_4"].is_in([0, 1])).all()
            assert (df["OneHotEncoder_5"].is_in([0, 1])).all()
        else:
            assert (df["OneHotEncoder_1"].isin([0, 1])).all()
            assert (df["OneHotEncoder_2"].isin([0, 1])).all()
            assert (df["OneHotEncoder_3"].isin([0, 1])).all()
            assert (df["OneHotEncoder_4"].isin([0, 1])).all()
            assert (df["OneHotEncoder_5"].isin([0, 1])).all()

    def test_label_encoder(self, example_recipe_w_categorical_label):
        example_recipe_w_categorical_label.add_step(StepSklearn(LabelEncoder(), sel=has_role(["outcome"]), columnwise=True))
        df = example_recipe_w_categorical_label.prep()
        assert ((0 <= df["y"]) & (df["y"] <= 2)).all()

    def test_label_binarizer(self, example_recipe_w_categorical_label):
        example_recipe_w_categorical_label.add_step(
            StepSklearn(LabelBinarizer(), sel=has_role(["outcome"]), columnwise=True, in_place=False, role="outcome")
        )
        df = example_recipe_w_categorical_label.prep()
        if example_recipe_w_categorical_label.get_backend() == Backend.POLARS:
            assert (df["LabelBinarizer_y_1"].is_in([0, 1])).all()
            assert (df["LabelBinarizer_y_2"].is_in([0, 1])).all()
            assert (df["LabelBinarizer_y_3"].is_in([0, 1])).all()
        else:
            assert (df["LabelBinarizer_y_1"].isin([0, 1])).all()
            assert (df["LabelBinarizer_y_2"].isin([0, 1])).all()
            assert (df["LabelBinarizer_y_3"].isin([0, 1])).all()

    def test_spline_transformer(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(StepSklearn(SplineTransformer(), sel=all_numeric_predictors(backend), in_place=False))
        df = example_recipe.prep()
        # FIXME assert correct number of new columns
        assert not df["SplineTransformer_1"].is_empty() if backend == Backend.POLARS else not df["SplineTransformer_1"].empty

    def test_polynomial_features(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(StepSklearn(PolynomialFeatures(), sel=all_numeric_predictors(backend), in_place=False))
        df = example_recipe.prep()
        # FIXME assert correct number of new columns
        assert not df["PolynomialFeatures_1"].is_empty() if backend == Backend.POLARS else not df["PolynomialFeatures_1"].empty

    def test_power_transformer(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(StepSklearn(PowerTransformer(), sel=all_numeric_predictors(backend), in_place=False))
        df = example_recipe.prep()
        # FIXME assert correct number of new columns
        assert not df["PowerTransformer_x1"].is_empty() if backend == Backend.POLARS else not df["PowerTransformer_x1"].empty

    def test_function_transformer(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(
            StepSklearn(
                FunctionTransformer(np.log1p), sel=all_numeric_predictors(example_recipe.get_backend()), in_place=False
            )
        )
        df = example_recipe.prep()
        # FIXME assert correct number of new columns
        assert (
            not df["FunctionTransformer_x1"].is_empty()
            if backend == Backend.POLARS
            else not df["FunctionTransformer_x1"].empty
        )

    def test_wrong_columnwise(self, example_df):
        if isinstance(example_df, pl.DataFrame):
            example_df = example_df.with_columns(
                y=pl.Series(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"], dtype=pl.Categorical)
            )
            example_df = example_df.with_columns(
                y1=pl.Series(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"], dtype=pl.Categorical)
            )
        else:
            example_df["y"] = pd.Categorical(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"])
            example_df["y1"] = pd.Categorical(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"])
        rec = Recipe(Ingredients(example_df), ["y", "y1"], ["x1", "x2", "x3"], ["id"], ["time"])
        rec.add_step(StepSklearn(LabelEncoder(), sel=has_role(["outcome"]), columnwise=False))
        with pytest.raises(ValueError) as exc_info:
            rec.prep()
        assert "columnwise=True" in str(exc_info.value)

    def test_wrong_in_place(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(
            StepSklearn(
                OneHotEncoder(sparse_output=False),
                sel=has_type([str(pl.Categorical(ordering="physical")) if backend == backend.POLARS else "category"]),
                in_place=True,
            )
        )
        with pytest.raises(ValueError) as exc_info:
            example_recipe.prep()
        assert "in_place=False" in str(exc_info.value)

    def test_sparse_output_error(self, example_recipe):
        backend = example_recipe.get_backend()
        example_recipe.add_step(
            StepSklearn(
                OneHotEncoder(sparse_output=True),
                sel=has_type([str(pl.Categorical(ordering="physical")) if backend == backend.POLARS else "category"]),
                in_place=False,
            )
        )
        with pytest.raises(TypeError) as exc_info:
            example_recipe.prep()
        assert "sparse_output=False" in str(exc_info.value)


def test_step_trained_property():
    step = Step()
    assert not step.trained  # Default should be False


def test_step_group_property():
    step = Step()
    assert step.group  # Default should be True


def test_step_fit(example_ingredients):
    step = Step()
    step.fit(example_ingredients)
    assert step.trained  # Ensure the step is marked as trained after fitting


def test_step_unsupported_backend(example_ingredients):
    step = Step(supported_backends=[Backend.PANDAS])
    example_ingredients.backend = Backend.POLARS
    with pytest.raises(ValueError):
        step.fit(example_ingredients)  # Should raise an error for unsupported backend


def test_step_impute_fill_invalid_strategy(example_ingredients):
    # Test invalid strategy in StepImputeFill
    rec = Recipe(example_ingredients, ["y"], ["x1", "x2"])
    step = StepImputeFill(strategy="invalid_strategy")
    rec.add_step(step)
    with pytest.raises(ValueError, match="No valid strategy provided. Strategy was: invalid_strategy"):
        rec.prep()


def test_step_scale_in_place_false(example_ingredients):
    # Test StepScale with in_place=False
    rec = Recipe(example_ingredients, ["y"], ["x1", "x2"])
    step = StepScale(in_place=False)
    rec.add_step(step)
    prepped = rec.prep()
    assert "x1" in prepped.columns and "StandardScaler_x1" in prepped.columns
    assert "x2" in prepped.columns and "StandardScaler_x2" in prepped.columns


def test_step_function(example_ingredients):
    rec = Recipe(example_ingredients, ["y"], ["x1", "x2"])
    if isinstance(example_ingredients.get_df(), pd.DataFrame):
        original_df = example_ingredients.get_df().copy()
    elif isinstance(example_ingredients.get_df(), pl.DataFrame):
        original_df = example_ingredients.get_df().clone()
    else:
        raise TypeError("Unsupported DataFrame type")

    # Define a transformation function that increments numeric columns by 1
    def add_one(data, columns):
        df = data.get_df()
        if isinstance(df, pd.DataFrame):
            df[columns] = df[columns] + 1
        elif isinstance(df, pl.DataFrame):
            df = df.with_columns([(df[col] + 1).alias(col) for col in columns])
        else:
            raise TypeError("Unsupported DataFrame type")
        data.set_df(df)
        return data

    # Create the StepFunction instance
    step = StepFunction(function=add_one, sel=all_numeric_predictors(example_ingredients.get_backend()))

    # Add the step to the recipe and prepare the data
    rec.add_step(step)
    prepped = rec.prep()
    # Verify the transformation

    if isinstance(original_df, pd.DataFrame):
        # For Pandas: Increment numeric columns in the expected DataFrame
        expected_df = original_df.copy()
        expected_df[["x1", "x2"]] += 1
        print("Expected DataFrame:")
        print(expected_df)
        print("Prepped DataFrame:")
        print(prepped)
        pd.testing.assert_frame_equal(
            prepped[["x1", "x2"]], expected_df[["x1", "x2"]], check_exact=False, rtol=1e-5, atol=1e-8
        )
    elif isinstance(original_df, pl.DataFrame):
        # For Polars: Increment numeric columns in the expected DataFrame
        expected_df = original_df.with_columns([(original_df[col] + 1).alias(col) for col in ["x1", "x2"]])
        assert prepped.equals(expected_df)


def test_step_repr(example_ingredients):
    # Create a dummy step
    class DummyStep(Step):
        def __init__(self, sel, desc="Dummy Step"):
            super().__init__(sel=sel)
            self.desc = desc

        def do_fit(self, data):
            pass

        def transform(self, data):
            return data

    # Instantiate the step
    step = DummyStep(sel=all_numeric_predictors(), desc="Test Step")

    # Test __repr__ before training
    repr_before_training = repr(step)
    assert "Test Step for" in repr_before_training
    assert "all numeric predictors" in repr_before_training
    assert "[trained]" not in repr_before_training

    # Fit the step
    step.fit(example_ingredients)

    # Test __repr__ after training
    repr_after_training = repr(step)
    assert "Test Step for" in repr_after_training
    assert "[trained]" in repr_after_training
    if len(step.columns) < 3:
        assert str(step.columns) in repr_after_training
    else:
        assert str(step.columns[:2] + ["..."]) in repr_after_training


# Create a dummy step
class DummyStep(Step):
    def do_fit(self, data):
        pass

    def transform(self, data):
        return data


def test_check_ingredients(example_ingredients):
    # Instantiate the step
    step = DummyStep()

    # Test with valid input
    validated_data = step._check_ingredients(example_ingredients)
    assert isinstance(validated_data, Ingredients)

    # Test with unsupported backend
    step.supported_backends = [Backend.PANDAS]
    example_ingredients.backend = Backend.POLARS
    with pytest.raises(ValueError, match="Backend.POLARS not supported by this step."):
        step._check_ingredients(example_ingredients)


def test_check_ingredients_grouping(example_ingredients):
    # Test with grouped data when grouping is not allowed
    step = DummyStep()
    step._group = False
    if example_ingredients.get_backend() == Backend.PANDAS:
        grouped_data = example_ingredients.get_df().groupby("id")
        with pytest.raises(ValueError, match="Step does not accept grouped data."):
            step._check_ingredients(grouped_data)
    elif example_ingredients.get_backend() == Backend.POLARS:
        grouped_data = example_ingredients.get_df().group_by("id")
        with pytest.raises(ValueError, match="Step does not accept grouped data."):
            step._check_ingredients(grouped_data)

    # Test with invalid input type
    with pytest.raises(ValueError, match="Expected Ingredients object, got <class 'str'>"):
        step._check_ingredients("invalid_input")
