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
)
from src.recipies.constants import Backend


@pytest.fixture()
def example_recipe(example_ingredients):
    return Recipe(example_ingredients, ["y"], ["x1", "x2", "x3", "x4"], ["id"], ["time"])  # FIXME: add squence when merged


@pytest.fixture()
def example_recipe_w_nan(example_ingredients):
    example_ingredients[[2, 4, 6], "x2"] = np.nan
    return Recipe(example_ingredients, ["y"], ["x1", "x2", "x3", "x4"], ["id"], ["time"])  # FIXME: add squence when merged


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
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.MIN, suffix="min"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.MAX, suffix="max"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.MEAN, suffix="mean"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.MEDIAN, suffix="median"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.COUNT, suffix="count"))
        rec.add_step(StepHistorical(sel=all_of(["x1", "x2"]), fun=Accumulator.VAR, suffix="var"))
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
        return Recipe(Ingredients(example_df), ["y"], ["x1", "x2", "x3", "x4"], ["id"])  # FIXME: add squence when merged

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
