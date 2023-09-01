from abc import abstractmethod
from copy import deepcopy
from typing import Union, Dict

import pandas as pd
import numpy as np
from scipy.sparse import isspmatrix
from pandas.core.groupby import DataFrameGroupBy
from sklearn.preprocessing import StandardScaler
from recipys.ingredients import Ingredients
from enum import Enum
from recipys.selector import (
    Selector,
    all_predictors,
    all_numeric_predictors,
    select_groups,
    select_sequence,
)
from pandas.api.types import is_timedelta64_dtype, is_datetime64_any_dtype


class Step:
    """This class represents a step in a recipe.

    Steps are transformations to be executed on selected columns of a DataFrame.
    They fit a transformer to the selected columns and afterwards transform the data with the fitted transformer.

    Args:
        sel: Object that holds information about the selected columns.

    Attributes:
        columns: List with the names of the selected columns.
    """

    def __init__(self, sel: Selector = all_predictors()):
        self.sel = sel
        self.columns = []
        self._trained = False
        self._group = True

    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def group(self) -> bool:
        return self._group

    def fit(self, data: Ingredients):
        """This function fits the transformer to the data.

        Args:
            data: The DataFrame to fit to.
        """
        data = self._check_ingredients(data)
        self.columns = self.sel(data)
        self.do_fit(data)
        self._trained = True

    @abstractmethod
    def do_fit(self, data: Ingredients):
        pass

    def _check_ingredients(self, data: Union[Ingredients, DataFrameGroupBy]) -> Ingredients:
        """Check input for allowed types

        Args:
            data: input to the step

        Raises:
            ValueError: If a grouped pd.DataFrame is provided to a step that can't use groups.
            ValueError: If input are not (potentially grouped) Ingredients.

        Returns:
            Validated input
        """
        if isinstance(data, DataFrameGroupBy):
            if not self._group:
                raise ValueError("Step does not accept grouped data.")
            data = data.obj
        if not isinstance(data, Ingredients):
            raise ValueError(f"Expected Ingredients object, got {data.__class__}")
        return data

    def transform(self, data: Ingredients) -> Ingredients:
        """This function transforms the data with the fitted transformer.

        Args:
            data: The DataFrame to transform.

        Returns:
            The transformed DataFrame.
        """
        pass

    def fit_transform(self, data: Ingredients) -> Ingredients:
        self.fit(data)
        return self.transform(data)

    def __repr__(self) -> str:
        repr = self.desc + " for "

        if not self.trained:
            repr += str(self.sel)
        else:
            repr += str(self.columns) if len(self.columns) < 3 else str(self.columns[:2] + ["..."])  # FIXME: remove brackets
            repr += " [trained]"

        return repr


class StepImputeFill(Step):
    """Uses pandas' internal `nafill` function to replace missing values.
    See `pandas.DataFrame.nafill` for a description of the arguments.
    """

    def __init__(self, sel=all_predictors(), value=None, method=None, limit=None):
        super().__init__(sel)
        self.desc = f"Impute with {method if method else value}"
        self.value = value
        self.method = method
        self.limit = limit

    def transform(self, data):
        new_data = self._check_ingredients(data)
        new_data[self.columns] = data[self.columns].fillna(self.value, method=self.method, axis=0, limit=self.limit)
        return new_data


class StepImputeFastZeroFill(Step):
    """Quick variant of pandas' internal `nafill(value=0)` for grouped dataframes."""

    def __init__(self, sel=all_predictors()):
        super().__init__(sel)
        self.desc = "Impute quickly with 0"

    def transform(self, data):
        new_data = self._check_ingredients(data)

        # Ignore grouping as grouping does not matter for zero fill.
        new_data[self.columns] = new_data[self.columns].fillna(0)

        return new_data


class StepImputeFastForwardFill(Step):
    """Quick variant of pandas' internal `nafill(method='ffill')` for grouped dataframes.

    Note: this variant does not allow for setting a limit.
    """

    def __init__(self, sel=all_predictors()):
        super().__init__(sel)
        self.desc = "Impute with fast ffill"

    def transform(self, data):
        new_data = self._check_ingredients(data)

        # Use cumsum (which is optimised for grouped frames) to figure out which
        # values should be left at NaN, then ffill on the ungrouped dataframe. Adopted from:
        # https://stackoverflow.com/questions/36871783/fillna-forward-fill-on-a-large-dataframe-efficiently-with-groupby
        nofill = new_data.copy()
        nofill[self.columns] = pd.notnull(nofill[self.columns])
        nofill = nofill.groupby(data.keys).cumsum()

        new_data[self.columns] = new_data[self.columns].ffill()
        for col in self.columns:
            new_data.loc[nofill[col].to_numpy() == 0, col] = np.nan

        return new_data


class StepImputeModel(Step):
    """Uses a pretrained imputation model to impute missing values.
    Args:
        model: A function that takes a dataframe and the grouping columns as input and
            returns a dataframe with imputed values without the grouping column.
    """

    def __init__(self, sel=all_predictors(), model=None):
        super().__init__(sel)
        self.desc = "Impute with pretrained imputation model"
        self.model = model

    def transform(self, data):
        new_data = self._check_ingredients(data)
        new_data[self.columns] = self.model(new_data[self.columns + select_groups(new_data)], select_groups(new_data))
        return new_data


class Accumulator(Enum):
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"
    VAR = "var"
    FIRST = "first"
    LAST = "last"


class StepHistorical(Step):
    """This step generates columns with a historical accumulator provided by the user.

    Args:
        fun: Instance of the Accumulator enumerable that signifies which type of historical accumulation
            to use (default is MAX).
        suffix: Defaults to none. Set the name to have the step generate new columns with this suffix
            instead of the default suffix.
        role: Defaults to 'predictor'. In case new columns are added, set their role to role.
    """

    def __init__(
            self,
            sel: Selector = all_numeric_predictors(),
            fun: Accumulator = Accumulator.MAX,
            suffix: str = None,
            role: str = "predictor",
    ):
        super().__init__(sel)

        self.desc = f"Create historical {fun}"
        self.fun = fun
        if suffix is None:
            try:
                suffix = fun.value
            except Exception:
                raise TypeError(f"Expected Accumulator enum for function, got {self.fun.__class__}")
        self.suffix = suffix
        self.role = role

    def transform(self, data: Ingredients) -> Ingredients:
        """
        Raises:
            TypeError: If the function is not of type Accumulator
        """
        new_data = self._check_ingredients(data)

        new_columns = [c + "_" + self.suffix for c in self.columns]

        if self.fun is Accumulator.MAX:
            res = data[self.columns].cummax(skipna=True)
        elif self.fun is Accumulator.MIN:
            res = data[self.columns].cummin(skipna=True)
        elif self.fun is Accumulator.MEAN:
            # Reset index, as we get back a multi-index, and we want a simple rolling index
            res = data[self.columns].expanding().mean().reset_index(drop=True)
        elif self.fun is Accumulator.MEDIAN:
            res = data[self.columns].expanding().median().reset_index(drop=True)
        elif self.fun is Accumulator.COUNT:
            res = data[self.columns].expanding().count().reset_index(drop=True)
        elif self.fun is Accumulator.VAR:
            res = data[self.columns].expanding().var().reset_index(drop=True)
        else:
            raise TypeError(f"Expected Accumulator enum for function, got {self.fun.__class__}")
        new_data[new_columns] = res

        # Update roles for the newly generated columns
        for nc in new_columns:
            new_data.update_role(nc, self.role)

        return new_data


class StepSklearn(Step):
    """This step takes a transformer from scikit-learn and makes it usable as a step in a recipe.

    Args:
        sklearn_transformer: Instance of scikit-learn transformer that implements fit() and transform().
        columnwise: Defaults to False. Set to True to fit and transform the DF column by column.
        in_place: Defaults to True. Set to False to have the step generate new columns
            instead of overwriting the existing ones.
        role (str, optional): Defaults to 'predictor'. Incase new columns are added, set their role to role.
    """

    def __init__(
            self,
            sklearn_transformer: object,
            sel: Selector = all_predictors(),
            columnwise: bool = False,
            in_place: bool = True,
            role: str = "predictor",
    ):
        super().__init__(sel)
        self.desc = f"Use sklearn transformer {sklearn_transformer.__class__.__name__}"
        self.sklearn_transformer = sklearn_transformer
        self.columnwise = columnwise
        self.in_place = in_place
        self.role = role
        self._group = False

    def do_fit(self, data: Ingredients) -> Ingredients:
        """
        Raises:
            ValueError: If the transformer expects a single column but gets multiple.
        """
        if self.columnwise:
            self._transformers = {
                # copy the transformer so we keep the distinct fit for each column and don't just refit
                col: deepcopy(self.sklearn_transformer.fit(data[col]))
                for col in self.columns
            }
        else:
            try:
                self.sklearn_transformer.fit(data[self.columns])
            except ValueError as e:
                if "should be a 1d array" in str(e) or "Multioutput target data is not supported" in str(e):
                    raise ValueError(
                        "The sklearn transformer expects a 1d array as input. " "Try running the step with columnwise=True."
                    )
                raise

    def transform(self, data: Ingredients) -> Ingredients:
        """
        Raises:
            TypeError: If the transformer returns a sparse matrix.
            ValueError: If the transformer returns an unexpected amount of columns.
        """
        new_data = self._check_ingredients(data)

        if self.columnwise:
            for col in self.columns:
                new_cols = self._transformers[col].transform(new_data[col])
                if self.in_place and new_cols.ndim == 2 and new_cols.shape[1] > 1:
                    raise ValueError(
                        "The sklearn transformer returned more than one column. Try running the step with in_place=False."
                    )
                col_names = (
                    col
                    if self.in_place
                    else [f"{self.sklearn_transformer.__class__.__name__}_{col}_{i + 1}" for i in range(new_cols.shape[1])]
                )
                new_data[col_names] = new_cols
        else:
            new_cols = self.sklearn_transformer.transform(new_data[self.columns])
            if isspmatrix(new_cols):
                raise TypeError(
                    "The sklearn transformer returns a sparse matrix, "
                    "but recipes expects a dense numpy representation. "
                    "Try setting sparse=False or similar in the transformer initilisation."
                )

            col_names = (
                self.columns
                if self.in_place
                else [f"{self.sklearn_transformer.__class__.__name__}_{i + 1}" for i in range(new_cols.shape[1])]
            )
            if new_cols.shape[1] != len(col_names):
                raise ValueError(
                    "The sklearn transformer returned a different amount of columns. Try running the step with in_place=False."
                )

            new_data[col_names] = new_cols

        # set role of new columns
        if not self.in_place:
            for col in col_names:
                new_data.update_role(col, self.role)

        return new_data


class StepResampling(Step):
    def __init__(
            self,
            new_resolution: str = "1h",
            accumulator_dict: Dict[Selector, Accumulator] = {all_predictors(): Accumulator.LAST},
            default_accumulator: Accumulator = Accumulator.LAST,
    ):
        """This class represents a step in a recipe.

        Steps are transformations to be executed on selected columns of a DataFrame.
        They fit a transformer to the selected columns and afterwards transform the data with the fitted transformer.

        Args:
            new_resolution: Resolution to resample to.
            accumulator_dict: Supply dictionary with individual accumulation methods for each Selector.
            default_accumulator: Accumulator to use for variables not supplied in dictionary.
        """
        super().__init__()
        self.new_resolution = new_resolution
        self.acc_dict = accumulator_dict
        self.default_accumulator = default_accumulator
        self._group = True

    def do_fit(self, data: Ingredients):
        self._trained = True

    def transform(self, data):
        new_data = self._check_ingredients(data)

        # Check for and save first sequence role
        if select_sequence(new_data) is not None:
            sequence_role = select_sequence(new_data)[0]
        else:
            raise AssertionError("Sequence role has not been assigned, resampling step not possible")
        sequence_datatype = new_data.dtypes[sequence_role]

        if not (is_timedelta64_dtype(sequence_datatype) or is_datetime64_any_dtype(sequence_datatype)):
            raise ValueError(f"Expected Timedelta or Timestamp object, got {sequence_role(data).__class__}")

        # Dictionary with the format column: str , accumulator:str is created
        col_acc_map = {}
        # Go through supplied Selector, Accumulator pairs
        for selector, accumulator in self.acc_dict.items():
            selected_columns = selector(new_data)
            # Add variables associated with selector with supplied accumulator
            col_acc_map.update({col: accumulator.value for col in selected_columns})

        # Add non-specified variables, if not a sequence role
        col_acc_map.update(
            {
                col: self.default_accumulator.value
                for col in new_data.columns.difference(col_acc_map.keys())
                if col not in select_sequence(new_data)
            }
        )

        # Resampling with the functions defined in col_acc_map
        new_data = data.resample(self.new_resolution, on=sequence_role).agg(col_acc_map)

        # Remove multi-index in case of grouped data
        if isinstance(data, DataFrameGroupBy):
            new_data = new_data.droplevel(select_groups(data.obj))

        # Remove sequence index, while keeping column
        new_data = new_data.reset_index(drop=False)

        return new_data


class StepScale:
    """Provides a wrapper for a scaling with StepSklearn.

    Args:
       with_mean: Defaults to True. If True, center the data before scaling.
       with_std: Defaults to True. If True, scale the data to unit variance (or equivalently, unit standard deviation).
       in_place: Defaults to True. Set to False to have the step generate new columns instead of overwriting the existing ones.
       role (str, optional): Defaults to 'predictor'. Incase new columns are added, set their role to role.
    """

    def __new__(
            cls,
            sel: Selector = all_numeric_predictors(),
            with_mean: bool = True,
            with_std: bool = True,
            in_place: bool = True,
            role: str = "predictor",
    ):
        return StepSklearn(StandardScaler(with_mean=with_mean, with_std=with_std), sel=sel, in_place=in_place, role=role)


class StepFunction(Step):
    """Provides a wrapper for a simple transformation function, without fitting."""

    def __init__(self, sel: Selector, function):
        super().__init__(sel=sel)
        self.function = function
        self._trained = True

    def transform(self, data: Ingredients) -> Ingredients:
        new_data = self._check_ingredients(data)
        new_data = self.function(new_data)
        return new_data
