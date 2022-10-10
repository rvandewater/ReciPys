from abc import abstractmethod
from copy import deepcopy
from typing import Union

import pandas as pd
from scipy.sparse import isspmatrix
from pandas.core.groupby import DataFrameGroupBy
from sklearn.preprocessing import StandardScaler
from icu_benchmarks.recipes.ingredients import Ingredients
from enum import Enum
from icu_benchmarks.recipes.selector import Selector, all_predictors, all_numeric_predictors, sequence, groups


class Step:
    """This class represents a step in a recipe.

    Steps are transformations to be executed on selected columns of a DataFrame.
    They fit a transformer to the selected columns and afterwards transform the data with the fitted transformer.

    Args:
        sel (Selector): Object that holds information about the selected columns.

    Attributes:
        columns (list): List with the names of the selected columns.
        _trained (bool): If the step was fitted already.
        _group (bool): If the step runs on grouped data.
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
            data (Ingredients): The DataFrame to fit to.
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
            data (Union[Ingredients, DataFrameGroupBy]): input to the step

        Raises:
            ValueError: If a grouped pd.DataFrame is provided to a step that can't use groups.
            ValueError: If input are not (potentially grouped) Ingredients.

        Returns:
            Ingredients: validated input
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
            data (Ingredients): The DataFrame to transform.

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


class StepScale(Step):
    def __init__(self, sel=all_numeric_predictors(), with_mean=True, with_std=True):
        super().__init__(sel)
        self.desc = f"Scale with mean ({with_mean}) and std ({with_std})"
        self.with_mean = with_mean
        self.with_std = with_std
        self._group = False

    def do_fit(self, data):
        self.scalers = {
            c: StandardScaler(copy=True, with_mean=self.with_mean, with_std=self.with_std).fit(data[c].values[:, None])
            for c in self.columns
        }

    def transform(self, data):
        new_data = self._check_ingredients(data)
        for c, sclr in self.scalers.items():
            new_data[c] = sclr.transform(data[c].values[:, None])
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
        fun (Accumulator): Instance of the Accumulator enumerable that signifies which type of historical accumulation
            to use (default is MAX).
        suffix (String, optional): Defaults to none. Set the name to have the step generate new columns with this suffix
            instead of the default suffix.
        role (str, optional): Defaults to 'predictor'. In case new columns are added, set their role to role.
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
        sklearn_transformer (object): Instance of scikit-learn transformer that implements fit() and transform().
        columnwise (bool, optional): Defaults to False. Set to True to fit and transform the DF column by column.
        in_place (bool, optional): Defaults to True.
            Set to False to have the step generate new columns instead of overwriting the existing ones.
        role (str, optional): Defaults to 'predictor'. Incase new columns are added, set their role to role.

    Attributes:
        _transformers (dict): If the transformer is applied columnwise,
            this dict holds references to the separately fitted instances.
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
            self._transformers = {}
            for col in self.columns:
                # copy the transformer so we keep the distinct fit for each column and don't just refit
                self._transformers[col] = deepcopy(self.sklearn_transformer.fit(data[col]))
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
        sel: Selector = all_numeric_predictors(),
        sel_acc: Accumulator = Accumulator.MEAN,
        not_sel_acc: Accumulator = Accumulator.LAST,
        seq_role: Selector = sequence(),
        acc_meth_dict: {Selector, Accumulator} = None,
    ):
        """This class represents a step in a recipe.

        Steps are transformations to be executed on selected columns of a DataFrame.
        They fit a transformer to the selected columns and afterwards transform the data with the fitted transformer.

        Args:
            new_resolution(str): Resolution to resample to.
            sel (Selector): Object that holds information about the selected columns.
            sel_acc: (Accumulator) Accumulation method for selected columns
            not_sel_acc: (Accumulator) Accumulation method for all non-selected columns
            seq_role (Selector) : Temporal role in supplied Ingredients.
            acc_meth_dict ({Selector, Accumulator}, Optional) : Alternatively supply dictionary with individual accumulator methods
        """
        super().__init__(sel=sel)
        self.new_resolution = new_resolution
        self.sel_acc = sel_acc
        self.not_sel_acc = not_sel_acc
        self.sequence_role = seq_role
        self.acc_dict = acc_meth_dict
        self._group = False

    def fit(self, data):
        self._trained = True

    def transform(self, data):
        new_data = self._check_ingredients(data)

        # Check for sequence role
        sequence_role= self.sequence_role(new_data)
        sequence_datatype = new_data.dtypes[sequence_role[0]]

        if not (pd.api.types.is_timedelta64_dtype(sequence_datatype) or pd.api.types.is_datetime64_any_dtype(sequence_datatype)):
            raise ValueError(f"Expected Timedelta or Timestamp object, got {self.sequence_role(data).__class__}")

        # Prepare accumulation method dictionary
        if self.acc_dict is None:
            # Dictionary with individual accumulators is not provided
            sel_columns = self.sel(new_data)
            # Get other columns
            not_sel_columns = new_data.columns.difference(sel_columns).values.tolist()

            print(self.sequence_role(new_data))

            # Remove sequence column(s)
            not_sel_columns = [item for item in not_sel_columns if item not in self.sequence_role(new_data)]

            sel_acc_dict = {el: self.sel_acc.value for el in sel_columns}
            not_sel_acc_dict = {el: self.not_sel_acc.value for el in not_sel_columns}
            sel_acc_dict.update(not_sel_acc_dict)

            acc_dict = sel_acc_dict
        else:
            # Dictionary with accumulators is provided
            new_dict = {}
            # Go through supplied Selector, Accumulator pairs
            for key in self.acc_dict:
                variables = key(new_data)
                # Add variables associated with selector with supplied accumulator
                new_dict.update({el: self.sel_acc.value for el in variables})

            # Add non-specified variables with not_sel_acc, if not a sequence role
            new_dict.update(
                {
                    el: self.not_sel_acc.value
                    for el in new_data.columns.difference(new_dict.keys())
                    if el not in self.sequence_role(new_data)
                }
            )
            acc_dict = new_dict

        # Resample per stay id
        new_data = new_data.groupby(groups()(data))

        # Resampling with the functions defined in sel_dictionary
        new_data = new_data.resample(self.new_resolution, on=self.sequence_role(data)[0]).agg(acc_dict)

        # Remove multi-index
        new_data = new_data.droplevel(groups()(data))
        new_data = new_data.reset_index(drop=False)
        new_data = new_data.set_index(groups()(data))

        return new_data
