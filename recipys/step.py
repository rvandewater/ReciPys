from abc import abstractmethod
from copy import deepcopy
from typing import Union, Dict
from scipy.sparse import isspmatrix
import polars as pl
from polars.dataframe.group_by import GroupBy
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

    def _check_ingredients(self, data: Union[Ingredients, ]) -> Ingredients:
        """Check input for allowed types

        Args:
            data: input to the step

        Raises:
            ValueError: If a grouped pd.DataFrame is provided to a step that can't use groups.
            ValueError: If input are not (potentially grouped) Ingredients.

        Returns:
            Validated input
        """
        if isinstance(data, GroupBy):
            if not self._group:
                raise ValueError("Step does not accept grouped data.")
            # data = data.apply(lambda df: df)
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
    def __init__(self, sel=all_predictors(), value=None, strategy=None, limit=None):
        super().__init__(sel)
        self.desc = f"Impute with {strategy if strategy else value}"
        self.value = value
        self.strategy = strategy
        self.limit = limit

    def transform(self, data):
        new_data = self._check_ingredients(data)
        selected_cols = pl.col(self.columns)
        groups = select_groups(new_data)
        if len(select_groups(new_data))>0:
            new_data.data = data.data.with_columns(
                pl.col(self.columns).fill_null(self.value, strategy=self.strategy, limit=self.limit).over(select_groups(new_data)))
        else:
            new_data.data = data.data.with_columns(
                pl.col(self.columns).fill_null(self.value, strategy=self.strategy, limit=self.limit))
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
        self.suffix = "_" + self.suffix
        new_columns = [c + self.suffix for c in self.columns]

        selected = new_data.data
        selected_cols = pl.col(self.columns)
        id = select_groups(new_data)
        if self.fun is Accumulator.MAX:
            res = selected.with_columns(selected_cols.cum_max().over(id).name.suffix(self.suffix))
        elif self.fun is Accumulator.MIN:
            res = selected.with_columns(selected_cols.cum_min().over(id).name.suffix(self.suffix))
        elif self.fun is Accumulator.MEAN:
            res = selected.with_columns(selected_cols.rolling_mean(window_size=selected.height, min_periods=0)
                                  .over(id).name.suffix(self.suffix))
        elif self.fun is Accumulator.MEDIAN:
            res = selected.with_columns(selected_cols.rolling_median(window_size=selected.height, min_periods=0)
                                  .over(id).name.suffix(self.suffix))
        elif self.fun is Accumulator.COUNT:
            res = selected.with_columns(selected_cols.cum_count().over(id).name.suffix(self.suffix))
        elif self.fun is Accumulator.VAR:
            res = selected.with_columns(selected_cols.rolling_var(window_size=selected.height, min_periods=0)
                                  .over(id).name.suffix(self.suffix))
        else:
            raise TypeError(f"Expected Accumulator enum for function, got {self.fun.__class__}")
        new_data.data = res
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
                print(data[self.columns])
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
                if isinstance(col_names,str):
                    col_names = [col_names]
                updated_cols = pl.from_numpy(new_cols, schema=col_names)
                new_data.data = new_data.data.with_columns(updated_cols)
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
                else [f"{self.sklearn_transformer.__class__.__name__}_{self.columns[i]}" for i in range(new_cols.shape[1])]
                if new_cols.shape[1] == len(self.columns)
                else
                    [f"{self.sklearn_transformer.__class__.__name__}_{i + 1}" for i in range(new_cols.shape[1])]
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
        """This class represents a resampling step in a recipe.

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

        # if not (isinstance(pl.datatypes.TemporalType,sequence_datatype)): #or is_datetime64_any_dtype(sequence_datatype)):
        if not (sequence_datatype.is_temporal()):  # or is_datetime64_any_dtype(sequence_datatype)):
            raise ValueError(f"Expected Timedelta or Timestamp object, got {sequence_role(data).__class__}")

        # Dictionary with the format column: str , accumulator:str is created
        col_acc_map = {}
        # Go through supplied Selector, Accumulator pairs
        for (selector, accumulator) in self.acc_dict.items():
            selected_columns = selector(new_data)
            # Add variables associated with selector with supplied accumulator
            col_acc_map.update({col: accumulator.value for col in selected_columns})

        # Add non-specified variables, if not a sequence role
        col_acc_map.update(
            {
                col: self.default_accumulator.value
                for col in set(new_data.columns).difference(col_acc_map.keys())
                if col not in select_sequence(new_data)
            }
        )
        # acc_col_map = dict((v, k) for k, v in col_acc_map.items())
        from collections import defaultdict

        acc_col_map = defaultdict(list)
        for k, v in col_acc_map.items():
            acc_col_map[v].append(k)

        grouping_role = select_groups(new_data)[0]
        # Resampling with the functions defined in col_acc_map
        print(acc_col_map)
        print(acc_col_map["mean"])
        new_data.set_df(new_data.get_df().sort(grouping_role, sequence_role).set_sorted(sequence_role))
        new_data.set_df(new_data.get_df().upsample(every=self.new_resolution, time_column=sequence_role, group_by=grouping_role)
                        .with_columns(pl.col(acc_col_map["last"]).fill_null(strategy="forward"))
                        .with_columns(pl.col(acc_col_map["mean"]).fill_null(strategy="mean"))
                        .with_columns(pl.col(acc_col_map["max"]).fill_null(strategy="max"))
                       .with_columns(pl.col(grouping_role).fill_null(strategy="forward")))
        return new_data


class StepScale(StepSklearn):
    """Provides a wrapper for a scaling with StepSklearn.
    Note that because SKlearn transforms None (nulls) to NaN, we have to revert.

    Args:
       with_mean: Defaults to True. If True, center the data before scaling.
       with_std: Defaults to True. If True, scale the data to unit variance (or equivalently, unit standard deviation).
       in_place: Defaults to True. Set to False to have the step generate new columns instead of overwriting the existing ones.
       role (str, optional): Defaults to 'predictor'. Incase new columns are added, set their role to role.
    """

    # def __new__(
    #     cls,
    #     sel: Selector = all_numeric_predictors(),
    #     with_mean: bool = True,
    #     with_std: bool = True,
    #     in_place: bool = True,
    #     role: str = "predictor",
    # ):
    #     return super(StepScale,self).StandardScaler(with_mean=with_mean, with_std=with_std), sel=sel, in_place=in_place, role=role)
    def __init__(self,
                 sel=all_numeric_predictors(),
                 with_mean: bool = True,
                 with_std: bool = True,
                 *args, **kwargs):
        super().__init__(sklearn_transformer=StandardScaler(with_mean=with_mean,with_std=with_std), sel=sel,
                         in_place=True, *args, **kwargs)
        self.desc = "Scale with StandardScaler"


    def transform(self, data: Ingredients) -> Ingredients:
        data = super().transform(data)
        # Revert null to nan conversion done by sklearn
        data.set_df(data.get_df().with_columns(pl.col(self.columns).fill_nan(None)))
        return data


