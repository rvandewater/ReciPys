from copy import deepcopy
import pandas as pd
import polars as pl
from typing import overload
from src.recipies.constants import Backend


class Ingredients:
    """Wrapper around either polars.DataFrame to store columns roles (e.g., predictor)
        Due to the workings of polars, we do not subclass pl.dataframe anymore,
        but instead store the dataframe as an attribute.
    Args:
        roles: roles of DataFrame columns as (list of) strings.
            Defaults to None.
        check_roles: If set to false, doesn't check whether the roles match existing columns.
            Defaults to True.

    See also: pandas.DataFrame

    Attributes:
        roles (dict): dictionary of column roles
    """

    _metadata = ["roles"]

    def __init__(
        self,
        data: pl.DataFrame | pd.DataFrame = None,
        copy: bool = None,
        roles: dict = None,
        check_roles: bool = True,
        backend: Backend = None,
    ):
        if backend is None:
            if isinstance(data, pl.DataFrame):
                self.backend = Backend.POLARS
            elif isinstance(data, pd.DataFrame):
                self.backend = Backend.PANDAS
            elif isinstance(data, Ingredients):
                self.backend = data.get_backend()
            else:
                raise ValueError("Backend not specified and could not be inferred from data.")
        else:
            self.backend = backend
        if isinstance(data, pd.DataFrame) or isinstance(data, pl.DataFrame):
            if self.backend == Backend.POLARS:
                if isinstance(data, pd.DataFrame):
                    self.data = pl.DataFrame(data)
                elif isinstance(data, pl.DataFrame):
                    self.data = data
                else:
                    raise TypeError(f"Expected DataFrame, got {data.__class__}")
            elif self.backend == Backend.PANDAS:
                if isinstance(data, pd.DataFrame):
                    self.data = data
                if isinstance(data, pl.DataFrame):
                    self.data = data.to_pandas()
            else:
                raise ValueError(f"Backend {self.backend} not supported.")
            self.schema = self.get_schema()
            self.dtypes = self.get_schema()

        if isinstance(data, Ingredients) and roles is None:
            if copy is None or copy is True:
                self.roles = deepcopy(data.roles)
            else:
                self.roles = data.roles
            self.data = data.data
            self.schema = data.schema
            self.dtypes = self.schema

        elif roles is None:
            self.roles = {}
        elif not isinstance(roles, dict):
            raise TypeError(f"Expected dict object for roles, got {roles.__class__}")
        elif check_roles and not all(set(k).issubset(set(self.data.columns)) for k, v in roles.items()):
            raise ValueError(
                f"Roles contains variable names that are not in the data {list(roles.values())} {self.data.columns}."
            )
        # Todo: do we want to allow ingredients without grouping columns?
        # elif check_roles and select_groups(self) == []:
        #     raise ValueError("Roles are given but no groups are found in the data.")
        else:
            if copy is None or copy is True:
                self.roles = deepcopy(roles)
            else:
                self.roles = roles

    @property
    def _constructor(self):
        return Ingredients

    @property
    def columns(self):
        return self.data.columns

    def to_df(self, output_format=None) -> pl.DataFrame:
        """Return the underlying DataFrame.


        Returns:
            Self as DataFrame.
        """
        if output_format == Backend.POLARS:
            if self.backend == Backend.POLARS:
                return self.data
            else:
                return pl.DataFrame(self.data)
        elif output_format == Backend.PANDAS:
            if self.backend == Backend.POLARS:
                return self.data.to_pandas()
            else:
                return self.data
        else:
            return self.data

    def _check_column(self, column):
        if not isinstance(column, str):
            raise ValueError(f"Expected string, got {column}")
        if column not in self.columns:
            raise ValueError(f"{column} does not exist in this Data object")

    def _check_role(self, new_role):
        if not isinstance(new_role, str):
            raise TypeError(f"new_role must be string, was {new_role.__class__}")

    def add_role(self, column: str, new_role: str):
        """Adds an additional role for a column that already has roles.

        Args:
            column: The column to receive additional roles.
            new_role: The role to add to the column.

        Raises:
            RuntimeError: If the column has no role yet.
        """
        self._check_column(column)
        self._check_role(new_role)
        if column not in self.roles.keys():
            raise RuntimeError(f"{column} has no roles yet, use update_role instead.")
        self.roles[column] += [new_role]

    def update_role(self, column: str, new_role: str, old_role: str = None):
        """Adds a new role for a column without roles or changes an existing role to a different one.

        Args:
            column: The column to update the roles of.
            new_role: The role to add or change to.
            old_role: Defaults to None. The role to be changed.

        Raises:
            ValueError: If old_role is given but column has no roles.
                If old_role is given but column has no role old_role.
                If no old_role is given but column has multiple roles already.
        """
        self._check_column(column)
        self._check_role(new_role)
        if old_role is not None:
            if column not in self.roles.keys():
                raise ValueError(
                    f"Attempted to update role of {column} from {old_role} to {new_role} "
                    f"but {column} does not have a role yet."
                )
            elif old_role not in self.roles[column]:
                raise ValueError(
                    f"Attempted to set role of {column} from {old_role} to {new_role} "
                    f"but {old_role} not among current roles: {self.roles[column]}."
                )
            self.roles[column].remove(old_role)
            self.roles[column].append(new_role)
        else:
            if column not in self.roles.keys() or len(self.roles[column]) == 1:
                self.roles[column] = [new_role]
            else:
                raise ValueError(
                    f"Attempted to update role of {column} to {new_role} but "
                    f"{column} has more than one current roles: {self.roles[column]}"
                )

    def select_dtypes(self, include=None):
        # if(isinstance(include,[str])):
        dtypes = self.get_str_dtypes()
        selected = [key for key, value in dtypes.items() if value in include]
        return selected

    def get_dtypes(self):
        dtypes = list(self.schema.values())
        return dtypes

    def get_str_dtypes(self):
        """ "
        Helper function for polar dataframes to return schema with dtypes as strings
        """
        dtypes = self.get_schema()
        return {key: str(value) for key, value in dtypes.items()}
        # return list(map(dtypes, cast()))

    def get_schema(self):
        if self.backend == Backend.POLARS:
            return self.data.schema
        else:
            return self.data.dtypes

    def get_df(self):
        return self.to_df()

    def set_df(self, df):
        self.data = df

    def groupby(self, by):
        if self.backend == Backend.POLARS:
            self.data.group_by(by)
        else:
            return self.data.groupby(by)

    def get_backend(self):
        return self.backend

    def __setitem__(self, idx, val):
        if self.backend == Backend.POLARS:
            self.data[idx] = val
        else:
            if isinstance(idx, tuple):
                rows, column = idx
                self.data.loc[rows, column] = val
            else:
                # Use assign for single column assignment to avoid fragmentation
                if isinstance(idx, str):
                    self.data = self.data.assign(**{idx: val})
                else:
                    # For non-string indices, use a more efficient approach
                    # that avoids the fragmentation warning
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore",
                                              message="DataFrame is highly fragmented",
                                              category=pd.errors.PerformanceWarning)
                        self.data[idx] = val

    @overload
    def __getitem__(self, list: list[str]) -> pl.DataFrame:
        return self.data[list]

    def __getitem__(self, idx: int) -> pl.Series:
        return self.data[idx]
