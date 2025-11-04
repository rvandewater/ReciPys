from __future__ import annotations
from collections import Counter
from copy import copy
from itertools import chain
from typing import Union

import pandas as pd
import polars as pl

from .ingredients import Ingredients
from .selector import select_groups
from .step import Step
from .constants import Backend


class Recipe:
    """Recipe for preprocessing data

    A Recipe object combines a pandas-like Ingredients object with one or more
    sklearn-inspired transformation Steps to turn into a model-ready input.

    Args:
        data: data to be preprocessed.
        outcomes: names of columns in data that are assigned the 'outcome' role
        predictors: names of columns in data that should be assigned the 'predictor' role
        groups: names of columns in data that should be assigned the 'group' role
        sequence: names of columns in data that should be assigned the 'sequence' role
    """

    columns = None
    roles = None

    def __init__(
        self,
        data: Ingredients | pl.DataFrame | pd.DataFrame,
        outcomes: Union[str, list[str]] = None,
        predictors: Union[str, list[str]] = None,
        groups: Union[str, list[str]] = None,
        sequences: Union[str, list[str]] = None,
        backend: Backend = None,
    ):
        if not isinstance(data, Ingredients):
            try:
                data = Ingredients(data, backend=backend)
            except Exception as e:
                raise (f"Expected Ingredients, got {data.__class__} {e}")
        self.data = data
        self.steps = []
        self.original_columns = copy(data.columns)
        self.roles = self.data.roles
        self.columns = self.data.columns

        if outcomes:
            self.update_roles(outcomes, "outcome")
        if predictors:
            self.update_roles(predictors, "predictor")
        if groups:
            self.update_roles(groups, "group")
        if sequences:
            self.update_roles(sequences, "sequence")

    def add_roles(self, vars: Union[str, list[str]], new_role: str = "predictor") -> Recipe:
        """Adds an additional role for one or more columns of the Recipe's Ingredients.

        Args:
            vars: The column to receive additional roles.
            new_role: Defaults to predictor. The role to add to the column.

        See also:
            Ingredients.add_role()

        Returns:
            self
        """
        if isinstance(vars, str):
            vars = [vars]
        for v in vars:
            self.data.add_role(v, new_role)
        return self

    def update_roles(self, vars: Union[str, list[str]], new_role: str = "predictor", old_role: str = None) -> Recipe:
        """Adds a new role for one or more columns of the Recipe's Ingredients without roles
        or changes an existing role to a different one.

        Args:
            vars: The column to receive additional roles.
            new_role: Defaults to predictor'. The role to add or change to.
            old_role: Defaults to None. The role to be changed.

        See also:
            Ingredients.update_role()

        Returns:
            self
        """
        if isinstance(vars, str):
            vars = [vars]
        for v in vars:
            self.data.update_role(v, new_role, old_role)
        return self

    def add_step(self, step: Step) -> Recipe:
        """Adds a new step to the Recipe

        Args:
            step: a transformation step that should be applied to the Ingredients during prep() and bake()

        Returns:
            self
        """
        self.steps.append(step)
        return self

    def _check_data(self, data: Union[pl.DataFrame | pd.DataFrame, Ingredients]) -> Ingredients:
        if data is None:
            data = self.data
        elif isinstance(data, pl.DataFrame) or isinstance(data, pd.DataFrame):
            # this is only executed when prep or bake receive a DF that is different to the original data
            # don't check the roles here, because self.data can have more roles than data (post feature generation)
            data = Ingredients(data, roles=self.data.roles, check_roles=False)
        # if not data.columns.equals(self.data.columns):
        if not set(data.columns) == set(self.original_columns):
            raise ValueError(
                f"Columns of data argument differs from recipe data: "
                f"{[x for x in data.columns if x not in self.original_columns]}."
            )
        return data

    def _apply_group(self, data, step):
        if step.group:
            group_vars = select_groups(data)
            if len(group_vars) > 0:
                data.groupby(group_vars)
        return data

    def prep(
        self, data: Union[pl.DataFrame | pd.DataFrame, Ingredients] = None, refit: bool = False
    ) -> pl.DataFrame | pd.DataFrame:
        """Fits and transforms, in other words preps, the data.

        Args:
            data: Data to fit and transform. Defaults to None.
            refit: Defaults to False. Whether to refit data.

        Returns:
            Transformed data.
        """
        data = self._check_data(data)
        # Todo: check why the roles dissapear after copying
        data = copy(data)
        data = self._apply_fit_transform(data, refit)
        # return pl.DataFrame(data)
        return data.get_df()

    def bake(self, data: Union[pl.DataFrame | pd.DataFrame, Ingredients] = None) -> pl.DataFrame | pd.DataFrame:
        """Transforms, or bakes, the data if it has been prepped.

        Args:
            data: Data to transform. Defaults to None.

        Returns:
            Transformed data.
        """
        data = self._check_data(data)
        # original_data = deepcopy(data)
        data = self._apply_fit_transform(data)
        # return pl.DataFrame(data)
        return data.get_df()

    def _apply_fit_transform(self, data=None, refit=False):
        # applies transform or fit and transform (when refit or not trained yet)
        for step in self.steps:
            data = self._apply_group(data, step)
            if refit or not step.trained:
                data = step.fit_transform(data)
            else:
                data = step.transform(data)
        return data

    def __repr__(self):
        repr = "Recipe\n\n"

        # Print all existing roles and how many variables are assigned to each
        num_roles = Counter(chain.from_iterable(self.data.roles.values()))
        num_roles = pl.DataFrame({"role": [r for r in num_roles.keys()], "#variables": [n for n in num_roles.values()]})
        repr += "Inputs:\n\n" + num_roles.__repr__() + "\n\n"

        # Print all steps
        repr += "Operations:\n\n"
        for step in self.steps:
            repr += str(step) + "\n"

        return repr

    def get_backend(self):
        return self.data.get_backend()

    def cache(self):
        """Prepares the recipe for caching"""
        if self.data is not None:
            del self.data
        return self
