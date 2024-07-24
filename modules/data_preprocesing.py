import os
import json
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    A class for preprocessing data.

    Args:
        competition_name (str): The name of the competition.
        project_root (str, optional): The root directory of the project. Defaults to the parent directory of the current file.

    Attributes:
        df_raw_path (str): The path to the raw data.
        df_path (str): The path to the preprocessed data.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(self, competition_name: str, project_root: str = None):
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        data_path = os.path.join(project_root, f"data/{competition_name}/preprocessed")
        self.df_raw_path = os.path.join(project_root, f"data/{competition_name}/raw")
        self.df_path = data_path
        os.makedirs(self.df_path, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.config = {
            "category_threshold": 0.5,
            "int_types": ["int8", "int16", "int32", "int64"],
            "float_types": ["float32", "float64"],
        }

    @staticmethod
    def adjust_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Adjusts column names by replacing special characters with underscores."""
        df = df.copy()
        df.columns = df.columns.str.replace(r"[.\(\) ]", "_", regex=True)
        return df

    def downcast_numeric_columns(self, df: pd.DataFrame, dtype: str) -> pd.DataFrame:
        """Downcasts numeric columns to reduce memory usage."""
        numeric_columns = df.select_dtypes(include=self.config[f"{dtype}_types"]).columns
        downcast_type = "integer" if dtype == "int" else "float"
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], downcast=downcast_type)
        return df

    def convert_object_to_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts object columns to category or datetime columns based on uniqueness."""
        object_columns = df.select_dtypes(include=["object"]).columns
        for col in object_columns:
            unique_ratio = len(df[col].unique()) / len(df[col])
            if unique_ratio < self.config["category_threshold"]:
                df[col] = df[col].astype("category")
            else:
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert column {col} to datetime.")
        return df

    def convert_float_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts float columns to integer columns if all fractional parts are 0."""
        float_columns = df.select_dtypes(include=self.config["float_types"]).columns
        for col in float_columns:
            if np.all(np.modf(df[col])[0] == 0):
                df[col] = self._select_int_type(df[col])
        return df

    @staticmethod
    def _select_int_type(series: pd.Series) -> pd.Series:
        """Selects the appropriate integer type for a series."""
        min_val, max_val = series.min(), series.max()
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            if min_val >= np.iinfo(dtype).min and max_val <= np.iinfo(dtype).max:
                return series.astype(dtype)
        return series.astype(np.int64)

    def store_df(self, df: pd.DataFrame, filename: str) -> None:
        """Stores the DataFrame as a pickle file."""
        file_path = os.path.join(self.df_path, filename)
        df.to_pickle(file_path)
        self.logger.info(f"Data stored in {file_path}")

    def optimize_dtypes(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
        convert_float_to_int: bool = True,
        save: bool = True,
    ) -> pd.DataFrame:
        """Optimizes the data types of the DataFrame by applying various preprocessing steps."""
        if verbose:
            initial_memory_gb = self._memory_usage_in_gb(df)
            self.logger.info(f"Initial memory usage: {initial_memory_gb:.6f} GB")

        old_columns = df.columns
        df = self.adjust_column_names(df)
        old_dtype = df.dtypes.apply(lambda x: x.name).to_dict()

        df = self.downcast_numeric_columns(df, "int")
        df = self.downcast_numeric_columns(df, "float")
        df = self.convert_object_to_category(df)

        if convert_float_to_int:
            df = self.convert_float_to_int(df)

        if verbose:
            final_memory_gb = self._memory_usage_in_gb(df)
            self.logger.info(f"Final memory usage: {final_memory_gb:.6f} GB")
            self.logger.info(
                f"Memory reduced by {initial_memory_gb - final_memory_gb:.6f} GB ({100 * (initial_memory_gb - final_memory_gb) / initial_memory_gb:.2f}%)."
            )

        new_columns = df.columns
        new_dtype = df.dtypes.apply(lambda x: x.name).to_dict()

        column_dict = dict(zip(old_columns, new_columns))
        combined_dtypes = {
            key: {"old_dtype": old_dtype.get(key), "new_dtype": new_dtype.get(key)}
            for key in set(old_dtype) | set(new_dtype)
        }

        if save:
            self.store_df(df, "data_dtype_optimized.pkl")
            self._save_json(column_dict, "column_changes.json")
            self._save_json(combined_dtypes, "dtype_changes.json")

        return df

    @staticmethod
    def _memory_usage_in_gb(df: pd.DataFrame) -> float:
        """Calculates the memory usage of a DataFrame in GB."""
        return df.memory_usage(deep=True).sum() / (1024**3)

    def _save_json(self, data: Dict[str, Any], filename: str) -> None:
        """Saves a dictionary as a JSON file."""
        file_path = os.path.join(self.df_path, filename)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        self.logger.info(f"JSON data saved to {file_path}")
