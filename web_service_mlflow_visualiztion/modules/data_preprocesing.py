import os
import json
import numpy as np
import pandas as pd


class DataPreprocessor:
    def __init__(self, competition_name):

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, f"data/{competition_name}/prepocessed")
        self.df_raw_path = os.path.join(project_root, f"data/{competition_name}/raw")
        self.df_path = data_path
        os.makedirs(self.df_path, exist_ok=True)

    def adjust_column_names(self, df):
        df = df.copy()
        df.columns = df.columns.str.replace(r"[.\(\) ]", "_", regex=True)
        return df

    def downcast_int_columns(self, df):
        int_columns = df.select_dtypes(include=["int64", "int32"]).columns
        for col in int_columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        return df

    def downcast_float_columns(self, df):
        float_columns = df.select_dtypes(include=["float64", "float32"]).columns
        for col in float_columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        return df

    def convert_object_to_category(self, df):
        object_columns = df.select_dtypes(include=["object"]).columns
        for col in object_columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype("category")
            else:
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    continue
        return df

    def convert_float_to_int(self, df):
        float_columns = df.select_dtypes(include=["float32", "float64"]).columns
        for col in float_columns:
            if np.all(np.modf(df[col])[0] == 0):  # Checking if all fractional parts are 0
                min_val, max_val = df[col].min(), df[col].max()
                if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
        return df

    def store_df(self, df, filename):

        df.to_pickle(os.path.join(self.df_path, filename))
        print(f"data stored in {os.path.join(self.df_path, filename)}")

    def optimize_dtypes(self, df, verbose=True, convert_float_to_int=True, save=True):
        def memory_usage_in_gb(df):
            return df.memory_usage(deep=True).sum() / (1024**3)

        if verbose:
            initial_memory_gb = memory_usage_in_gb(df)
            print(f"Initial memory usage: {initial_memory_gb:.6f} GB")

        # create
        old_columns = df.columns
        df = self.adjust_column_names(df)
        old_dtype = df.dtypes.apply(lambda x: x.name).to_dict()
        df = self.downcast_int_columns(df)
        df = self.downcast_float_columns(df)
        df = self.convert_object_to_category(df)
        if convert_float_to_int:
            df = self.convert_float_to_int(df)

        if verbose:
            final_memory_gb = memory_usage_in_gb(df)
            print(f"Final memory usage: {final_memory_gb:.6f} GB")
            print(
                f"Memory reduced from {initial_memory_gb:.6f} GB to {final_memory_gb:.6f} GB, "
                f"which is a reduction of {initial_memory_gb - final_memory_gb:.6f} GB "
                f"({100 * (initial_memory_gb - final_memory_gb) / initial_memory_gb:.2f}%)."
            )

        new_columns = df.columns
        new_dtype = df.dtypes.apply(lambda x: x.name).to_dict()
        # create a dictionary of old and new columns and data types
        column_dict = dict(zip(old_columns, new_columns))
        # Combine old and new dtypes
        combined_dtypes = {}
        for key in set(old_dtype.keys()).union(new_dtype.keys()):
            combined_dtypes[key] = {
                "old_dtype": old_dtype.get(key),
                "new_dtype": new_dtype.get(key),
            }

        if save:
            self.store_df(df, f"data_dtype_optimized.pkl")
            # save save_dict as json file
            with open(os.path.join(self.df_path, "column_changes.json"), "w") as f:
                json.dump(column_dict, f, indent=4)

            with open(os.path.join(self.df_path, "dtype_changes.json"), "w") as f:
                json.dump(combined_dtypes, f, indent=4)

        return df
