import os
import ast
import pandas as pd
import numpy as np

from pandas import DataFrame


def csv_to_parquet(folder_path, path_to_save, overwrite=False):
    # Get list of CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if len(csv_files) == 0:
        print("No CSV files found.")
        return

    # Convert each CSV file to Parquet if the corresponding Parquet file doesn't exist
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        parquet_path = os.path.join(path_to_save, csv_file.replace('.csv', '.parquet'))

        # Check if the Parquet file already exists
        if not os.path.exists(parquet_path) or overwrite:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)
            except UnicodeDecodeError:
                # Handle encoding issues
                df = pd.read_csv(csv_path, encoding='ISO-8859-1')

            # Save as Parquet with the same name as the CSV but .parquet extension
            df.to_parquet(parquet_path)
            print(f"Converted {csv_file} to {parquet_path}")
        else:
            print(f"Parquet file for {csv_file} already exists and will not be overwritten.")

def reduce_memory_usage_pd(
        df: pd.DataFrame,
        verbose: bool = True) -> pd.DataFrame:
    """Optimize memory usage of a pandas DataFrame."""

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif pd.api.types.is_float_dtype(col_type):
                max_val = df[col].max()
                if max_val < np.finfo(np.float32).max:  # Check if max value fits in float32
                    df[col] = df[col].astype(np.float32)  # Downcast to float32

        elif pd.api.types.is_object_dtype(col_type):
            num_unique = df[col].nunique()
            num_total = len(df[col])

            # Convert strings to categorical if unique values are much less than total
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:
        print(f"Memory before: {start_mem:.2f} MB. \n"
              f"Memory after: {end_mem:.2f} MB.\n"
              f"Percent of reduction: ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")

    return df

def missing_percentage(
        df: pd.DataFrame) -> DataFrame:
    """
    Calculate the percentage of missing values in a DataFrame.
    :param df: pandas DataFrame    :return:
    """
    total_missing = df.isnull().sum()
    percent_of_missing = total_missing / df.isnull().count() * 100
    concat_missing = pd.concat(
        [total_missing, percent_of_missing],
        axis=1,
        keys=["Total_missing", "Percent_missing"]
    ).sort_values(by=["Percent_missing"], ascending=False)
    return concat_missing

def count_duplicated_rows(
        df: pd.DataFrame) -> None:
    """
    Count and print the number of duplicated rows in a DataFrame
    (based on all columns).
    """
    num_duplicated_rows = df.duplicated().sum()
    print(f"The DataFrame contains {num_duplicated_rows} duplicated rows.")

def clean_params(param_str):
    if isinstance(param_str, str):
        param_dict = ast.literal_eval(param_str)  # safely convert string to dict
        return {k.replace("model__", ""): v for k, v in param_dict.items()}
    return param_str
