
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import random
# Enable StringCache
pl.StringCache()




def generate_generic_datetime_data(stay_ids: list[int], max_rows: int, start_time: datetime, timestep: timedelta, columns: list[str], missingness: dict[str, float],
                                   categorical_columns=None) -> pl.DataFrame:
    """
    Generate a Polars DataFrame with the specified number of rows and schema for multiple stay IDs.

    Parameters:
    - stay_ids: The list of stay IDs to use.
    - max_rows: The maximum number of rows to generate.
    - start_time: The starting time for the time column.
    - timestep: The time interval between rows.
    - columns: A list of column names to generate data for.
    - missingness: A dictionary where keys are column names and values are the percentage of missing values.

    Returns:
    - A Polars DataFrame with the generated data.
    """
    # Distribute rows randomly among stay IDs
    rows_per_stay_id = np.random.multinomial(max_rows, [1/len(stay_ids)]*len(stay_ids))

    all_data = []

    for stay_id, num_rows in zip(stay_ids, rows_per_stay_id):
        # Generate the time column
        time_column = [start_time + i * timestep for i in range(num_rows)]

        # Initialize the data dictionary with the time and stay_id columns
        data = {
            "stay_id": [stay_id] * num_rows,
            "time": time_column
        }

        # Generate random data for other columns
        for column in columns:
            col_data = np.random.rand(num_rows)
            # Introduce missingness
            if column in missingness:
                missing_count = int(num_rows * missingness[column])
                missing_indices = np.random.choice(num_rows, missing_count, replace=False)
                col_data[missing_indices] = np.nan
            data[column] = col_data

        # Generate categorical data if specified
        if categorical_columns:
            for column, values in categorical_columns.items():
                col_data = random.choices(values, k=num_rows)
                # Introduce missingness
                if column in missingness:
                    missing_count = int(num_rows * missingness[column])
                    missing_indices = random.sample(range(num_rows), missing_count)
                    for idx in missing_indices:
                        col_data[idx] = None
                data[column] = col_data
                data[column] = pl.Series(data[column], dtype=pl.Categorical)
                # data[column] = pl.Series(data[column], dtype=pl.Enum)


        # Create the DataFrame for the current stay_id
        df = pl.DataFrame(data)
        all_data.append(df)

    # Concatenate all DataFrames
    final_df = pl.concat(all_data)

    return final_df

def generate_icu_data(max_rows: int, missingness_threshold: tuple[int,int]=(0.1,0.7), seed:int=None) -> pl.DataFrame:
    # Example usage
    if seed is not None:
        random.seed(seed)
    stay_ids = random.sample(range(1, max_rows*100), int(max_rows/24))
    start_time = datetime(2024, 1, 1)
    timestep = timedelta(hours=1)

    # Define the list of column names
    columns = ["alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun", "ca", "cai", "ck",
               "ckmb", "cl", "crea", "crp", "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map",
               "mch", "mchc", "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp",
               "sbp", "temp", "tnt", "urine", "wbc"]
    categorical_columns = {"surgery":["Cardiac", "General", "Orthopedic", "Neurosurgery", "Vascular"],
                           "drainage":["Chest", "Abdominal", "Other"],
                           "ventilator":["Yes", "No"],
                           }
    # Define the missingness dictionary
    missingness = {item:random.uniform(missingness_threshold[0], missingness_threshold[1]) for item in columns}
    generated_df = generate_generic_datetime_data(stay_ids, max_rows, start_time, timestep, columns, missingness, categorical_columns)
    return generated_df




