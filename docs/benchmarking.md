# Benchmarking

This page provides an overview of the benchmarking process implemented in the `perform_benchmark.py` script. The benchmarking evaluates the performance of various preprocessing steps in the `ReciPies` library using different data sizes and backends (Polars and Pandas).

---

## Overview

The benchmarking script measures the **execution time** and **memory usage** of preprocessing steps applied to synthetic ICU data. The results are aggregated and saved as a CSV file for further analysis.

---

## How It Works

### 1. **Synthetic Data Generation**
The script uses the `generate_icu_data` function to create synthetic ICU data with configurable:

- **Data sizes**: Number of rows in the dataset.
- **Missingness thresholds**: Proportion of missing values in the dataset.
- **Random seeds**: For reproducibility.

### 2. **Backends**
The benchmarking supports two backends:

- **Polars**: A high-performance DataFrame library.
- **Pandas**: The standard Python DataFrame library.

The script converts the synthetic data to the appropriate backend format before running the benchmarks.

### 3. **Preprocessing Steps**
The following preprocessing steps are benchmarked:

- **Imputation**:
  - `StepImputeFill`: Forward-fill and zero-fill strategies.
  - `StepSklearn`: Using `MissingIndicator` from `sklearn`.
- **Scaling**:
  - `StepSklearn`: Using scalers like `StandardScaler`, `MinMaxScaler`, etc.
- **Discretization**:
  - `StepSklearn`: Using `KBinsDiscretizer`.
- **Historical Accumulation**:
  - `StepHistorical`: Using accumulators like `MIN`, `MAX`, `MEAN`, and `COUNT`.

### 4. **Metrics**
For each preprocessing step, the script measures:

- **Execution Time**: The time taken to apply the step.
- **Memory Usage**: The peak memory usage during the step.

---

## Running the Benchmark

### Command
Run the script using the following command:

```bash
python perform_benchmark.py --data_sizes 1000 10000 --seeds 42 41
```

### Arguments
- `--data_sizes`: A list of data sizes to benchmark (e.g., `1000`, `10000`).
- `--seeds`: A list of random seeds for reproducibility.

### Example
```bash
python perform_benchmark.py --data_sizes 1000 10000 100000 --seeds 42 43
```

---

## Results

### Output
The script generates a CSV file with the benchmarking results. The filename includes the data sizes, seeds, and a timestamp. For example:

```
results_datasizes_[1000, 10000]_seeds_[42, 43]_datetime_2025-11-19_12-00-00.csv
```

### Aggregated Metrics
The results include:

- **Mean Execution Time** (`duration_mean`)
- **Standard Deviation of Execution Time** (`duration_std`)
- **Mean Memory Usage** (`memory_mean`)
- **Standard Deviation of Memory Usage** (`memory_std`)
- **Speed Difference**: Difference in execution time between Pandas and Polars.
- **Speedup**: Ratio of Pandas execution time to Polars execution time.

---

## Example Workflow

### 1. **Dynamic Recipe Benchmark**
The `benchmark_dynamic_recipe` function benchmarks a dynamic recipe with multiple preprocessing steps applied sequentially.

### 2. **Step-Specific Benchmark**
The `benchmark_step` function benchmarks a single preprocessing step.

### 3. **Backend Comparison**
The `benchmark_backend` function compares the performance of Polars and Pandas for the same preprocessing steps.

---

## Example Results

### Sample Output
| Data Size | Step               | Backend | Duration Mean (ms) | Memory Mean (MB) | Speedup |
|-----------|--------------------|---------|---------------------|------------------|---------|
| 1000      | StepImputeFill     | Pandas  | 50.0                | 10.0             | 1.5     |
| 1000      | StepImputeFill     | Polars  | 33.3                | 8.0              | 1.5     |
| 10000     | StepHistoricalMean | Pandas  | 500.0               | 50.0             | 2.0     |
| 10000     | StepHistoricalMean | Polars  | 250.0               | 30.0             | 2.0     |

---

## Customization

### Adding New Steps
To benchmark additional steps:

1. Define the step in the `steps_complete` or `steps_missing` list.
2. Ensure the step is compatible with both Pandas and Polars.

### Modifying Metrics
To add or modify metrics:

1. Update the `run_step_benchmark` function to include the new metric.
2. Update the aggregation logic in the `benchmark_backend` function.

---

## Notes

- **Dependencies**: Ensure all required libraries (e.g., `polars`, `pandas`, `sklearn`, `tqdm`, `memory_profiler`) are installed.
- **Performance**: Polars is generally faster and more memory-efficient than Pandas for large datasets.
- **Reproducibility**: Use the `--seeds` argument to ensure consistent results across runs.

---

## Next Steps

- Analyze the results using the `aggregate_results.ipynb` notebook.
- Use the benchmarking results to optimize preprocessing pipelines in the `ReciPies` library.
```