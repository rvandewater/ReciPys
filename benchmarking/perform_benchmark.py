import argparse
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import polars as pl
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, \
    QuantileTransformer, SplineTransformer

from generate_data import generate_icu_data
from src.recipies.constants import Backend
from src.recipies import Recipe
from src.recipies.selector import all_numeric_predictors, all_predictors
from src.recipies.step import StepScale, StepSklearn, StepImputeFill, StepHistorical, Accumulator
import logging
import polars.selectors as cs
from memory_profiler import memory_usage
from tqdm import tqdm

def benchmark_dynamic_recipe(data, backend):
    if backend == Backend.POLARS:
        numeric_predictors = data.select(pl.exclude("stay_id", "time")).columns
    else:
        numeric_predictors = list(set(data.columns).difference(["stay_id", "time"]))
    dyn_rec = Recipe(data, [],numeric_predictors, "stay_id", "time", backend=backend)

    dyn_rec.add_step(StepScale(all_numeric_predictors(backend)))

    dyn_rec.add_step(StepSklearn(MissingIndicator(features="all"), all_predictors(), in_place=False))
    # dyn_rec.add_step(StepImputeFastForwardFill())

    dyn_rec.add_step(StepImputeFill(all_predictors(), strategy="forward"))
    # dyn_rec.add_step(StepImputeFastZeroFill())
    dyn_rec.add_step(StepImputeFill(all_predictors(), strategy="zero"))

    dyn_rec = dynamic_feature_generation(dyn_rec, backend)
    data = dyn_rec.bake()
    # print(data.head())


def benchmark_step(data, backend, step):
    timer = datetime.now()
    if backend == Backend.POLARS:
        numeric_predictors = data.select(pl.exclude("stay_id", "time")).columns
    else:
        numeric_predictors = list(set(data.columns).difference(["stay_id", "time"]))

    recipe = Recipe(data, [],numeric_predictors, "stay_id", "time", backend=backend)
    recipe.add_step(step)
    data = recipe.bake()
    time_passed = datetime.now() - timer
    return time_passed


def dynamic_feature_generation(data, backend):
    data.add_step(StepHistorical(sel=all_numeric_predictors(), fun=Accumulator.MIN, suffix="min_hist"))
    data.add_step(StepHistorical(sel=all_numeric_predictors(), fun=Accumulator.MAX, suffix="max_hist"))
    data.add_step(StepHistorical(sel=all_numeric_predictors(), fun=Accumulator.COUNT, suffix="count_hist"))
    data.add_step(StepHistorical(sel=all_numeric_predictors(), fun=Accumulator.MEAN, suffix="mean_hist"))
    return data


def benchmark_backend(backend, data_size, seed):
    # metrics = {}
    df_missing = generate_icu_data(data_size, seed=seed)
    df_complete = generate_icu_data(data_size, missingness_threshold=(0,0), seed=seed)
    if backend == Backend.PANDAS:
        backend_name = "Pandas"
        df_missing = df_missing.to_pandas()
        df_complete = df_complete.to_pandas()
    else:
        backend_name = "Polars"
    steps_missing = [
            StepImputeFill(all_predictors(), strategy="forward"),
            StepSklearn(MissingIndicator(features="all"), all_predictors(), in_place=False)]
    steps_complete = [
        # StepSklearn(
        #  OrdinalEncoder(),
        #  sel=has_type([str(pl.Categorical(ordering="physical")) if backend == backend.POLARS else "category"]),
        #  in_place=False,),
        # StepSklearn(
        #     OneHotEncoder(sparse_output=False),
        #     sel=has_type([str(pl.Categorical(ordering="physical")) if backend == backend.POLARS else "category"]),
        #     in_place=False,
        # ),
        StepSklearn(
         KBinsDiscretizer(n_bins=2, strategy="uniform", encode="ordinal"),
         sel=all_numeric_predictors(),
         in_place=False,
        ),
        StepHistorical(sel=all_numeric_predictors(), fun=Accumulator.MEAN, suffix="Mean"),
        StepHistorical(sel=all_numeric_predictors(), fun=Accumulator.MIN, suffix="Min"),
        StepHistorical(sel=all_numeric_predictors(), fun=Accumulator.MAX, suffix="Max"),
        StepHistorical(sel=all_numeric_predictors(), fun=Accumulator.COUNT, suffix="Count"),]

    for item in [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]:
        steps_complete.append(StepSklearn(item, sel=all_numeric_predictors()))

    for transformer in [PowerTransformer(), QuantileTransformer(n_quantiles=10), SplineTransformer()]:
        steps_complete.append(StepSklearn(transformer, sel=all_numeric_predictors(), in_place=False))

    # benchmark_dynamic_recipe(df, backend)
    results = []

    def run_step_benchmark(backend, backend_name, data_size, results, step, df):
        metrics = {}
        metrics["data_size"] = data_size
        mem_usage, time_passed = memory_usage((benchmark_step, (df, backend, step)), retval=True, max_iterations=1, interval=0.5)
        metrics["backend"] = backend_name
        if isinstance(step, StepSklearn):
            metrics["step"] = str(step.sklearn_transformer.__class__.__name__)
        elif isinstance(step, StepHistorical):
            metrics["step"] = f"Historical{step.suffix}"
        else:
            metrics["step"] = str(step.__class__.__name__)
        metrics["time_passed"] = time_passed / timedelta(microseconds=1)
        metrics["memory_usage"] = max(mem_usage)
        results.append(metrics)

    for step in steps_complete:
        run_step_benchmark(backend, backend_name, data_size, results, step, df_complete)

    for step in steps_missing:
        run_step_benchmark(backend, backend_name, data_size, results, step, df_missing)
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking script for recipies with ICU data generation and preprocessing.",
                                     usage="python perform_benchmark.py --data_sizes 10000 100000 --seeds 42 41")
    parser.add_argument('--data_sizes', type=int, nargs='+', default=[1000], help='List of data sizes to benchmark.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 41], help='Random seeds for data generation.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data_sizes = args.data_sizes
    seeds = args.seeds

    # Detect if output is being redirected to a file
    use_tqdm = sys.stdout.isatty() and sys.stderr.isatty()

    # Create filename once at the beginning
    csv_filename = f'results_datasizes_{data_sizes}_seeds_{seeds}_datetime_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv'
    size_results = []
    with pl.StringCache():
        data_size_iter = tqdm(data_sizes, desc="Processing data sizes", unit="size") if use_tqdm else data_sizes
        for data_size in data_size_iter:
            logging.info(f"Starting with data size {data_size}")
            timer = datetime.now()

            # Collect results for this data size
            seed_iter = tqdm(seeds, desc=f"Seeds for size {data_size}", unit="seed", leave=False) if use_tqdm else seeds
            for seed in seed_iter:
                timer = datetime.now()
                logging.info(f"Starting with seed {seed}")
                polars = benchmark_backend(Backend.POLARS, data_size, seed=seed)
                pandas = benchmark_backend(Backend.PANDAS, data_size, seed=seed)
                polars = [pl.from_dict(item) for item in polars]
                pandas = [pl.from_dict(item) for item in pandas]
                polars = pl.concat(polars, how="vertical_relaxed")
                pandas = pl.concat(pandas, how="vertical_relaxed")
                size_results.extend([polars, pandas])
                logging.info(f"Time taken for seed {seed}: {datetime.now() - timer}")

            # Process results for this data size
            df_size = pl.concat(size_results, how="vertical_relaxed")
            df_size = df_size.group_by(["data_size", "step", "backend"]).agg([
                pl.col("time_passed").mean().alias("duration_mean"),
                pl.col("time_passed").std().alias("duration_std"),
                pl.col("memory_usage").mean().alias("memory_mean"),
                pl.col("memory_usage").std().alias("memory_std")
            ])

            columns = ["duration_mean", "duration_std", "memory_mean", "memory_std"]
            df_size = (df_size
                      .pivot(on="backend", values=columns, index=["data_size", "step"])
                      .with_columns(
                          speed_difference=(pl.col("duration_mean_Pandas") - pl.col("duration_mean_Polars")),
                          speedup=(pl.col("duration_mean_Pandas") / pl.col("duration_mean_Polars"))
                      ))
            df_size = df_size.with_columns(cs.numeric().round(1))
            df_size = df_size.sort(by=["data_size", "step"])

            df_size.write_csv(csv_filename)

            logging.info(f"Results for data size {data_size} written to {csv_filename}")
            print(f"Completed data size {data_size}")
# if __name__ == "__main__":
#     args = parse_args()
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     data_sizes = args.data_sizes
#     seeds = args.seeds
#
#     df = None
#     df_list = []
#     for data_size in data_sizes:
#         logging.info(f"Starting with data size {data_size}")
#         timer = datetime.now()
#         for seed in seeds:
#             timer = datetime.now()
#             logging.info(f"Starting with seed {seed}")
#             combined = {}
#             polars = benchmark_backend(Backend.POLARS, data_size, seed=seed)
#             pandas = benchmark_backend(Backend.PANDAS, data_size, seed=seed)
#             polars = [pl.from_dict(item) for item in polars]
#             pandas = [pl.from_dict(item) for item in pandas]
#             polars = pl.concat(polars, how="vertical_relaxed")
#             pandas = pl.concat(pandas, how="vertical_relaxed")
#             if df is not None:
#                 df = pl.concat([df, polars,pandas], how="vertical_relaxed")
#             else:
#                 df = pl.concat([polars,pandas], how="vertical_relaxed")
#             logging.info(f"Time taken for data size {data_size}: {datetime.now() - timer}")
#             df_list.append(df)
#
#     df = pl.concat(df_list, how="vertical_relaxed")
#     df = df.group_by(["data_size", "step", "backend"]).agg([
#         pl.col("time_passed").mean().alias("duration_mean"),
#         pl.col("time_passed").std().alias("duration_std"),
#         pl.col("memory_usage").mean().alias("memory_mean"),
#         pl.col("memory_usage").std().alias("memory_std")
#     ])
#     columns = ["duration_mean", "duration_std", "memory_mean", "memory_std"]
#
#     df = (df#.group_by(["data_size", "step", "backend"])#.agg(pl.col("time_passed_mean"), pl.col("time_passed_std"))
#           .pivot(on="backend",values=columns, index=["data_size", "step"])
#           .with_columns(speed_difference = (pl.col("duration_mean_Pandas") - pl.col("duration_mean_Polars")),
#                         speedup = (pl.col("duration_mean_Pandas") / pl.col("duration_mean_Polars"))))
#     df = df.with_columns(cs.numeric().round(1))
#         #  value_name="time_passed"))
#     df = df.sort(by=["data_size", "step"])
#     df.write_csv(f'results_datasizes_{[data_sizes]}_seeds_{[seeds]}_datetime_{datetime.now():%Y-%m-%d_%H-%M-%S%z}.csv')
#     print(df)
        # combined = pl.concat([pl.from_dict(combined), pl.from_dict(polars, strict=False), pl.from_dict(pandas, strict=False)], how="vertical_relaxed")
        # prefix = "pandas"
        # pandas = {f"{prefix}_{key}": value for key, value in pandas.items()}
        # prefix = "polars"
        # polars = {f"{prefix}_{key}": value for key, value in polars.items()}
        # combined = {**polars, **pandas}

        # df = generate_icu_data(1000000)
        # timer = datetime.now()
        # benchmark_dynamic_recipe(df, Backend.POLARS)
        # polars_time = datetime.now() - timer
        # timer = datetime.now()
        # benchmark_dynamic_recipe(df, Backend.PANDAS)
        # pandas_time = datetime.now() - timer
        # print(f"Polars: time taken: {polars_time}")
        # print(f"Pandas time taken: {pandas_time}")
        # print(f"Speedup: {pandas_time/polars_time}")
