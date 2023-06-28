from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import click
import pandas as pd
from tqdm.auto import tqdm


def file_exists(path):
    return Path(path).is_file()


def check_df(df_path):
    cols = [f"FileName_Orig{chan}" for chan in ["DNA", "AGP", "ER", "Mito", "RNA"]]

    paths = pd.read_parquet(df_path).loc[:, cols].stack().tolist()

    missing = 0
    for path in paths:
        missing += not file_exists(path)

    return missing, len(paths)


def download_with_multiprocessing_generator(paths, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(check_df, path): path for path in paths}

        for future in futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()

            if not exception:
                yield key, future.result()
            else:
                yield key, exception


def download_objects_from_paths(paths, max_workers=4):
    missing = 0
    tot = 0
    with tqdm(total=len(paths)) as pbar:
        download_results = []
        for path, result in download_with_multiprocessing_generator(paths, max_workers):
            pbar.update(1)
            missing += result[0] if isinstance(result, tuple) else 0
            tot += result[1] if isinstance(result, tuple) else 0
            tmp = {
                "path": path,
                "missing": result[0] if isinstance(result, tuple) else result,
                "total": result[1] if isinstance(result, tuple) else None,
            }
            download_results.append(tmp)

            pbar.set_postfix({"missing": missing, "total": tot})

    return download_results


@click.command()
@click.option(
    "--path_to_df",
    "-i",
    default="/projects/cpjump1/jump/load_data/final",
    help="Path to the dataframe.",
)
@click.option(
    "--path_to_output",
    "-o",
    default="/projects/cpjump1/jump/load_data/check",
    help="Path to the output.",
)
@click.option("--max_workers", "-m", default=16, help="Number of workers to use.")
def main(path_to_df, path_to_output, max_workers):
    print(f"Reading dataframe from {path_to_df}...")
    print(f"Writing output to {path_to_output}...")
    print(f"Using {max_workers} workers...")

    dfs = glob(f"{path_to_df}/*.parquet")
    download_results = download_objects_from_paths(dfs, max_workers=max_workers)

    df = pd.DataFrame(download_results)

    print("Writing output stats...")
    ex = df["missing"].sum()
    tot = df["total"].sum()
    print(f"{ex}/{tot} ({ex/tot:.0%}) files exist.")

    Path(path_to_output).mkdir(exist_ok=True)

    df.to_csv(f"{path_to_output}/stats.csv", index=False)

    with open(f"{path_to_output}/stats.txt", "w") as f:
        f.write(f"{ex}/{tot} ({ex/tot:.0%}) files exist.")


if __name__ == "__main__":
    main()
