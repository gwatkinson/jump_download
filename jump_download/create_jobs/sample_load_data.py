"""Module that creates csv files containing the jobs to run wiht bioclust."""

# flake8: noqa

import os
from functools import partial
from glob import glob
from pathlib import Path
from typing import Callable, Optional

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


def source_and_plate_filter(
    load_data: pd.DataFrame,
    plate_types_to_keep: list[str],
    sources_to_exclude: list[str],
) -> pd.DataFrame:
    """Filter the load_data dataframe based on plate type and source.

    This function filters the load_data dataframe based on the plate type and source.
    And assigns an unique id and a drop_sample column that indicates if the sample should be dropped.

    Args:
        load_data (pandas.DataFrame): The load_data dataframe.
        plate_types_to_keep (list[str], optional): The plate types to keep. Defaults to PLATE_TYPES_TO_KEEP.
        sources_to_exclude (list[str], optional): The sources to exclude. Defaults to SOURCES_TO_EXCLUDE.

    Returns:
        pandas.DataFrame: The filtered load_data dataframe.
    """
    load_data = load_data.assign(
        id=(
            load_data["Metadata_Source"].astype(str)
            + "/"
            + load_data["Metadata_Batch"].astype(str)
            + "/"
            + load_data["Metadata_Plate"].astype(str)
            + "/"
            + load_data["Metadata_Well"].astype(str)
            + "/"
            + load_data["Metadata_Site"].astype(str)
        ),
        drop_sample=(
            (~load_data["Metadata_PlateType"].isin(plate_types_to_keep))
            | (load_data["Metadata_Source"].isin(sources_to_exclude))
            | (load_data["_merge"] != "both")
        ),
    )

    return load_data


def sample_from_load_data(
    load_data: pd.DataFrame,
    plate_type: str | list[str],
    positive_controls: list[str],
    negative_controls: list[str],
    neg_per_well: int,
    pos_per_well: int,
    trt_per_well: int,
    filter_var: str,
    pert_to_drop: list[str] | None = None,
    sample_per_source: bool = False,
) -> dict[str, pd.Series]:
    """Sample from the load_data dataframe with controls.

    This takes the entire load_data dataframe and samples from it.
    It samples at the well level, taking the minimum between the number of samples in the well and the given value.

    Args:
        load_data (pandas.DataFrame): The dataframe to sample from.
        plate_type (str | list[str]): The plate type to sample from.
        positive_controls (list): The list of positive controls to sample.
        negative_controls (list): The list of negative controls to sample.
        total_controls (list): The list of total controls to sample.
        neg_per_well (int): The number of negative controls to sample per well.
        pos_per_well (int): The number of positive controls to sample per well.
        trt_per_well (int): The number of treatments to sample per well.
        filter_var (str): The variable to filter on (inchi, jcp2022, ...).
        pert_to_drop (list[str] | None, optional): Perturbations to drop. Defaults to None.
        sample_per_source (bool, optional): Whether to sample per source. Defaults to False.

    Returns:
        dict[str, pd.Series]: A dictionary of the samples.
            Each key is the type of plate with the sample type (positive, negative, treatment)
            and the value is a boolean array the length of the load data dataframe,
            indicating which observation were sampled.
    """
    samples = {}

    total_controls = positive_controls + negative_controls
    plate_type_list = plate_type if isinstance(plate_type, list) else [plate_type]
    first_query = "drop_sample == False"
    first_query += " & ~{filter_var}.isin({pert_to_drop})" if pert_to_drop else ""

    print("Sampling positive controls")
    positive_control_samples = (
        load_data.query(first_query)
        .query(f"Metadata_PlateType.isin({plate_type_list})")
        .query(f"{filter_var}.isin({positive_controls})")
        .groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"])
        .apply(lambda x: x.sample(min(pos_per_well, len(x)), replace=False))["id"]
    )

    samples[f"{plate_type}_poscon"] = load_data["id"].isin(positive_control_samples)

    print("Sampling negative controls")
    negative_control_samples = (
        load_data.query(first_query)
        .query(f"Metadata_PlateType.isin({plate_type_list})")
        .query(f"{filter_var}.isin({negative_controls})")
        .groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"])
        .apply(lambda x: x.sample(min(neg_per_well, len(x)), replace=False))["id"]
    )

    samples[f"{plate_type}_negcon"] = load_data["id"].isin(negative_control_samples)

    print("Sampling treatment")
    if sample_per_source:
        sources = load_data["Metadata_Source"].unique()

        for source in tqdm(sources):
            trt_samples = (
                load_data.query(first_query)
                .query(f"Metadata_PlateType.isin({plate_type_list})")
                .query("Metadata_Source == @source")
                .query(f"~{filter_var}.isin({total_controls})")
                .groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"])
                .apply(lambda x: x.sample(min(trt_per_well, len(x)), replace=False))["id"]
            )

            samples[f"{plate_type}_trt_{source}"] = load_data["id"].isin(trt_samples)

    else:
        trt_samples = (
            load_data.query(first_query)
            .query(f"Metadata_PlateType.isin({plate_type_list})")
            .query(f"~{filter_var}.isin({total_controls})")
            .groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"])
            .apply(lambda x: x.sample(min(trt_per_well, len(x)), replace=False))["id"]
        )

        samples[f"{plate_type}_trt"] = load_data["id"].isin(trt_samples)

    return samples


def sample_compound(
    load_data: pd.DataFrame,
    positive_controls: list[str],
    negative_controls: list[str],
    neg_per_well: int,
    pos_per_well: int,
    trt_per_well: int,
    pert_to_drop: list[str] | None = None,
    filter_var: str = "Metadata_InChI",
    sample_per_source: bool = True,
):
    return sample_from_load_data(
        load_data=load_data,
        plate_type="COMPOUND",
        positive_controls=positive_controls,
        negative_controls=negative_controls,
        neg_per_well=neg_per_well,
        pos_per_well=pos_per_well,
        trt_per_well=trt_per_well,
        filter_var=filter_var,
        pert_to_drop=pert_to_drop,
        sample_per_source=sample_per_source,
    )


def sample_orf(
    load_data: pd.DataFrame,
    positive_controls: list[str],
    negative_controls: list[str],
    neg_per_well: int,
    pos_per_well: int,
    trt_per_well: int,
    pert_to_drop: list[str] | None = None,
    filter_var: str = "Metadata_JCP2022",
    sample_per_source: bool = False,
):
    return sample_from_load_data(
        load_data=load_data,
        plate_type="ORF",
        positive_controls=positive_controls,
        negative_controls=negative_controls,
        neg_per_well=neg_per_well,
        pos_per_well=pos_per_well,
        trt_per_well=trt_per_well,
        filter_var=filter_var,
        pert_to_drop=pert_to_drop,
        sample_per_source=sample_per_source,
    )


def sample_crispr(
    load_data: pd.DataFrame,
    positive_controls: list[str],
    negative_controls: list[str],
    neg_per_well: int,
    pos_per_well: int,
    trt_per_well: int,
    pert_to_drop: list[str] | None = None,
    filter_var: str = "Metadata_JCP2022",
    sample_per_source: bool = False,
):
    return sample_from_load_data(
        load_data=load_data,
        plate_type="CRISPR",
        positive_controls=positive_controls,
        negative_controls=negative_controls,
        neg_per_well=neg_per_well,
        pos_per_well=pos_per_well,
        trt_per_well=trt_per_well,
        filter_var=filter_var,
        pert_to_drop=pert_to_drop,
        sample_per_source=sample_per_source,
    )


def sample_target(
    load_data: pd.DataFrame,
    positive_controls: list[str],
    negative_controls: list[str],
    neg_per_well: int,
    pos_per_well: int,
    trt_per_well: int,
    pert_to_drop: list[str] | None = None,
    filter_var: str = "Metadata_JCP2022",
    sample_per_source: bool = False,
):
    return sample_from_load_data(
        load_data=load_data,
        plate_type=["TARGET1", "TARGET2"],
        positive_controls=positive_controls,
        negative_controls=negative_controls,
        neg_per_well=neg_per_well,
        pos_per_well=pos_per_well,
        trt_per_well=trt_per_well,
        filter_var=filter_var,
        pert_to_drop=pert_to_drop,
        sample_per_source=sample_per_source,
    )


def create_output_dir_column(
    load_data: pd.DataFrame, source_to_dict: dict[str, str], col_name: Optional[str] = "output_dir"
) -> pd.DataFrame:
    source_to_disk = pd.Series(source_to_dict, name=col_name)
    return load_data.merge(source_to_disk, left_on="Metadata_Source", right_index=True)


def create_job_id_column(
    load_data: pd.DataFrame, col_name: Optional[str] = "job_id"
) -> pd.DataFrame:
    def transform_func(row):
        kept = row["filter"]
        source = row["Metadata_Source"]
        batch = row["Metadata_Batch"]
        plate = row["Metadata_Plate"]

        if not kept:
            return f"{source}__dropped"
        else:
            return f"{source}__{batch}__{plate}"

    job_id = load_data.apply(transform_func, axis=1)

    load_data[col_name] = job_id

    return load_data


def create_filter_column(
    load_data: pd.DataFrame,
    samples_dict: list[dict[str, pd.Series]],
    output_dir_func: Callable[[pd.DataFrame, Optional[str]], pd.DataFrame],
    job_id_func: Callable[[pd.DataFrame, Optional[str]], pd.DataFrame],
) -> pd.DataFrame:
    """Create a filter column and a job id column for the load data dataframe."""

    print("Marking rows to keep")
    samples = pd.DataFrame({k: v for d in samples_dict for k, v in d.items()})
    rows_to_keep = samples.apply(lambda row: any(row), axis=1)

    print(
        f"Keeping {rows_to_keep.sum():,}/{rows_to_keep.shape[0]:,} rows ({rows_to_keep.sum()/rows_to_keep.shape[0]:.0%})"
    )
    load_data["filter"] = rows_to_keep

    print("Adding output dir column")
    load_data = output_dir_func(load_data, "output_dir")

    print("Adding job id column")
    load_data = job_id_func(load_data, "job_id")

    return load_data


def create_submission_csv(job_output_dir):
    jobs_csv = glob(os.path.join(job_output_dir, "ids", "*.csv.gz"))
    submission_df_path = os.path.join(job_output_dir, "submission.csv")
    job_ids = [Path(p).stem.split(".")[0] for p in jobs_csv]
    dropped = ["__dropped" in p for p in job_ids]

    submission_df = (
        pd.DataFrame(
            {
                "jobs_csv": jobs_csv,
                "dropped": dropped,
            }
        )
        .query("dropped == False")
        .drop(columns=["dropped"])
        .sample(frac=1, random_state=42)
    )

    submission_df.to_csv(submission_df_path, index=False, header=False)


@hydra.main(config_path="config", config_name="config")
def sample_load_data(cfg: DictConfig):
    load_data_dir = cfg.output_dir.load_data_dir
    job_output_dir = cfg.output_dir.job_output_dir

    load_data_with_metadata_out_dir = os.path.join(load_data_dir, "load_data_with_metadata")
    load_data_samples_out_dir = os.path.join(load_data_dir, "load_data_with_samples")
    job_output_dir_ids = Path(job_output_dir) / "ids"
    job_output_dir_ids.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading load data with metadata...")
    load_data_with_metadata = pd.read_parquet(load_data_with_metadata_out_dir, use_threads=False)

    print("\n=== Filtering load data on plate type and source...")
    load_data = source_and_plate_filter(
        load_data=load_data_with_metadata,
        plate_types_to_keep=cfg.filters.plate_types_to_keep,
        sources_to_exclude=cfg.filters.sources_to_exclude,
    )

    print("\n=== Sampling from COMPOUND load data...")
    compound_samples = sample_compound(
        load_data=load_data,
        positive_controls=cfg.filters.compound_negative_controls,
        negative_controls=cfg.filters.compound_positive_controls,
        neg_per_well=cfg.filters.compound_number_of_negcon_to_keep_per_well,
        pos_per_well=cfg.filters.compound_number_of_poscon_to_keep_per_well,
        trt_per_well=cfg.filters.compound_number_of_trt_to_keep_per_well,
        pert_to_drop=None,
        filter_var="Metadata_InChI",
        sample_per_source=True,
    )

    print("\n=== Sampling from ORF load data...")
    orf_samples = sample_orf(
        load_data=load_data,
        positive_controls=cfg.filters.orf_negative_controls,
        negative_controls=cfg.filters.orf_positive_controls,
        neg_per_well=cfg.filters.orf_number_of_negcon_to_keep_per_well,
        pos_per_well=cfg.filters.orf_number_of_poscon_to_keep_per_well,
        trt_per_well=cfg.filters.orf_number_of_trt_to_keep_per_well,
        pert_to_drop=cfg.filters.orf_pert_to_drop,
        filter_var="Metadata_JCP2022",
        sample_per_source=False,
    )

    print("\n=== Sampling from CRISPR load data...")
    crispr_samples = sample_crispr(
        load_data=load_data,
        positive_controls=cfg.filters.crispr_negative_controls,
        negative_controls=cfg.filters.crispr_positive_controls,
        neg_per_well=cfg.filters.crispr_number_of_negcon_to_keep_per_well,
        pos_per_well=cfg.filters.crispr_number_of_poscon_to_keep_per_well,
        trt_per_well=cfg.filters.crispr_number_of_trt_to_keep_per_well,
        pert_to_drop=cfg.filters.crispr_pert_to_drop,
        filter_var="Metadata_JCP2022",
        sample_per_source=False,
    )

    print("\n=== Sampling from TARGET load data...")
    target_samples = sample_target(
        load_data=load_data,
        positive_controls=cfg.filters.target_negative_controls,
        negative_controls=cfg.filters.target_positive_controls,
        neg_per_well=cfg.filters.target_number_of_negcon_to_keep_per_well,
        pos_per_well=cfg.filters.target_number_of_poscon_to_keep_per_well,
        trt_per_well=cfg.filters.target_number_of_trt_to_keep_per_well,
        pert_to_drop=cfg.filters.target_pert_to_drop,
        filter_var="Metadata_JCP2022",
        sample_per_source=False,
    )

    print("\n=== Creating filter column and job ids...")
    samples_dict = [compound_samples, orf_samples, crispr_samples, target_samples]
    job_df = create_filter_column(
        load_data=load_data,
        samples_dict=samples_dict,
        output_dir_func=partial(
            create_output_dir_column, source_to_dict=cfg.output_dir.source_to_disk
        ),
        job_id_func=create_job_id_column,
    )

    print("\n=== Saving job dataframe to parquet...")
    job_df.to_parquet(
        path=load_data_samples_out_dir,
        index=False,
        engine="pyarrow",
        compression="snappy",
        partition_cols=["Metadata_Source"],
    )

    print("\n=== Splitting job dataframe into csv files...")
    split = {k: v for k, v in job_df.groupby("job_id")}
    for k, v in tqdm(split.items()):
        output = os.path.join(job_output_dir, "ids", f"{k}.csv.gz")
        v.to_csv(output, index=False, compression="gzip")

    print("\n=== Creating submission csv...")
    create_submission_csv(job_output_dir)


if __name__ == "__main__":
    sample_load_data()
