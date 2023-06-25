"""Module that runs the bash script to download the JUMP metadata files."""

import os
import subprocess

import hydra
import pandas as pd
from omegaconf import DictConfig


def get_metadata_df(
    metadata_dir: str,
    force: bool,
):
    """Create a dataframe containing all the important metadata and saves it to a csv."""
    plate_csv_file = os.path.join(metadata_dir, "plate.csv")
    well_csv_file = os.path.join(metadata_dir, "well.csv")
    compound_csv_file = os.path.join(metadata_dir, "compound.csv")
    orf_csv_file = os.path.join(metadata_dir, "orf.csv")
    crispr_csv_file = os.path.join(metadata_dir, "crispr.csv")
    microscope_config_csv_file = os.path.join(metadata_dir, "microscope_config.csv")
    # microscope_filter_csv_file = os.path.join(metadata_dir, "microscope_filter.csv")
    complete_metadata_csv_file = os.path.join(metadata_dir, "complete_metadata.csv")

    if not force and os.path.exists(complete_metadata_csv_file):
        print("Metadata csv already exists, loading it.")
        return pd.read_csv(complete_metadata_csv_file)

    print("Loading metadata csvs...")
    plate_df = pd.read_csv(plate_csv_file)
    well_df = pd.read_csv(well_csv_file)
    compound_df = pd.read_csv(compound_csv_file)
    orf_df = pd.read_csv(orf_csv_file)
    crispr_df = pd.read_csv(crispr_csv_file)
    microscope_config_df = pd.read_csv(microscope_config_csv_file).assign(
        Metadata_Source=lambda x: "source_" + x["Metadata_Source"].astype(str)
    )

    # microscope_filter_df = pd.read_csv(microscope_filter_csv_file)
    # resolution = pd.read_csv(os.path.join(metadata_dir, "resolution.csv"))

    print("Merging metadata csvs...")
    metadata_df = (
        plate_df.merge(well_df, how="outer", on=["Metadata_Source", "Metadata_Plate"])
        .merge(compound_df, how="left", on=["Metadata_JCP2022"])
        .merge(orf_df, how="left", on=["Metadata_JCP2022"])
        .merge(crispr_df, how="left", on=["Metadata_JCP2022"])
        .merge(microscope_config_df, how="left", on=["Metadata_Source"])
        # .merge(resolution, how="left", on=["Metadata_Source"])
        # .merge(microscope_filter_df, how="left", on=["Metadata_Filter_Configuration"])
    )

    print(f"Saving metadata csv to {complete_metadata_csv_file} ...")
    metadata_df.to_csv(complete_metadata_csv_file, index=False)

    return metadata_df


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Download the JUMP metadata files."""
    metadata_script = cfg.output_dirs.metadata_download_script
    metadata_dir = cfg.output_dirs.metadata_dir
    force = cfg.run.force

    # Run bash script to download metadata files
    bash_script = f"bash {metadata_script} {metadata_dir} {'--force' if force else '--no-force'}"
    print(f"=== Running bash script: {bash_script}")
    subprocess.run(bash_script, shell=True)

    print("\n=== Creating complete metadata file...")
    get_metadata_df(metadata_dir, force=force)


if __name__ == "__main__":
    main()
