"""Module to create total load data file for all plates in the JUMP dataset."""

import os
import shutil
from glob import glob
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

from jump_download.base_class import BaseDownload
from jump_download.utils import apply_dtypes_with_large_dict, get_size_of_folder


class DownloadLoadData(BaseDownload):
    """Class that defines the methods to download load data files from S3."""

    def __init__(
        self,
        plate_metadata_file,
        load_data_dir,
        tmp_load_data_dir,
        remove_tmp,
        channels,
        max_workers,
        force,
    ) -> None:
        super().__init__(max_workers)
        self.plate_metadata_file = plate_metadata_file
        self.plate_df = pd.read_csv(plate_metadata_file)
        self.load_data_dir = load_data_dir
        self.tmp_load_data_dir = tmp_load_data_dir
        self.remove_tmp = remove_tmp
        self.channels = channels
        self.force = force
        self.query_format = "s3://cellpainting-gallery/cpg0016-jump/{source}/workspace/load_data_csv/{batch}/{plate}/load_data_with_illum.parquet"
        self.total_image_file = os.path.join(load_data_dir, "total_load_data.csv.gz")
        self.total_illum_file = os.path.join(load_data_dir, "total_illum.csv.gz")

    def get_job_list(self):
        """Create a list of jobs to download load data files from S3, one for each plate in the JUMP dataset."""
        out_dir = self.tmp_load_data_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        jobs = []
        for _, row in tqdm(self.plate_df.iterrows(), total=len(self.plate_df)):
            source = row["Metadata_Source"]
            batch = row["Metadata_Batch"]
            plate = row["Metadata_Plate"]

            output = os.path.join(out_dir, f"{source}_{batch}_{plate}.csv")

            skip = not self.force and os.path.exists(output)

            jobs.append(
                {
                    "skip": skip,
                    "source": source,
                    "batch": batch,
                    "plate": plate,
                    "out_dir": output,
                }
            )

        print(f"Skipping {sum([job['skip'] for job in jobs])}/{len(jobs)} jobs")

        return jobs

    def execute_job(self, job):
        """Execute a single load data job that consists of downloading a load data file from S3 and saving it to disk."""
        if job["skip"]:
            return "Skipped"

        source = job["source"]
        batch = job["batch"]
        plate = job["plate"]
        output = job["out_dir"]

        load_data_df = pd.read_parquet(
            self.query_format.format(source=source, batch=batch, plate=plate),
            storage_options={"anon": True},
        )

        load_data_df.to_csv(output, index=False)

        return "Success"

    def post_process(self, download_results) -> None:
        # Check that all jobs were successful
        print("\n=== Checking that all load data files were successfully downloaded...")
        result_df = pd.DataFrame(download_results)
        if (result_df["result"].isin(["Success", "Skipped"])).all():
            print("All images downloaded successfully")
        else:
            print("Not all images were downloaded successfully")
            raise Exception(f"Not all images were downloaded successfully: {result_df}")

        # List of all load data files in tmp directory
        csv_files = glob(f"{self.tmp_load_data_dir}/*.csv")

        # Columns to keep in total load data files
        key_cols = [
            "Metadata_Source",
            "Metadata_Batch",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_Site",
        ]
        illum_cols = [
            *key_cols,
            *[f"FileName_Illum{channel}" for channel in self.channels],
            *[f"PathName_Illum{channel}" for channel in self.channels],
        ]
        image_cols = [
            *key_cols,
            *[f"FileName_Orig{channel}" for channel in self.channels],
            *[f"PathName_Orig{channel}" for channel in self.channels],
        ]

        # Path to total load data files
        Path(self.total_illum_file).parent.mkdir(parents=True, exist_ok=True)

        # Remove total load data files if they already exist
        if os.path.exists(self.total_illum_file):
            os.remove(self.total_illum_file)
        if os.path.exists(self.total_image_file):
            os.remove(self.total_image_file)

        # Create total load data files
        print("\n=== Creating total load data files...")
        for file in tqdm(csv_files):
            df = pd.read_csv(file)
            # Separate illum and image columns
            illum_df = df[illum_cols].astype(str).drop_duplicates()
            image_df = df[image_cols].astype(str)
            illum_df.to_csv(
                self.total_illum_file,
                mode="a",
                index=False,
                header=(not os.path.exists(self.total_illum_file)),
                compression="gzip",
            )
            image_df.to_csv(
                self.total_image_file,
                mode="a",
                index=False,
                header=(not os.path.exists(self.total_image_file)),
                compression="gzip",
            )

        # Delete temporary load data files
        if self.remove_tmp:
            print("\n=== Deleting temporary load data files...")
            size_of_folder = get_size_of_folder(self.tmp_load_data_dir)
            print(f"Size of {self.tmp_load_data_dir}: {size_of_folder / 1e9:.2f} GB")
            shutil.rmtree(self.tmp_load_data_dir)

        return

    def download(self) -> None:
        """Method that queues the other methods."""
        if not self.force:
            print("\n=== Checking if total load data files already exist...")
            if os.path.exists(self.total_image_file) and os.path.exists(self.total_illum_file):
                print("Total load data files already exist. Skipping.")
                print(
                    f"Total image file in {self.total_image_file} ({os.path.getsize(self.total_image_file) / 1e6:.2f} MB)"
                )
                print(
                    f"Total illum file in {self.total_illum_file} ({os.path.getsize(self.total_illum_file) / 1e6:.2f} MB)"
                )
                return

        # Create a list of jobs to download load data files from S3
        print("\n=== Creating load data jobs...")
        jobs = self.get_job_list()

        # Run the jobs in parallel using multiprocessing
        print("\n=== Run the jobs in parallel...")
        download_results = self.download_objects_from_jobs(jobs)

        # Post process the results
        print("\n=== Post processing the results...")
        self.post_process(download_results)

        # Print the size of the total load data files
        print("\n=== Print results ...")
        print(
            f"Saved total load data file to {self.total_image_file} ({os.path.getsize(self.total_image_file) / 1e6:.2f} MB)"
        )
        print(
            f"Saved total illum file to {self.total_illum_file} ({os.path.getsize(self.total_illum_file) / 1e6:.2f} MB)"
        )

        return


def get_load_data_with_metadata(
    total_load_data_csv_file,
    complete_metadata_csv_file,
    load_data_with_metadata_out_dir,
    total_load_data_dtypes=None,
):
    """Get the total load data with metadata."""
    total_load_data_dtypes = total_load_data_dtypes or {}

    print("Loading metadata csv...")
    complete_metadata_df = pd.read_csv(complete_metadata_csv_file)
    complete_metadata_df = apply_dtypes_with_large_dict(
        complete_metadata_df, total_load_data_dtypes
    )

    print("Loading total load data csv...")
    total_load_data_df = pd.read_csv(total_load_data_csv_file, compression="gzip")
    total_load_data_df = apply_dtypes_with_large_dict(total_load_data_df, total_load_data_dtypes)

    # print("Save total load data to pickle...")
    # total_load_data_df.to_pickle(Path(total_load_data_csv_file).with_suffix(".pkl.gz"), compression="gzip")

    print("Merging total load data with metadata...")
    images_with_metadata = total_load_data_df.merge(
        complete_metadata_df,
        on=[
            "Metadata_Source",
            "Metadata_Batch",
            "Metadata_Plate",
            "Metadata_Well",
        ],
        how="left",
        indicator=True,
    ).drop(
        columns=[
            "Metadata_Microscope_Name",
            "Metadata_Widefield_vs_Confocal",
            "Metadata_Excitation_Type",
            "Metadata_Objective_NA",
            "Metadata_N_Brightfield_Planes_Min",
            "Metadata_N_Brightfield_Planes_Max",
            "Metadata_Distance_Between_Z_Microns",
            "Metadata_Filter_Configuration",
        ]
    )

    # print(f"Saving total load data with metadata {load_data_with_metadata_out_dir} ...")
    # images_with_metadata.to_pickle(load_data_with_metadata_out_file, compression="gzip")

    print(
        f"Saving total load data with metadata to parquet partitionned by source {load_data_with_metadata_out_dir} ..."
    )
    Path(load_data_with_metadata_out_dir).parent.mkdir(parents=True, exist_ok=True)
    images_with_metadata.to_parquet(
        load_data_with_metadata_out_dir,
        index=False,
        compression="snappy",
        engine="pyarrow",
        partition_cols=["Metadata_Source"],
    )

    size_of_folder = get_size_of_folder(load_data_with_metadata_out_dir)
    print(f"Size of {load_data_with_metadata_out_dir}: {size_of_folder / 1e6:.0f} MB")

    return images_with_metadata


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Create total load data file for all plates in the JUMP dataset."""
    metadata_dir = cfg.output_dirs.metadata_dir
    load_data_dir = cfg.output_dirs.load_data_dir
    tmp_load_data_dir = os.path.join(load_data_dir, "tmp")

    plate_metadata_file = os.path.join(metadata_dir, "plate.csv")
    complete_metadata_csv_file = os.path.join(metadata_dir, "complete_metadata.csv")
    total_load_data_csv_file = os.path.join(load_data_dir, "total_load_data.csv.gz")
    load_data_with_metadata_out_dir = os.path.join(load_data_dir, "load_data_with_metadata")

    download_class = DownloadLoadData(
        plate_metadata_file=plate_metadata_file,
        load_data_dir=load_data_dir,
        tmp_load_data_dir=tmp_load_data_dir,
        remove_tmp=True,
        channels=cfg.processing.channels,
        max_workers=cfg.run.max_workers,
        force=cfg.run.force,
    )

    # Download load data files
    print("=== Downloading load data files...")
    download_class.download_timed()

    # # Merge with metadata
    print("\n=== Merging load data with metadata...")
    get_load_data_with_metadata(
        total_load_data_csv_file=total_load_data_csv_file,
        complete_metadata_csv_file=complete_metadata_csv_file,
        load_data_with_metadata_out_dir=load_data_with_metadata_out_dir,
    )


if __name__ == "__main__":
    main()
