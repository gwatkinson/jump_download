import os
from io import BytesIO
from pathlib import Path

import boto3
import hydra
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from omegaconf import DictConfig
from PIL import Image

from jump_download.base_class import BaseDownload
from jump_download.utils import (
    apply_dtypes_with_large_dict,
    crop_min_resolution,
    robust_convert_to_8bit,
)


def initialise_client():
    global s3_client
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))


class GetRawImages(BaseDownload):
    """Class that get the raw images from S3 and store them in a buffer.

    This should not be used as is, since the resulting images are too large.
    """

    def __init__(
        self,
        load_data_df,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(max_workers=max_workers)

        initialise_client()

        self.load_data_df = load_data_df
        self.channels = channels
        self.bucket_name = bucket_name
        self.force = force

    def get_job_list(self):
        jobs = []
        for _, row in self.load_data_df.iterrows():
            source = row["Metadata_Source"]
            batch = row["Metadata_Batch"]
            plate = row["Metadata_Plate"]
            well = row["Metadata_Well"]
            site = row["Metadata_Site"]

            job = {
                "source": source,
                "batch": batch,
                "plate": plate,
                "well": well,
                "site": site,
            }

            for channel in self.channels:
                filename = row[f"FileName_Orig{channel}"]
                s3_path = row[f"PathName_Orig{channel}"].replace(f"s3://{self.bucket_name}/", "")
                s3_filename = os.path.join(s3_path, filename)

                buffer = BytesIO()

                job[channel] = {
                    "channel": channel,
                    "s3_filename": s3_filename,
                    "bucket_name": self.bucket_name,
                    "buffer": buffer,
                }

            # 5 channel (files) per job
            jobs.append(job)

        return jobs

    def execute_job(self, job):
        """Execute a single image job that consists of downloading the images from S3.

        This is the main difference between the scenarios.
        """
        result = []
        for channel in self.channels:
            sub_job = job[channel]
            s3_client.download_fileobj(
                Bucket=sub_job["bucket_name"],
                Key=sub_job["s3_filename"],
                Fileobj=sub_job["buffer"],
            )
            result.append("Success")

        if all(r == "Success" for r in result):
            short_result = "Success"
        else:
            short_result = "Failed"

        return short_result

    def post_process(self, download_results):
        return download_results


class Robust8BitCropPNGScenario(GetRawImages):
    def __init__(
        self,
        load_data_df,
        out_dir,
        out_df_path,
        max_workers,
        force,
        percentile,
        min_resolution_x,
        min_resolution_y,
        channels,
        bucket_name,
    ) -> None:
        """Download the images and process them on the fly using 8 bit, cropping and PNG compression.

        Args:
            load_data_df (pd.DataFrame): The dataframe with the metadata to download.
            out_dir (str): The output directory where the images will be saved.
            out_df_path (str, optional): The path to save the dataframe with the downloaded images in parquet.
                Defaults to None.
            percentile (float, optional): The percentile to use for the scaling. Defaults to 1.0.
            min_resolution_x (int, optional): The minimum resolution in the x axis. Defaults to 768.
            min_resolution_y (int, optional): The minimum resolution in the y axis. Defaults to 768.
            channels (list, optional): The list of channels to download. Defaults to CHANNELS.
            bucket_name (str, optional): The name of the bucket to download the images from. Defaults to BUCKET_NAME.
            force (bool, optional): Whether to force the download of the images. Defaults to False.
            max_workers (int, optional): The maximum number of workers to use. Defaults to 4.

        Methods:
            get_job_list: Get the list of jobs to execute.
            image_processing: How to process a single image from an array.
            process_image_from_job: Load the images from the jobs and apply the image_processing to the loaded arrays.
            get_output_file: Get the output file for a job. And create the parent directory if it does not exist.
            job_processing: Process a single job using the previous methods and save the outputs.
            execute_job: Execute a single job and return the result.
                Uses the method from the parent class and job_processing.
                This function is called by other functions defined in the parent class.
            post_process: Post process the results from the jobs.
                This function is called by other functions defined in the parent class.

        Parent class:
            DownloadRawImages: Download the raw images from the bucket and save them to disk. Inherit from BaseDownload.

        Base class methods:
            download_with_multiprocessing_generator: Create the job generator using multiprocessing.
                This uses the execute_job method from the child class.
            download_objects_from_jobs: Run the jobs using the job generator with multiprocessing.
            download: Execute the methods get_job_list, download_objects_from_jobs and post_process.
            download_timed: Execute the download method and time it. Returns (download_results, exec_time).
        """
        super().__init__(
            load_data_df=load_data_df,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )

        if not s3_client:
            initialise_client()

        self.percentile = percentile
        self.min_resolution_x = min_resolution_x
        self.min_resolution_y = min_resolution_y
        self.out_dir = out_dir
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.out_df_path = out_df_path
        Path(self.out_df_path).parent.mkdir(parents=True, exist_ok=True)

        self.output_format = "{source}__{plate}__{well}__{site}__{channel}.png"

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(
    load_data_df     = DataFrame {self.load_data_df.shape},
    out_dir          = {self.out_dir},
    out_df_path      = {self.out_df_path},
    percentile       = {self.percentile},
    max_workers      = {self.max_workers},
    force            = {self.force},
    min_resolution_x = {self.min_resolution_x},
    min_resolution_y = {self.min_resolution_y},
    channels         = {self.channels},
    bucket_name      = {self.bucket_name},
)"""

    def get_job_list(self):
        jobs = super().get_job_list()
        for job in jobs:
            source = job["source"]
            batch = job["batch"]
            plate = job["plate"]
            well = job["well"]
            site = job["site"]

            for channel in self.channels:
                filename = self.output_format.format(
                    source=source, plate=plate, well=well, site=site, channel=channel
                )
                job[channel]["channel_output"] = os.path.join(
                    self.out_dir, source, batch, plate, filename
                )

            job["skip"] = not self.force and all(
                os.path.exists(job[channel]["channel_output"]) for channel in self.channels
            )

        print(f"Skipping {sum(job['skip'] for job in jobs)}/{len(jobs)} jobs")

        return jobs

    def image_processing(self, img_arr):
        cropped_img = crop_min_resolution(
            img_arr, min_resolution_x=self.min_resolution_x, min_resolution_y=self.min_resolution_y
        )
        processed_img = robust_convert_to_8bit(cropped_img, percentile=self.percentile)
        return processed_img

    def process_image_from_job(self, job_with_channels):
        """Load the images from the jobs and apply the process_func to the loaded numpy arrays."""
        images = []
        for channel in self.channels:
            buffer = job_with_channels[channel]["buffer"]
            img_arr = np.array(Image.open(buffer))
            img_arr = self.image_processing(img_arr)
            images.append(img_arr)
        return images

    def get_output_files(self, job):
        output_files = []

        for channel in self.channels:
            output = job[channel]["channel_output"]
            output_files.append(output)
            Path(output).parent.mkdir(parents=True, exist_ok=True)

        return output_files

    def job_processing(self, job):
        # Load the images and apply the simple scaling function
        images = self.process_image_from_job(job)

        # Save the images
        output_files = self.get_output_files(job)
        for output, img_arr in zip(output_files, images):
            img = Image.fromarray(img_arr, mode="L")
            img.save(output)

        return "Success"

    def execute_job(self, job):
        try:
            if job["skip"]:
                return "Skipped"

            super().execute_job(job)

            result = self.job_processing(job)

        except Exception as e:
            result = f"Failed: {str(e)}"

        return result

    def post_process(self, download_results):
        image_df = pd.DataFrame(
            [
                {
                    "Metadata_Source": job["source"],
                    "Metadata_Batch": job["batch"],
                    "Metadata_Plate": job["plate"],
                    "Metadata_Well": job["well"],
                    "Metadata_Site": job["site"],
                    "channel": f"FileName_Orig{channel}",
                    "output": job[channel]["channel_output"],
                    "result": job["result"],
                }
                for job in download_results
                for channel in self.channels
            ]
        )

        if (image_df["result"].isin(["Success", "Skipped"])).all():
            print("All images downloaded successfully")
        else:
            to_display = image_df[~image_df["result"].isin(["Success", "Skipped"])].drop(
                columns="Metadata_Batch"
            )
            unique_errors = to_display["result"].unique().tolist()
            raise Exception(
                "Not all images were downloaded successfully:\n"
                f"\n{to_display.shape[0]} errors encountered\n"
                f"\n Error messages {unique_errors}\n"
                f"\n{to_display.head(10).to_string(index=False)}"
            )

        pivot_image_df = image_df.pivot(
            index=[
                "Metadata_Source",
                "Metadata_Batch",
                "Metadata_Plate",
                "Metadata_Well",
                "Metadata_Site",
            ],
            columns="channel",
            values="output",
        )
        clean_image = pivot_image_df.rename_axis(None, axis=1).reset_index()

        clean_image.to_parquet(
            path=self.out_df_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        return clean_image


def download_images_from_dataframe(
    job_csv_path,
    load_data_out_dir,
    percentile,
    min_resolution_x,
    min_resolution_y,
    max_workers,
    force,
    job_dtypes,
    channels,
    bucket_name,
):
    """Download images from the S3 bucket given a scenario and a load data file.

    Args:
        job_csv_path (str): Path to the job.csv.gz file
        tmp_out_dir (str): Path to the temporary image directory
        load_data_out_dir (str): Path to the directory containing the temporary images
        percentile (float): Percentile to use for the scaling function
        min_resolution_x (int): Minimum resolution in the x direction for the cropping function
        min_resolution_y (int): Minimum resolution in the y direction for the cropping function
        max_workers (int): Maximum number of workers to use
        force (bool): Whether to run the job even if the output file already exists
        channels (list): Channels to download
        bucket_name (str): Name of the bucket to download from
    """
    cols_to_keep = [
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
        "FileName_OrigDNA",
        "FileName_OrigAGP",
        "FileName_OrigER",
        "FileName_OrigMito",
        "FileName_OrigRNA",
        "PathName_OrigDNA",
        "PathName_OrigAGP",
        "PathName_OrigER",
        "PathName_OrigMito",
        "PathName_OrigRNA",
        "filter",
        "output_dir",
        "job_id",
    ]

    print(f"\n=== Currently in {os.getcwd()} ===\n")
    print(f"\n=== Target csv file {job_csv_path} exists: {os.path.exists(job_csv_path)} ===\n")

    print(f"\n=== Loading {job_csv_path} ===\n")
    try:
        job_df = pd.read_csv(job_csv_path, usecols=cols_to_keep)
        job_df = apply_dtypes_with_large_dict(job_df, job_dtypes)

        job_id = job_df["job_id"].unique()[0]
        out_df_path = os.path.join(load_data_out_dir, "final", f"{job_id}.parquet")

        assert len(job_df["job_id"].unique()) == 1, f"Multiple job IDs in csv {job_csv_path}"
        assert (
            job_id.split("__")[-1] != "dropped"
        ), f"Job ID {job_id} says it should be dropped {job_csv_path}"

    except Exception:
        print(f"=== {job_csv_path} could not be loaded as CSV, trying parquet ===")
        job_df = pd.read_parquet(job_csv_path, columns=cols_to_keep)
        job_df = apply_dtypes_with_large_dict(job_df, job_dtypes)
        out_df_path = os.path.join(load_data_out_dir, "final", "total.parquet")

    filter = job_df["filter"].unique()[0]
    output_dir = job_df["output_dir"].unique()[0]
    assert filter, f"All images are dropped in this csv {job_csv_path}"
    assert (
        len(job_df["filter"].unique()) == 1
    ), f"Some images to drop and some to keep in csv {job_csv_path}"
    assert (
        len(job_df["output_dir"].unique()) == 1
    ), f"Multiple output directories in csv {job_csv_path}"

    print(f"\n=== Downloading images into {output_dir} ===\n")
    download_class = Robust8BitCropPNGScenario(
        load_data_df=job_df,
        out_dir=output_dir,
        out_df_path=out_df_path,
        max_workers=max_workers,
        force=force,
        percentile=percentile,
        min_resolution_x=min_resolution_x,
        min_resolution_y=min_resolution_y,
        channels=channels,
        bucket_name=bucket_name,
    )

    download_class.download_timed()

    print(f"\n=== Resulting metadata file in {out_df_path} ===\n")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Test docstring"""
    download_images_from_dataframe(
        job_csv_path=cfg.run.job_csv_path,
        load_data_out_dir=cfg.output_dirs.load_data_dir,
        percentile=cfg.processing.percentile,
        min_resolution_x=cfg.processing.min_x,
        min_resolution_y=cfg.processing.min_y,
        max_workers=cfg.run.max_workers,
        force=cfg.run.force,
        job_dtypes=cfg.output_dirs.col_dtypes,
        channels=cfg.processing.channels,
        bucket_name=cfg.processing.bucket_name,
    )


if __name__ == "__main__":
    main()
