"""OLD module containing different scenarios.

This is kept for reference, as it contains many different scenarios that were tested.
Use the `final_image_class` module instead.
"""

import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

import boto3
import click
import h5py
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image
from tqdm.auto import tqdm

from jump_download.base_class import BaseDownload
from jump_download.utils import (
    crop_min_resolution,
    get_otsu_threshold,
    robust_convert_to_8bit,
    simple_convert_to_8bit,
)


def initialise_client():
    global s3_client
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))


class DownloadRawImages(BaseDownload):
    """Class that downloads the raw images from S3.

    This should not be used as is, since the resulting images are too large.
    """

    def __init__(
        self,
        load_data_df,
        tmp_out_dir,
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

        self.tmp_out_dir = tmp_out_dir
        Path(self.tmp_out_dir).mkdir(parents=True, exist_ok=True)

    def get_job_list(self):
        skip_count = 0
        jobs = []
        for _, row in tqdm(self.load_data_df.iterrows(), total=len(self.load_data_df)):
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

                output = os.path.join(
                    self.tmp_out_dir, str(source), str(batch), str(plate), str(filename)
                )

                skip = not self.force and os.path.exists(output)

                if skip:
                    skip_count += 1

                job[channel] = {
                    "channel": channel,
                    "s3_filename": s3_filename,
                    "bucket_name": self.bucket_name,
                    "tmp_output": output,
                    "skip": skip,
                }

            # 5 channel (files) per job
            jobs.append(job)

        print(f"Skipping {skip_count}/{len(jobs) * len(self.channels)} downloads")

        return jobs

    def execute_job(self, job):
        """Execute a single image job that consists of downloading the images from S3.

        This is the main difference between the scenarios.
        """
        result = []
        for channel in self.channels:
            sub_job = job[channel]
            if sub_job["skip"]:
                result.append("Skipped")
                continue

            s3_filename = sub_job["s3_filename"]
            bucket_name = sub_job["bucket_name"]
            output = sub_job["tmp_output"]

            Path(output).parent.mkdir(parents=True, exist_ok=True)

            s3_client.download_file(
                bucket_name,
                s3_filename,
                output,
            )
            result.append("Success")

        if all(r == "Skipped" for r in result):
            short_result = "Skipped"
        elif all(r == "Success" or r == "Skipped" for r in result):
            short_result = "Success"
        else:
            short_result = "Failed"

        return short_result

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
                    "output": job[channel]["tmp_output"],
                    "result": job["result"],
                }
                for job in download_results
                for channel in self.channels
            ]
        )

        if (image_df["result"].isin(["Success", "Skipped"])).all():
            print("All images processed successfully")
        else:
            raise Exception("Not all images were processed successfully")

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
        clean_image = pivot_image_df.rename_axis(None, axis=1)

        output = Path(self.tmp_out_dir) / "downloaded_raw_images.csv.gz"
        clean_image.to_csv(output, compression="gzip")

        return clean_image


class DownloadWithJobProcessing(DownloadRawImages):
    """Meta class that applies processing to the output of the DownloadRawImages class.

    This is the main class used in the different scenarios, as it allows to try different processing steps.

    Methods:
        job_processing: Abstract method that should be implemented in the child classes.
            It takes a job (a dict) after the execution of the DownloadRawImages job and should return a result (a string).
            The result is then added to the job dict and returned by the execute_job method.

    Help:
        The download process, used when download method is called, are the following steps:
            1. Init the DownloadRawImages class, with a tmp_out_dir
            2. Create the out_dir
            3. Get the job list from the get_job_list method
            4. Execute the jobs in parallel, using the execute_job and the job_processing methods.
                This calls the DownloadRawImages execute_job method, which downloads the images from S3 and puts them in
                a temporary directory.
                The job_processing method is then called, which should process the images in the temporary directory.
                It can apply transformations, etc. then save the results in the out_dir.
                It can also remove the images from the temporary directory on the fly.
            5. Post process the results with the post_process method
                This removes the temporary directory by default.
    """

    __metaclass__ = ABCMeta

    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            tmp_out_dir=tmp_out_dir,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )

        self.remove_tmp = remove_tmp
        self.out_dir = out_dir
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.output_format = "{plate}-{well}-{site}.npz"

    def get_job_list(self):
        jobs = super().get_job_list()
        for job in jobs:
            source = job["source"]
            batch = job["batch"]
            plate = job["plate"]
            well = job["well"]
            site = job["site"]

            job["output"] = os.path.join(
                self.out_dir,
                source,
                batch,
                plate,
                self.output_format.format(plate=plate, well=well, site=site),
            )
            job["skip"] = not self.force and os.path.exists(job["output"])

        return jobs

    def job_processing(self, job):
        # Load the images and apply the simple scaling function
        images = self.process_image_from_job(job)
        channels_arr = np.array(self.channels)

        # Save the images
        output = self.get_output_file(job)
        np.savez_compressed(output, images=images, channels=channels_arr)

        # Remove the temporary files
        if self.remove_tmp:
            self.remove_tmp_files(job)

        return "Success"

    def execute_job(self, job):
        if job["skip"]:
            return "Skipped"

        super().execute_job(job)

        try:
            result = self.job_processing(job)
        except Exception as e:
            result = f"Failed: {str(e)}"

        return result

    def post_process(self, download_results):
        # Create a dataframe with the results for each 5 channel image
        result_df = pd.DataFrame(
            [
                {
                    "Metadata_Source": job["source"],
                    "Metadata_Batch": job["batch"],
                    "Metadata_Plate": job["plate"],
                    "Metadata_Well": job["well"],
                    "Metadata_Site": job["site"],
                    "FileName": job["output"],
                    "result": job["result"],
                }
                for job in download_results
            ]
        )

        if (result_df["result"].isin(["Success", "Skipped"])).all():
            print("All images downloaded successfully")
        else:
            print("Not all images were downloaded successfully")
            return result_df

        clean_image = result_df.drop(columns=["result"])

        output = Path(self.out_dir) / "downloaded_images.csv.gz"
        clean_image.to_csv(output, compression="gzip")

        # Remove the tmp folder
        # if self.remove_tmp:
        #     shutil.rmtree(self.tmp_out_dir, ignore_errors=True)

        return clean_image

    def remove_tmp_files(self, job_with_channels):
        for channel in self.channels:
            input = job_with_channels[channel]["tmp_output"]
            os.remove(input)

    @abstractmethod
    def image_processing(self, img_arr):
        return img_arr

    def process_image_from_job(self, job_with_channels):
        """Load the images from the jobs and apply the process_func to the loaded numpy arrays."""
        images = []
        for channel in self.channels:
            tmp_file = job_with_channels[channel]["tmp_output"]
            img_arr = np.array(Image.open(tmp_file))
            img_arr = self.image_processing(img_arr)
            images.append(img_arr)
        return images

    def get_output_file(self, job):
        output = job["output"]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        return output


class PNGCompressionScenario(DownloadRawImages):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        out_df_path,
        percentile,
        min_resolution_x,
        min_resolution_y,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        """Download the images and process them on the fly using 8 bit, cropping and PNG compression.

        Args:
            load_data_df (pd.DataFrame): The dataframe with the metadata to download.
            out_dir (str): The output directory where the images will be saved.
            tmp_out_dir (str): The temporary output directory where the images will be saved.
            out_df_path (str, optional): The path to save the dataframe with the downloaded images in parquet.
                Defaults to None.
            percentile (float, optional): The percentile to use for the scaling. Defaults to 1.0.
            min_resolution_x (int, optional): The minimum resolution in the x axis. Defaults to 768.
            min_resolution_y (int, optional): The minimum resolution in the y axis. Defaults to 768.
            remove_tmp (bool, optional): Whether to remove the temporary files. Defaults to True.
            channels (list, optional): The list of channels to download. Defaults to CHANNELS.
            bucket_name (str, optional): The name of the bucket to download the images from. Defaults to BUCKET_NAME.
            force (bool, optional): Whether to force the download of the images. Defaults to False.
            max_workers (int, optional): The maximum number of workers to use. Defaults to 4.

        Methods:
            get_job_list: Get the list of jobs to execute.
            image_processing: How to process a single image from an array.
            process_image_from_job: Load the images from the jobs and apply the image_processing to the loaded arrays.
            get_output_file: Get the output file for a job. And create the parent directory if it does not exist.
            remove_tmp_files: Remove the temporary files for a job.
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
            tmp_out_dir=tmp_out_dir,
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
        self.remove_tmp = remove_tmp
        self.out_dir = out_dir
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.out_df_path = out_df_path
        Path(self.out_df_path).parent.mkdir(parents=True, exist_ok=True)

        self.output_format = "{source}__{plate}__{well}__{site}__{channel}.png"

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(
    load_data_df     = DataFrame {self.load_data_df.shape},
    out_dir          = {self.out_dir},
    tmp_out_dir      = {self.tmp_out_dir},
    out_df_path      = {self.out_df_path},
    percentile       = {self.percentile},
    min_resolution_x = {self.min_resolution_x},
    min_resolution_y = {self.min_resolution_y},
    remove_tmp       = {self.remove_tmp},
    channels         = {self.channels},
    bucket_name      = {self.bucket_name},
    force            = {self.force},
    max_workers      = {self.max_workers},
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
                job[channel]["channel_output"] = os.path.join(
                    self.out_dir,
                    source,
                    batch,
                    plate,
                    self.output_format.format(
                        source=source, plate=plate, well=well, site=site, channel=channel
                    ),
                )

            job["skip"] = not self.force and all(
                os.path.exists(job[channel]["channel_output"]) for channel in self.channels
            )

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
            tmp_file = job_with_channels[channel]["tmp_output"]
            img_arr = np.array(Image.open(tmp_file))
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

    def remove_tmp_files(self, job_with_channels):
        for channel in self.channels:
            input = job_with_channels[channel]["tmp_output"]
            os.remove(input)

    def job_processing(self, job):
        # Load the images and apply the simple scaling function
        images = self.process_image_from_job(job)

        # Save the images
        output_files = self.get_output_files(job)
        for output, img_arr in zip(output_files, images):
            img = Image.fromarray(img_arr, mode="L")
            img.save(output)

        # Remove the temporary files
        if self.remove_tmp:
            self.remove_tmp_files(job)

        return "Success"

    def execute_job(self, job):
        if job["skip"]:
            return "Skipped"

        super().execute_job(job)

        try:
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
            raise Exception(f"Not all images were downloaded successfully: \n{image_df.head(10)}")

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


class OnlyCompressionScenario(DownloadWithJobProcessing):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            out_dir=out_dir,
            tmp_out_dir=tmp_out_dir,
            remove_tmp=remove_tmp,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )

    def image_processing(self, img_arr):
        return super().image_processing(img_arr)


class SimpleDownsamplingScenario(DownloadWithJobProcessing):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            out_dir=out_dir,
            tmp_out_dir=tmp_out_dir,
            remove_tmp=remove_tmp,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )

    def image_processing(self, img_arr):
        return simple_convert_to_8bit(img_arr)


class RobustScalingScenario(DownloadWithJobProcessing):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        percentile,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            out_dir=out_dir,
            tmp_out_dir=tmp_out_dir,
            remove_tmp=remove_tmp,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )
        self.percentile = percentile

    def image_processing(self, image):
        return robust_convert_to_8bit(image, self.percentile)


class RobustScalingScenarioNotArray(DownloadWithJobProcessing):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        percentile,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            out_dir=out_dir,
            tmp_out_dir=tmp_out_dir,
            remove_tmp=remove_tmp,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )
        self.percentile = percentile

    def image_processing(self, img_arr):
        return robust_convert_to_8bit(img_arr, self.percentile)

    def job_processing(self, job):
        # Load the images and apply the robust scaling
        images = self.process_image_from_job(job)
        images = dict(zip(self.channels, images))

        # Save the output in dict format
        output = self.get_output_file(job)
        np.savez_compressed(output, **images)

        # Remove the temporary files
        if self.remove_tmp:
            self.remove_tmp_files(job)

        return "Success"


class CropMinResolutionScenario(DownloadWithJobProcessing):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        percentile,
        min_resolution_x,
        min_resolution_y,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            out_dir=out_dir,
            tmp_out_dir=tmp_out_dir,
            remove_tmp=remove_tmp,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )
        self.percentile = percentile
        self.min_resolution_x = min_resolution_x
        self.min_resolution_y = min_resolution_y

    def image_processing(self, img_arr):
        return robust_convert_to_8bit(
            crop_min_resolution(img_arr, self.min_resolution_x, self.min_resolution_y),
            self.percentile,
        )


class NoChannelInNpz(DownloadWithJobProcessing):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        percentile,
        min_resolution_x,
        min_resolution_y,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            out_dir=out_dir,
            tmp_out_dir=tmp_out_dir,
            remove_tmp=remove_tmp,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )
        self.percentile = percentile
        self.min_resolution_x = min_resolution_x
        self.min_resolution_y = min_resolution_y

    def image_processing(self, img_arr):
        return robust_convert_to_8bit(
            crop_min_resolution(img_arr, self.min_resolution_x, self.min_resolution_y),
            self.percentile,
        )

    def job_processing(self, job):
        # Load the images and apply the simple scaling function
        images = self.process_image_from_job(job)

        # Save the images
        output = self.get_output_file(job)
        np.savez_compressed(output, images=images)

        # Remove the temporary files
        if self.remove_tmp:
            self.remove_tmp_files(job)

        return "Success"


class OtsuThresholdScenario(DownloadWithJobProcessing):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        percentile,
        min_resolution_x,
        min_resolution_y,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            out_dir=out_dir,
            tmp_out_dir=tmp_out_dir,
            remove_tmp=remove_tmp,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )
        self.percentile = percentile
        self.min_resolution_x = min_resolution_x
        self.min_resolution_y = min_resolution_y

    def image_processing(self, img_arr):
        return robust_convert_to_8bit(
            crop_min_resolution(img_arr, self.min_resolution_x, self.min_resolution_y),
            self.percentile,
        )

    def job_processing(self, job):
        # Load the images and apply the simple scaling function
        images = self.process_image_from_job(job)
        channels_arr = np.array(self.channels)

        # Get the id of DNA in list
        dna_idx = self.channels.index("DNA")
        dna_img = images[dna_idx]

        # Calculate the Otsu Threshold on the DNA channel
        threshold, foreground_area = get_otsu_threshold(dna_img)
        otsu_np = np.array([threshold, foreground_area])

        # Save the images
        output = self.get_output_file(job)
        np.savez_compressed(output, images=images, channels=channels_arr, otsu=otsu_np)

        # Remove the temporary files
        if self.remove_tmp:
            self.remove_tmp_files(job)

        return "Success"


class H5CompressionScenario(DownloadWithJobProcessing):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        percentile,
        min_resolution_x,
        min_resolution_y,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            out_dir=out_dir,
            tmp_out_dir=tmp_out_dir,
            remove_tmp=remove_tmp,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )
        self.percentile = percentile
        self.min_resolution_x = min_resolution_x
        self.min_resolution_y = min_resolution_y
        self.output_format = "{plate}-{well}-{site}.hdf5"

    # def get_job_list(self):
    #     jobs = super().get_job_list()
    #     for job in jobs:
    #         source = job["source"]
    #         batch = job["batch"]
    #         plate = job["plate"]
    #         well = job["well"]
    #         site = job["site"]

    #         job["output"] = os.path.join(
    #             self.out_dir, source, batch, plate, f"{plate}-{well}-{site}.hdf5"
    #         )
    #         job["skip"] = not self.force and os.path.exists(job["output"])

    #     return jobs

    def image_processing(self, img_arr):
        return robust_convert_to_8bit(
            crop_min_resolution(img_arr, self.min_resolution_x, self.min_resolution_y),
            self.percentile,
        )

    def job_processing(self, job):
        # Load the images and apply the simple scaling function
        images = self.process_image_from_job(job)
        channels_arr = np.array(self.channels)

        # Save the images
        output = self.get_output_file(job)
        with h5py.File(output, "w") as f:
            f.create_dataset("images", data=images, compression="gzip", compression_opts=9)
            f.create_dataset("channels", data=channels_arr)

        # Remove the temporary files
        if self.remove_tmp:
            self.remove_tmp_files(job)

        return "Success"


class JPGCompressionScenario(PNGCompressionScenario):
    def __init__(
        self,
        load_data_df,
        out_dir,
        tmp_out_dir,
        percentile,
        min_resolution_x,
        min_resolution_y,
        remove_tmp,
        channels,
        bucket_name,
        force,
        max_workers,
    ) -> None:
        super().__init__(
            load_data_df=load_data_df,
            out_dir=out_dir,
            out_df_path=os.path.join(out_dir, "download_results.parquet"),
            tmp_out_dir=tmp_out_dir,
            percentile=percentile,
            min_resolution_x=min_resolution_x,
            min_resolution_y=min_resolution_y,
            remove_tmp=remove_tmp,
            channels=channels,
            bucket_name=bucket_name,
            force=force,
            max_workers=max_workers,
        )

        self.output_format = "{plate}-{well}-{site}-{channel}.jpg"

    def job_processing(self, job):
        # Load the images and apply the simple scaling function
        images = self.process_image_from_job(job)

        # Save the images
        output_files = self.get_output_files(job)
        for output, img_arr in zip(output_files, images):
            img = Image.fromarray(img_arr, mode="L")
            img.save(output, quality=100, subsampling=0)

        # Remove the temporary files
        if self.remove_tmp:
            self.remove_tmp_files(job)

        return "Success"


@click.command(
    "download_images",
    short_help="Download images from the S3 bucket given a scenario",
    context_settings={"show_default": True, "help_option_names": ("-h", "--help")},
)
@click.option(
    "--job_csv_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the job.csv.gz file",
)
@click.option(
    "--load_data_out_dir",
    help="Path to the directory containing the load_data output",
)
@click.option(
    "--percentile",
    type=float,
    help="Percentile to use for the scaling function",
)
@click.option(
    "--min_resolution_x",
    type=int,
    help="Minimum resolution in the x direction",
)
@click.option(
    "--min_resolution_y",
    type=int,
    help="Minimum resolution in the y direction",
)
@click.option(
    "--max_workers",
    type=int,
    help="Maximum number of workers to use",
)
@click.option(
    "--remove_tmp/--no_remove_tmp",
    default=True,
    help="Remove the temporary files",
)
@click.option(
    "--force/--no_force",
    default=False,
    help="Force overwrite of existing files",
)
@click.option(
    "--channels",
    multiple=True,
    help="Channels to download",
)
@click.option(
    "--bucket_name",
    type=str,
    help="Name of the bucket to download from",
)
def download_images(
    job_csv_path,
    load_data_out_dir,
    percentile,
    min_resolution_x,
    min_resolution_y,
    max_workers,
    remove_tmp,
    force,
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
        remove_tmp (bool): Remove the temporary files from the tmp_out_dir
        force (bool): Whether to run the job even if the output file already exists
        channels (list): Channels to download
        bucket_name (str): Name of the bucket to download from
    """
    cols_to_keep = {
        "Metadata_Source": object,
        "Metadata_Batch": object,
        "Metadata_Plate": object,
        "Metadata_Well": object,
        "Metadata_Site": np.int64,
        "FileName_OrigDNA": object,
        "FileName_OrigAGP": object,
        "FileName_OrigER": object,
        "FileName_OrigMito": object,
        "FileName_OrigRNA": object,
        "PathName_OrigDNA": object,
        "PathName_OrigAGP": object,
        "PathName_OrigER": object,
        "PathName_OrigMito": object,
        "PathName_OrigRNA": object,
        "filter": bool,
        "output_dir": object,
        "tmp_out_dir": object,
        "job_id": object,
    }

    print(f"\n=== Loading {job_csv_path} ===\n")
    job_df = pd.read_csv(job_csv_path, usecols=list(cols_to_keep), dtype=cols_to_keep)

    output_dir = job_df["output_dir"].unique()[0]
    tmp_out_dir = job_df["tmp_out_dir"].unique()[0]
    job_id = job_df["job_id"].unique()[0]
    filter = job_df["filter"].unique()[0]
    out_df_path = os.path.join(load_data_out_dir, "final", f"{job_id}.parquet")

    assert (
        len(job_df["output_dir"].unique()) == 1
    ), f"Multiple output directories in csv {job_csv_path}"
    assert (
        len(job_df["tmp_out_dir"].unique()) == 1
    ), f"Multiple temporary output directories in csv {job_csv_path}"
    assert len(job_df["job_id"].unique()) == 1, f"Multiple job IDs in csv {job_csv_path}"
    assert (
        len(job_df["filter"].unique()) == 1
    ), f"Some images to drop and some to keep in csv {job_csv_path}"
    assert filter, f"All images are dropped in this csv {job_csv_path}"
    assert (
        job_id.split("__")[-1] != "dropped"
    ), f"Job ID {job_id} says it should be dropped {job_csv_path}"

    print(f"\n=== Downloading images for {job_id} into {tmp_out_dir} and {output_dir} ===\n")
    download_class = PNGCompressionScenario(
        load_data_df=job_df,
        out_dir=output_dir,
        tmp_out_dir=tmp_out_dir,
        out_df_path=out_df_path,
        percentile=percentile,
        min_resolution_x=min_resolution_x,
        min_resolution_y=min_resolution_y,
        remove_tmp=remove_tmp,
        force=force,
        max_workers=max_workers,
        channels=channels,
        bucket_name=bucket_name,
    )

    (downloaded_images, _), _ = download_class.download_timed()

    print(f"\n=== Resulting metadata file in {out_df_path} ===\n")

    return downloaded_images


if __name__ == "__main__":
    download_images()
