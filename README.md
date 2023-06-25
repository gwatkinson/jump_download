# JUMP Cell Painting dataset download

This is a repository aimed at downloading the JUMP Cell Painting dataset, which is a large dataset of molecule/image pairs obtained using High Content Screening, and can be used to predict molecular properties from images, and vice versa.
The data can then be used to train downstream deep learning models.

The data follows the Cell Painting protocol, a common effort between multiple labs that contains more than 116k tested compound for a equivalent of more that 3M images.

This repository was created once the code was cleaned, the entire Git history is still available.

## Citation

We used the JUMP Cell Painting datasets (Chandrasekaran et al., 2023), available from the Cell Painting Gallery on the Registry of Open Data on AWS (https://registry.opendata.aws/cellpainting-gallery/).

> Chandrasekaran, S. N., Ackerman, J., Alix, E., Ando, D. M., Arevalo, J., Bennion, M., ... & Carpenter, A. E. (2023).
> JUMP Cell Painting dataset: morphological impact of 136,000 chemical and genetic perturbations. bioRxiv, 2023-03: 2023-03.
> doi:10.1101/2023.03.23.534023

## Installation

Clone the code from GitHub:

```bash
git clone https://github.com/gwatkinson/jump_download.git
cd jump_download
```

Use [Poetry](https://python-poetry.org/docs/#installation) to install the Python dependencies (via pip). This command creates an environment in a default location (in `~/.cache/pypoetry/virtualenvs/`). You can create and activate an environment, poetry will then install the dependencies in that environment:

```bash
poetry install --without dev            # Install the dependencies

POETRY_ENV=$(poetry env info --path)    # Get the path of the environment
source "$POETRY_ENV/bin/activate"       # Activate the environment
```

## Setup the metadata

First, the metadata for the JUMP cpg0016 dataset can be found on [github.com/jump-cellpainting/datasets](https://github.com/jump-cellpainting/datasets/tree/main/metadata). The script `download_metadata` (defined with poetry) gets the metadata and does some light processing (merge, ...):

```bash
download_metadata  # -h to see help
```

The main parameters that can be changed in the `constants.py` file are:

```python
METADATA_DIR = os.path.join(DATA_ROOT_DIR, "metadata")
METADATA_DOWNLOAD_SCRIPT = os.path.join(
    PACKAGE_DIR, "mice/download/metadata/download_jump_metadata.sh"
)
```

with:

```python
PACKAGE_DIR = "/workspaces/biocomp/watkinso/mice/"  # The directory where the mice package is located
DATA_ROOT_DIR = "/projects/cpjump1/jump/"           # The directory where the metadata should be located, this also contains some images
```

## Get the "load data" files from S3

Then, we need the load data files from the [S3 bucket](https://registry.opendata.aws/cellpainting-gallery/) (see [here](https://cellpainting-gallery.s3.amazonaws.com/index.html#cpg0016-jump/) to explore the bucket).

Those file contains the urls to download the images on S3 and are used to make the link with the metadata (merging by source, batch, plate and well).

This downloads all the load data files (around 216MB compressed but 11GB to download) and does some processing on it (merge the metadata, etc...)
resulting in a directory containing parquet files called `/projects/cpjump1/jump/load_data/load_data_with_metadata` by default
(can be changed with the options or in the [`constants.py`](https://github.com/gwatkinson/mice/blob/main/mice/download/constants.py) file):

```bash
download_load_data_files
```

Then, to read the entire load data dataframe, you can use:

```python
import pandas as pd
from jump_download.constants import LOAD_DATA_WITH_METADATA_DIR

load_data_with_metadata = pd.read_parquet(LOAD_DATA_WITH_METADATA_DIR)
```

The main parameters that can be changed are:

```python
LOAD_DATA_DIR = os.path.join(DATA_ROOT_DIR, "load_data")  # Constant
TMP_LOAD_DATA_DIR = os.path.join(DATA_ROOT_DIR, "load_data", "tmp")  # Temporary, removed after the download by default
```

## Create the job files

This is the next step, where we create the csv that are used to run the jobs on condor. This step also samples observations given rules set in the `constants.py` file.

Run the following command to create the job files:

```bash
create_job_split
```

This script uses the `load_data_with_metadata` dataframe to create the job files.
It samples a number of images per well (defined in the `constants.py` file) and applies source and plate level filters as well.

The output is a directory called `/projects/cpjump1/jump/jobs` by default.
It contains a `ids` folder with around 2100 csv file that are all equivalent to one plate.
They have a similar format to the `load_data_with_metadata` dataframe but with only the selected observations, and with the additional columns:

* `output_dir`: The directory where the images will be downloaded for that job
* `tmp_output_dir`: The directory where the images will be downloaded temporarily for that job (then removed on the fly)
* `filter`: Whether the observation was filtered out or not (this should always be `True` as only the selected observations are kept)
* `job_id`: The id of the job (equivalent to the name of the csv file), the form is `{source}__{batch}__{plate}`

The function giving the `output_dir`, `tmp_output_dir` and `job_id` are **hard coded** in the [`sample_load_data.py`](https://github.com/gwatkinson/mice/blob/main/mice/download/create_jobs/sample_load_data.py) file. To modify the output paths you should change the functions [`create_output_dir_column`](https://github.com/gwatkinson/mice/blob/main/mice/download/create_jobs/sample_load_data.py#L288), [`create_tmp_dir_column`](https://github.com/gwatkinson/mice/blob/main/mice/download/create_jobs/sample_load_data.py#L295) and [`create_job_id_column`](https://github.com/gwatkinson/mice/blob/main/mice/download/create_jobs/sample_load_data.py#L302).

The main parameters that can be changed are:

```python
LOAD_DATA_DIR = os.path.join(DATA_ROOT_DIR, "load_data")
JOB_OUTPUT_DIR = os.path.join(DATA_ROOT_DIR, "jobs")

SOURCE_TO_DISK = {
    "source_13": "/projects/cpjump1/jump/images",  # CRISPR
    "source_4": "/projects/cpjump1/jump/images",  # ORF
    "source_1": "/projects/cpjump1/jump/images",  # Large plates
    "source_9": "/projects/cpjump1/jump/images",  # cpjump1 approx 2.41 TB
    "source_3": "/projects/cpjump2/jump/images",  # "Standard" plates
    "source_5": "/projects/cpjump2/jump/images",  # /projects/cpjump2 approx 2.61 TB
    "source_8": "/projects/cpjump2/jump/images",
    "source_11": "/projects/cpjump2/jump/images",
    "source_2": "/projects/cpjump3/jump/images",  # 6 views plates
    "source_10": "/projects/cpjump3/jump/images",  # 6 views plates
    "source_6": "/projects/cpjump3/jump/images",  # Large source, cpjump3 approx 2.37 TB
    "source_7": "dropped",  # Dropped
}  # Total approx: 7.39 TB

SOURCE_TO_TMP_IMAGE_FOLDER = {
    "source_13": "/projects/cpjump1/jump/tmp_images",
    "source_4": "/projects/cpjump1/jump/tmp_images",
    "source_1": "/projects/cpjump1/jump/tmp_images",
    "source_9": "/projects/cpjump1/jump/tmp_images",
    "source_3": "/projects/cpjump2/jump/tmp_images",
    "source_5": "/projects/cpjump2/jump/tmp_images",
    "source_8": "/projects/cpjump2/jump/tmp_images",
    "source_11": "/projects/cpjump2/jump/tmp_images",
    "source_2": "/projects/cpjump3/jump/tmp_images",
    "source_10": "/projects/cpjump3/jump/tmp_images",
    "source_6": "/projects/cpjump3/jump/tmp_images",
    "source_7": "dropped",
}

create_output_dir_column()  # Function to change the output dir, uses the SOURCE_TO_DISK dict by default

create_tmp_dir_column()     # Function to change the tmp dir, uses the SOURCE_TO_TMP_IMAGE_FOLDER dict by default

PLATE_TYPES_TO_KEEP = ["COMPOUND", "ORF", "CRISPR", "TARGET1", "TARGET2"]   # Plate types to keep
SOURCES_TO_EXCLUDE = ["source_7"]                                           # Sources to exclude

# For compound plates
NUMBER_OF_POSCON_TO_KEEP_PER_WELL = 4   # Number of positive controls to keep per well for the compound plates
NUMBER_OF_NEGCON_TO_KEEP_PER_WELL = 3   # Number of negative controls to keep per well
NUMBER_OF_TRT_TO_KEEP_PER_WELL = 6      # Number of treatments to keep per well

# For ORF plates
ORF_NUMBER_OF_POSCON_TO_KEEP_PER_WELL = 4
ORF_NUMBER_OF_NEGCON_TO_KEEP_PER_WELL = 3
ORF_NUMBER_OF_TRT_TO_KEEP_PER_WELL = 6

# For CRISPR plates
CRISPR_NUMBER_OF_POSCON_TO_KEEP_PER_WELL = 4
CRISPR_NUMBER_OF_NEGCON_TO_KEEP_PER_WELL = 3
CRISPR_NUMBER_OF_TRT_TO_KEEP_PER_WELL = 6

# For TARGET plates
TARGET_NUMBER_OF_POSCON_TO_KEEP_PER_WELL = 4
TARGET_NUMBER_OF_NEGCON_TO_KEEP_PER_WELL = 3
TARGET_NUMBER_OF_TRT_TO_KEEP_PER_WELL = 6
```

The resulting folder structure is:

```bash
* load_data/jobs/
  * ids/
    * {plate}__{batch}__{source}.csv.gz  # The job csv files, one per plate, and the dropped images are all in the same file per source
  * submission.csv                       # The csv file that contain the path to all the job csv files that should not be dropped
```

The original load_data file is split into many subfiles in order to use them with Condor. If you don't need that you can just use the `load_data_with_samples` parquet directory directly, which contains the same information.

## Download the images

Finally, the images can be downloaded using the following command:

```bash
download_images_from_csv --job_csv_path PATH
```

Usage:

```bash
Usage: download_images_from_csv [OPTIONS]

  Download images from the S3 bucket given a scenario and a load data file.

  Args:
    job_csv_path (str): Path to the job.csv.gz file defined in the previous step
    load_data_out_dir (str): Path to the directory containing the temporary images
    max_workers (int): Maximum number of workers to use
    force (bool): Whether to run the job even if the output file already exists
    percentile (float): Percentile to use for the scaling function
    min_resolution_x (int): Minimum resolution in the x direction for the cropping function
    min_resolution_y (int): Minimum resolution in the y direction for the cropping function
    channels (list): Channels to download
    bucket_name (str): Name of the bucket to download from

Options:
  --job_csv_path PATH             Path to the job.csv.gz file  [required]
  --load_data_out_dir TEXT        Path to the directory containing the
                                  load_data output
                                  [default: /projects/cpjump1/jump/load_data]
  --max_workers INTEGER           Maximum number of workers to use on this single plate / job
                                  [default: 8]
  --force / --no_force            Force overwrite of existing files
                                  [default: no_force]
  --percentile FLOAT              Percentile to use for the scaling function
                                  [default: 1.0]
  --min_resolution_x INTEGER      Minimum resolution in the x direction
                                  [default: 768]
  --min_resolution_y INTEGER      Minimum resolution in the y direction
                                  [default: 768]
  --channels TEXT                 Channels to download
                                  [default: DNA, AGP, ER, Mito, RNA]
  --bucket_name TEXT              Name of the bucket to download from
                                  [default: cellpainting-gallery]
  -h, --help                      Show this message and exit.
```

By default, the command uses the values defined in the `constants.py` file. The important parameters are:

```python
LOAD_DATA_DIR = os.path.join(DATA_ROOT_DIR, "load_data")
MAX_WORKERS = 8  # The maximum number of workers to use for multiprocessing
PERCENTILE = 1.0  # The percentile to use to normalize the images (1.0 = 99th percentile)
MIN_X = 768  # The crop size in the x and y directions. Note that the crop is centered, and if this value is larger than the image size,
MIN_Y = 768  # the image is not cropped. To keep the original size, you can set this value to 2000 (bigger than the largest image size)
```

This script uses the [`Robust8BitCropPNGScenario`](https://github.com/gwatkinson/mice/blob/main/mice/download/images/final_image_class.py#128).
It can be modified to use different parameters, or to use a different scenario.

The `--remove_tmp` flag is important, as it allows to remove the raw file after the images have been processed.

More details about the classes used in this process in the [README of the module](https://github.com/gwatkinson/mice/blob/main/mice/download/images/README.md).

## Run the jobs on the cluster with Condor

This is the last step of the pipeline. It uses the `download_plate` command to download the images from the S3 bucket.

It consists only of a single `.sub` file that tells how much ressources should be used, and how many jobs should be run in parallel on different nodes.

```bash
condor_submit mice/download/condor/download_plate.sub
```