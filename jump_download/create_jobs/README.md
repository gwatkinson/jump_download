# Create the job files

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

## Usage

```bash
Usage: create_job_split [OPTIONS]

Options:
  --load_data_dir TEXT   Directory to temporary save load data files to.
                         [default: /projects/cpjump1/jump/load_data]
  --job-output-dir TEXT  [default: /projects/cpjump1/jump/jobs]
  -h, --help             Show this message and exit.
```
