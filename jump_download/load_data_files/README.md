# Get the "load data" files from S3

Then, we need the load data files from the [S3 bucket](https://registry.opendata.aws/cellpainting-gallery/) (see [here](https://cellpainting-gallery.s3.amazonaws.com/index.html#cpg0016-jump/) to explore the bucket).
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

## Usage

```bash
Usage: download_load_data_files [OPTIONS]

  Create total load data file.

Options:
  --load_data_dir TEXT            Directory to save load data files to.
                                  [default: /projects/cpjump1/jump/load_data]
  --tmp_load_data_dir TEXT        Directory to temporary save load data files
                                  to.  [default:
                                  /projects/cpjump1/jump/load_data/tmp]
  --metadata_dir TEXT             Path to plate metadata folder.  [default:
                                  /projects/cpjump1/jump/metadata]
  --remove_tmp / --no-remove_tmp  Whether to remove temporary files.
                                  [default: remove_tmp]
  --channels TEXT                 Channels to include in total load data file.
                                  [default: DNA, AGP, ER, Mito, RNA]
  --max_workers INTEGER           Number of workers to use for
                                  multiprocessing.  [default: 8]
  --force / --no-force            Whether to force redownload of load data
                                  files.  [default: force]
  -h, --help                      Show this message and exit.
```
