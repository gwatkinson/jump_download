# Downloading the metadata

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

Usage:

```bash
Usage: download_metadata [OPTIONS]

  Download the JUMP metadata files.

Options:
  --metadata_dir TEXT     Path to directory to save metadata files.
                          [default: /projects/cpjump1/jump/metadata]
  --metadata_script TEXT  Path to the script to download the metadata.
                          [default: /workspaces/biocomp/watkinso/mice/mice/download/metadata/download_jump_metadata.sh]
  --force / --no-force    Whether to force download the metadata files.
                          [default: no-force]
  -h, --help              Show this message and exit.
```
