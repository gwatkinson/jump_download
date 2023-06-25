#!/bin/bash

# Script Name: download_plate.sh
#
# Description: This script retrieves all the images from a plate given the source, batch, etc.
#
# Usage: ./download_plate.sh [JOB_CSV_FILE] [MAX_WORKERS] [REMOVE_TMP] [FORCE]
#
# This uses the default value defined in the mice/download/constants.py file.

# Check that the csv file exists.
JOB_CSV_FILE=$1
MAX_WORKERS=$2  # Number of workers to use.

# Activate the conda environment and check that it exists.
echo "=== Activating poetry environment '$POETRY_ENV'... ==="

source "$POETRY_ENV/bin/activate"
if [ $? -ne 0 ]; then
    echo "Poetry environment '$POETRY_ENV' does not exist."
    exit 1
fi

# Run python script to download the images.
echo "=== Running job $JOB_CSV_FILE... ==="
download_images_from_csv run.job_csv_path=$JOB_CSV_FILE run.max_workers=$MAX_WORKERS
