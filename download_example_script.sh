# This script runs the commands in sequence to download the metadata and data files

CONF_NAME=$1
JOB_PATH=$2

echo "Downloading metadata and data files using the configuration file: $CONF_NAME"

echo "Creating Poetry environment"
poetry install --without dev            # Install the dependencies
POETRY_ENV=$(poetry env info --path)    # Get the path of the environment
source "$POETRY_ENV/bin/activate"       # Activate the environment

echo "Downloading metadata files"
download_metadata -cn $CONF_NAME

echo "Downloading load data files"
download_load_data_files -cn $CONF_NAME

echo "Filtering and sampling images to download"
create_job_split -cn $CONF_NAME

echo "Downloading images using job in $JOB_PATH"
download_images_from_job -cn $CONF_NAME run.job_path=$JOB_PATH

# Or use the sub file
# condor_submit ./jump_download/condor/submit.sub
