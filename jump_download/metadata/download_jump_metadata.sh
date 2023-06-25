# Script Name: download_jump_metadata.sh
#
# Description: This script retrieves the metadata for the JUMP dataset.
#
# Usage: ./download_jump_metadata.sh [METADATA_DIR]
#
# The resulting metadata will be stored in $METADATA_DIR.
#
# Output Files:
#   - compound.csv.gz: the table that lists all the compound perturbations in the JUMP.
#   - crispr.csv.gz: the table that lists all the CRISPR perturbations in the JUMP.
#   - orf.csv.gz: the table that lists all the ORF perturbations in the JUMP.
#   - plate.csv.gz: the table that lists all the plates in the JUMP. This is the most important file as it allows to create the path to the other files.
#   - well.csv.gz: the table that links the wells in the JUMP with the perturbation and plate.
#   - microscope_config.csv and microscope_filter.csv: the tables that list the microscope configuration and filters used in the JUMP. Not used in this project.
#   - the compressed files all have an uncompressed version in the same directory.
#
# Author: Gabriel Watkinson
# Date: 2023-05-31

# Set the metadata directory.
METADATA_DIR=${1:-"$METADATA_DIR"}
FORCE=${2:-"--force"}

# Check that files don't already exist.
if [ "$FORCE" != "--force" ] && [ -f "$METADATA_DIR/plate.csv" ] && [ -f "$METADATA_DIR/well.csv" ] && [ -f "$METADATA_DIR/plate.csv" ] && [ -f "$METADATA_DIR/crispr.csv" ] && [ -f "$METADATA_DIR/orf.csv" ] && [ -f "$METADATA_DIR/microscope_config.csv" ] && [ -f "$METADATA_DIR/microscope_filter.csv" ]; then
    echo "Metadata files already exist. Use --force to overwrite."
    exit 1
fi

# Create the directory where the metadata will be stored if needed.
echo "Creating metadata directory: $METADATA_DIR ..."
mkdir -p $METADATA_DIR

# Clone the repository containing the metadata for the JUMP dataset.
echo "Cloning metadata repository https://github.com/jump-cellpainting/datasets.git ..."
git clone https://github.com/jump-cellpainting/datasets.git "$METADATA_DIR/tmp"

# Keep only the interesting folder.
echo "Moving metadata folder..."
mv "$METADATA_DIR/tmp/metadata/"* "$METADATA_DIR"

# Decrompress the files.
echo "Decompressing metadata files..."
for file in $METADATA_DIR/*.csv.gz; do
    gunzip -c "$file" > "${file%.gz}"
done

# Remove the temporary folder.
echo "Removing temporary folder..."
rm -rf "$METADATA_DIR/tmp"

echo "Metadata retrieval complete."
