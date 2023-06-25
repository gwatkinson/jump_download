"""Recreate the submission.csv file mid way through the download. Used when there is a problem with the download."""

from glob import glob
from pathlib import Path

import click
import pandas as pd


def recreate_submission(jobs_dir, final_dir, file_name="submissions_left.csv"):
    """Recreate the submission.csv file mid way through the download. Used when there is a problem with the download."""
    # Read in the submission file
    submission = pd.read_csv(Path(jobs_dir) / "submission.csv", header=None, names=["file"])

    finished_jobs = glob(f"{final_dir}/final/*.parquet")
    finished_ids = [Path(job).stem for job in finished_jobs]

    jobs_left = ~submission["file"].apply(lambda x: Path(x).stem.split(".")[0]).isin(finished_ids)

    submissions_left = submission[jobs_left]

    output = Path(jobs_dir) / file_name
    submissions_left.to_csv(output, index=False, header=False)

    print(f"Created {str(output)} with {jobs_left.sum()}/{len(jobs_left)} submissions left.")

    return submissions_left


@click.command(
    "recreate_submission",
    help="Recreate the submission.csv file mid way through the download. Used when there is a problem with the download.",
    context_settings={
        "show_default": True,
        "help_option_names": ["-h", "--help"],
    },
)
@click.option("--jobs_dir", help="Directory where the jobs are stored.")
@click.option("--final_dir", help="Directory where the final files are stored.")
@click.option(
    "--file_name",
    default="submissions_left.csv",
    help="Name of the file to save the submissions left to.",
)
def main(jobs_dir, final_dir, file_name):
    """Recreate the submission.csv file mid way through the download. Used when there is a problem with the download."""
    recreate_submission(jobs_dir, final_dir, file_name)

    print("Don't forget to modify the condor/submit.sub file to use the new submission csv file.")


if __name__ == "__main__":
    main()
