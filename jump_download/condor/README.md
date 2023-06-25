## Run the jobs on the cluster with Condor

This is the last step of the pipeline. It uses the `download_plate` command to download the images from the S3 bucket.

It consists only of a single `.sub` file that tells how much ressources should be used, and how many jobs should be run in parallel on different nodes.

```bash
condor_submit mice/download/condor/download_plate.sub
```
