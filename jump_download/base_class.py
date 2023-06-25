"""Main abstract base class to download objects from S3."""

import sys
from abc import ABC, abstractmethod
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor

from tqdm.auto import tqdm

from jump_download.utils import time_decorator


class BaseDownload(ABC):
    """Meta class that defines the methods to download objects from S3.

    It is an abstract class that needs to be inherited and the methods need to be implemented.
    It is used to download objects from S3 in parallel using multiprocessing.

    Methods:
        execute_job: Execute a single image job that consists of downloading objects from S3.
        get_job_list: Return a list of jobs to be executed.
        post_process: Post process the results of the download.
        download_with_multiprocessing_generator: Download from S3 using multiprocessing as a generator.
        download_objects_from_jobs: Iterate over the result of the generator and return a list of jobs with the result.
        download: Method that queues the other methods.
        download_timed: Method that queues the other methods and times the execution.
    """

    def __init__(self, max_workers) -> None:
        self._max_workers = max_workers

    @property
    def max_workers(self):
        return self._max_workers

    @abstractmethod
    def execute_job(self, job) -> str:
        """Execute a single image job that consists of downloading objects from S3."""
        raise NotImplementedError

    @abstractmethod
    def get_job_list(self) -> list:
        """Return a list of jobs to be executed."""
        raise NotImplementedError

    @abstractmethod
    def post_process(self, download_results):
        """Post process the results of the download."""
        raise NotImplementedError

    def download_with_multiprocessing_generator(self, jobs):
        """Download from S3 using multiprocessing as a generator."""
        with tqdm(total=len(jobs), file=sys.stdout) as pbar, ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_key = {executor.submit(self.execute_job, job): job for job in jobs}

            for future in futures.as_completed(future_to_key):
                pbar.update(1)
                key = future_to_key[future]
                exception = future.exception()

                if not exception:
                    yield key, future.result()
                else:
                    yield key, exception

    def download_objects_from_jobs(self, jobs):
        """Iterate over the result of the generator and return a list of jobs with the result."""
        download_results = []
        for job, result in self.download_with_multiprocessing_generator(jobs):
            job["result"] = result
            download_results.append(job)

        return download_results

    def download(self):
        """Method that queues the other methods."""
        print("\n=== Creating the jobs...")
        jobs = self.get_job_list()

        print("\n=== Running the jobs...")
        download_results = self.download_objects_from_jobs(jobs)

        print("\n=== Post processing the results...")
        post_process_results = self.post_process(download_results)

        print("\n=== Done!")

        return post_process_results, download_results

    @time_decorator
    def download_timed(self):
        res = self.download()
        return res
