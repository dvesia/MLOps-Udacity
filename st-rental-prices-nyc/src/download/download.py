#!/usr/bin/env python
"""
Download data and save it in wandb

author: Vesia Domenico
date: July 2022
"""
import argparse
import logging
import os
import pathlib
import tempfile
from unicodedata import name
import requests

import wandb

from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Download and upload data to W&B
    """
    basename = pathlib.Path(args.file_url).name.split('?')[0].split('#')[0]

    logger.info("Downloading %s ...", args.file_url)

    # create a temporary file and write it in binary mode
    with tempfile.NamedTemporaryFile(mode='wb+') as file:
        logger.info("Creating run")
        # the response will be streamed 
        with requests.get(args.file_url, stream=True) as r:
            # iterate over the response 8192 bytes per time
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)
        # The flush() method in Python file handling clears the internal buffer of the file
        file.flush()

        logger.info("Creating Artifact")
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description,
            metadata={'original_url': args.file_url}
        )
        artifact.add_file(file.name, name=basename)

        logger.info("Logging artifact")
        run.log_artifact(artifact)


if __name__ == "__main__":
    # Create CLI 
    parser=argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B",
        fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--file_url", type=str, help="URL to the input file", required=True
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description", type=str, help="Description for the artifact", required=True
    )

    args = parser.parse_args()

    go(args)
