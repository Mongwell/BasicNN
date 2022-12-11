"""
A few utility functions for setting up the dataset.

Intended to  be used with MNIST style datasets.
"""


class DownloadError(Exception):
    """Exception type for failed downloads."""

    pass


def data_prep(url, checksum=""):
    """
    Download file from URL and verify checksum if provided.

    The URL should point to a gzip compressed MNIST-style dataset,
    for example fashion MNIST: https://github.com/zalandoresearch/fashion-mnist
    """
    import requests
    import hashlib
    import os

    data_dir = "./data/"
    resp = requests.get(url)
    filename = data_dir + url.split("/")[-1]

    if checksum != "":
        hash = hashlib.md5(resp.content).hexdigest()
        if hash != checksum:
            raise DownloadError("Download does not match provided checksum!")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(filename, "wb") as f:
        f.write(resp.content)
        f.close()

    return filename


def load_labels(filename):
    """Load a gzip compressed file of dataset labels."""
    return _load(filename, 8)


def load_data(filename):
    """Load a gzip compressed file of dataset examples."""
    return _load(filename, 16)


def _load(filename, offset):
    """Open a gzip compressed numpy bytes file."""
    import gzip
    import numpy as np

    with gzip.open(filename, "rb") as npfile:
        return np.frombuffer(npfile.read(), dtype=np.uint8, offset=offset)
