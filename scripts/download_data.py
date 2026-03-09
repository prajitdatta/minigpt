#!/usr/bin/env python3
"""
Dataset Downloader
==================

Download and prepare common training datasets.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import os
import argparse
import urllib.request


DATASETS = {
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "shakespeare.txt",
        "description": "Tiny Shakespeare (~1MB, good for quick experiments)",
    },
}


def download_file(url: str, dest: str):
    """Download a file with progress."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"Downloaded: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument(
        "dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to download",
    )
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    info = DATASETS[args.dataset]
    dest = os.path.join(args.output_dir, info["filename"])

    print(f"Dataset: {args.dataset}")
    print(f"Description: {info['description']}")
    download_file(info["url"], dest)
    print("Done!")


if __name__ == "__main__":
    main()
