"""`fetch.py` - fetch data; expects a command line argument specifying parameter group"""
import argparse
import gzip
import os
import pathlib
import shutil
from zipfile import ZipFile
import yaml

import requests
from tqdm import tqdm
from utils import fix_path


def get_file(uri: str, filename: str) -> None:
    """
    Download the file, and decompress if gzip
    """
    download_path = pathlib.Path(uri)
    downloaded_file = download_path.name
    downloaded_path = fix_path(f"../data/raw/{downloaded_file}")
    final_path = fix_path(f"../data/raw/{filename}")
    if pathlib.Path(final_path).exists():
        print(f"File {final_path} exists; remove to re-download")
        return

    print(f"Downloading: {uri} ...")
    response = requests.get(uri, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(downloaded_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if downloaded_file.endswith(".gz"):
        with gzip.open(downloaded_path, "rb") as f_in:
            with open(final_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(downloaded_path)
    if downloaded_file.endswith(".zip"):
        final_path = fix_path("../data/raw/")
        print("Unzipping...")
        with ZipFile(downloaded_path) as z_f:
            z_f.extractall(path=final_path)
        os.remove(downloaded_path)
        print("Done!")


if __name__ == "__main__":
    # Fetch processing params
    with open(fix_path("../params.yaml"), "r") as fd:
        params = yaml.safe_load(fd)
    parser = argparse.ArgumentParser(description="find where to fetch params.")
    parser.add_argument("--param_group", nargs=1, type=str)
    args = parser.parse_args()
    PARAM_GROUP = args.param_group[0]
    EMBEDDINGS_FILE = params[PARAM_GROUP]["file"]
    EMBEDDINGS_URI = params[PARAM_GROUP]["uri"]
    get_file(EMBEDDINGS_URI, EMBEDDINGS_FILE)
