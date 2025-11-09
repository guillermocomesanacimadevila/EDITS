import os
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


def download_and_unzip(url: str, new_folder: Path):
    import shutil
    if new_folder.exists():
        print(f"{new_folder} already downloaded, skipping.")
        return

    # Download the zip file

    if not Path("downloaded_file.zip").exists():
        download(url, "downloaded_file.zip")

    # Unzip the file
    with zipfile.ZipFile("downloaded_file.zip", "r") as zip_ref:
        zip_ref.extractall("temp_folder")

    # Rename the top-level folder
    new_folder.parent.mkdir(parents=True, exist_ok=True)
    # os.rename("temp_folder/" + os.listdir("temp_folder")[0], new_folder)
    src = "temp_folder/" + os.listdir("temp_folder")[0]
    dst = new_folder
    shutil.copytree(src, dst)
    # Delete the zip file and temporary folder
    os.remove("downloaded_file.zip")
    # os.rmdir("temp_folder")
    shutil.rmtree(src)


def download(url: str, fname: Path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(str(fname), "wb") as file, tqdm(
        desc=str(fname),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":
    from pdb import set_trace
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    args = parser.parse_args()

    out_path = Path(args.data_path)
    print(f"output path : {out_path}")
    # Hela
    download_and_unzip(
        "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip",
        out_path / "hela" / "train",
    )

    download_and_unzip(
        "http://data.celltrackingchallenge.net/test-datasets/Fluo-N2DL-HeLa.zip",
        out_path / "hela" / "test",
    )

    # Mdck
    mdck_path = out_path / "mdck.tif"
    if not mdck_path.exists():
        download("https://rdr.ucl.ac.uk/ndownloader/files/31127035", mdck_path)
    else:
        print(f"{mdck_path} already downloaded, skipping.")
