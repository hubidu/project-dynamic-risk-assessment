import pandas as pd
import glob
from os import mkdir
from os.path import join, exists
import json
import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s", filename="debug.log", level=logging.DEBUG
)

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


def record_ingestion(files: list):
    logging.info(f"Recording ingested files {files} to {output_folder_path}")

    with open(join(output_folder_path, "ingested_files.txt"), "w") as f:
        f.write(str(files))


def merge_files(filepaths):
    logging.info(f"Merging data files {filepaths}")

    all_dfs = [pd.read_csv(f) for f in filepaths]

    merged_df = pd.concat(all_dfs)

    logging.info(f"Dropping duplicates in merged dataframe")
    merged_df.drop_duplicates(inplace=True, ignore_index=True)

    return merged_df


def merge_multiple_dataframe():
    logging.info(f"Starting ingestion")

    if not exists(output_folder_path):
        mkdir(output_folder_path)

    datafiles = glob.glob(f"*.csv", root_dir=input_folder_path)
    datafilepaths = glob.glob(f"{input_folder_path}/*.csv")

    merged_df = merge_files(datafilepaths)

    logging.info(f"Saving merged datafile to {output_folder_path}")
    merged_df.to_csv(join(output_folder_path, "finaldata.csv"))

    record_ingestion(datafiles)

    logging.info(f"Successfully ingested data.")


if __name__ == "__main__":
    merge_multiple_dataframe()
