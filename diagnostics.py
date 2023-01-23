import re
import pandas as pd
import timeit
import os
from os.path import join
import json
import pickle

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = join(config["output_folder_path"])
test_data_path = join(config["test_data_path"])
prod_deployment_path = join(config["prod_deployment_path"])


def model_predictions(df: pd.DataFrame):
    input_data = df[
        ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    ].values.reshape(-1, 3)
    y = None
    if "exited" in df:
        y = df["exited"].values

    model = pickle.load(open(join(prod_deployment_path, "trainedmodel.pkl"), "rb"))
    return [model.predict(input_data), y]


def dataframe_summary():
    df = pd.read_csv(join(dataset_csv_path, "finaldata.csv"))
    df_numeric = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    return [
        df_numeric.mean().round(decimals=2).values.tolist(),
        df_numeric.median().round(decimals=2).values.tolist(),
        df_numeric.std().round(decimals=2).values.tolist(),
    ]


def dataframe_missing():
    # calculate summary statistics here
    df = pd.read_csv(join(dataset_csv_path, "finaldata.csv"))
    return (df.isna().sum() / len(df)).values.tolist()


def execution_time():
    # calculate timing of training.py and ingestion.py
    ingestion_time = timeit.timeit(lambda: os.system("python ingestion.py"), number=1)
    training_time = timeit.timeit(lambda: os.system("python training.py"), number=1)
    return [ingestion_time, training_time]


def outdated_packages_list():
    # get a list of
    stream = os.popen("pip list --outdated")
    output = stream.read()
    lines = output.split("\n")[2:]

    return [re.split(r"\s+", line)[:2] for line in lines]


if __name__ == "__main__":
    df = pd.read_csv(join(test_data_path, "testdata.csv"))
    model_predictions(df)

    dataframe_summary()
    dataframe_missing()
    execution_time()
    outdated_packages_list()
