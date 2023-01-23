import pandas as pd
import pickle
from os import mkdir
from os.path import join, exists
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s", filename="debug.log", level=logging.DEBUG
)

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = join(config["output_folder_path"])
model_path = join(config["output_model_path"])


def train_model():
    if not exists(model_path):
        mkdir(model_path)

    logging.info(f"Reading training data from {dataset_csv_path}")
    df = pd.read_csv(join(dataset_csv_path, "finaldata.csv"))

    X = df[
        ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    ].values.reshape(-1, 3)
    y = df["exited"].values

    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    logging.info(
        f"Fitting model to training data ({X.shape}) using logistic regression"
    )
    model = lr.fit(X, y)

    logging.info(f"Saving model to {model_path}")
    pickle.dump(model, open(join(model_path, "trainedmodel.pkl"), "wb"))

    logging.info(f"Successfully trained model")


if __name__ == "__main__":
    train_model()
