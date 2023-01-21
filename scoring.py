import pandas as pd
import pickle
import os
from os.path import join
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', filename='debug.log', level=logging.DEBUG)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = join(config['output_model_path']) 


def record_score(score):
    logging.info(f"Recording score {score} ")

    with open(join(model_path, "latestscore.txt"), "w") as f:
        f.write(str(score))

def score_model():
    logging.info(f"Reading testdata from {test_data_path}")
    testdata = pd.read_csv(join(test_data_path, "testdata.csv"))
    X_test = testdata[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1, 3)
    y_test = testdata["exited"].values

    logging.info(f"Deserializing trained model from {model_path}")
    model = pickle.load(open(join(model_path, 'trainedmodel.pkl'), 'rb'))

    logging.info(f"Calculating f1 score on {X_test.shape} training samples")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    record_score(f1)

    logging.info(f"Successfully scored model.")
    return f1

if __name__ == '__main__':
    score_model()