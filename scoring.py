import pandas as pd
import pickle
import os
from os.path import join
from sklearn.metrics import f1_score
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

def score_model(testdata: pd.DataFrame):
    X_test = testdata[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1, 3)
    y_test = testdata["exited"].values

    logging.info(f"Deserializing trained model from {model_path}")
    model = pickle.load(open(join(model_path, 'trainedmodel.pkl'), 'rb'))

    logging.info(f"Calculating f1 score on {X_test.shape} test samples")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    return f1

if __name__ == '__main__':
    logging.info(f"Scoring model")

    logging.info(f"Reading testdata from {test_data_path}")
    testdata = pd.read_csv(join(test_data_path, "testdata.csv"))

    f1_score = score_model(testdata)
    record_score(f1_score)

    logging.info(f"Successfully scored model {f1_score}")