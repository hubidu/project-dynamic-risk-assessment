import pandas as pd
import ast
import os
import glob
import json
import pickle
from sklearn.metrics import f1_score
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', filename='fullprocess.log', level=logging.DEBUG)

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = os.path.join(config['input_folder_path']) 
output_folder_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

def check_and_read_new_data():
    logging.info(f"Checking for new data files in {input_folder_path}")

    contents = ""
    with open(os.path.join(prod_deployment_path, "ingested_files.txt")) as f:
        contents = f.read()
    ingested_files = ast.literal_eval(contents)

    new_files = glob.glob(f'*.csv', root_dir=input_folder_path)
    
    new_files_to_process = set(new_files).difference(set(ingested_files)) 

    return list(new_files_to_process)


def score_model(testdata: pd.DataFrame):
    X_test = testdata[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1, 3)
    y_test = testdata["exited"].values

    logging.info(f"Deserializing trained model from {prod_deployment_path}")
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))

    logging.info(f"Calculating f1 score on {X_test.shape} training samples")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    return f1

def model_drift_has_occurred(new_files:  list):
    with open(os.path.join(prod_deployment_path, "latestscore.txt")) as f:
        contents = f.read()
    latest_score = float(contents)

    df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv")) 
    f1_score = score_model(df)

    logging.info(f"f1 score on new data is {f1_score}")

    if f1_score < latest_score:
        logging.info(f"Model drift detected: Merged {new_files} have a lower score {f1_score} than latest score {latest_score}")
        return True

    return False

def reingest():
    os.system("python ingestion.py")

def retrain():
    os.system("python training.py")

def rescore():
    os.system("python scoring.py")

def redeploy():
    os.system("python deployment.py")

def report():
    os.system("python reporting.py")

def apicalls():
    os.system("python apicalls.py")
	

def fullprocess():
    logging.info(f"Starting fullprocess...")

    new_files_to_process = check_and_read_new_data()
    if new_files_to_process == []:
        logging.info(f"There are no new files to process -> finishing")
        return

    reingest()

    logging.info(f"Found new files to process {new_files_to_process}")
    if not model_drift_has_occurred(new_files_to_process):
        logging.info(f"There is no model drift -> finishing")
        return

    logging.info(f"Retraining and redeploying model")
    retrain()
    rescore()
    redeploy()
    
    logging.info(f"Regenerating model reports")
    report()
    apicalls()

    logging.info(f"Successfully finished fullprocess")
    
if __name__ == '__main__':
    fullprocess()





