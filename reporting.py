from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from diagnostics import model_predictions


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 

def calculate_confusion_matrix():
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    y_pred, y = model_predictions(df)

    cm = metrics.confusion_matrix(y, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(output_path, "confusionmatrix.png"))


if __name__ == '__main__':
    calculate_confusion_matrix()
