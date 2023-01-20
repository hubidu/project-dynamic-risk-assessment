import shutil
from os import mkdir
from os.path import join, exists
import json

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = join(config['output_folder_path']) 
prod_deployment_path = join(config['prod_deployment_path']) 
model_path = join(config['output_model_path']) 

def store_model_into_pickle():
    if (not exists(prod_deployment_path)):
        mkdir(prod_deployment_path)

    shutil.copyfile(join(model_path, "trainedmodel.pkl"), join(prod_deployment_path, "trainedmodel.pkl"))
    shutil.copyfile(join(model_path, "latestscore.txt"), join(prod_deployment_path, "latestscore.txt"))
    shutil.copyfile(join(dataset_csv_path, "ingested_files.txt"), join(prod_deployment_path, "ingested_files.txt"))
        
if __name__ == '__main__':
    store_model_into_pickle()
        

