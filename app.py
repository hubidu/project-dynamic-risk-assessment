from flask import Flask, session, jsonify, request
import pandas as pd
import json
import os
from diagnostics import model_predictions, dataframe_summary, dataframe_missing, execution_time, outdated_packages_list
from scoring import score_model


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 


@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    if request.method == 'POST':
        post_data = request.get_json()
        df = pd.DataFrame(post_data, index=[0])
        y_pred, _ = model_predictions(df) 
        return jsonify({ "prediction": int(y_pred[0]) })
    

@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    testdata = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    return jsonify({ "score": score_model(testdata)}) 

@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    return  jsonify({ "summarystats": dataframe_summary() })

@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    missing_values = dataframe_missing()
    timings = execution_time()
    deps = outdated_packages_list()
    return jsonify({
        "missing_values": missing_values,
        "timings": timings,
        "outdated_packages": deps
    }) 


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
