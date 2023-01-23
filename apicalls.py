import requests
import json
import os
import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s", filename="debug.log", level=logging.DEBUG
)

with open("config.json", "r") as f:
    config = json.load(f)

output_model_path = os.path.join(config["output_model_path"])

URL = "http://127.0.0.1:8000/"

predictionRes = requests.post(
    f"{URL}prediction",
    json={
        "lastmonth_activity": 14,
        "lastyear_activity": 2145,
        "number_of_employees": 99,
    },
)
logging.info(f"Prediction Response: {predictionRes.json()}")

scoringRes = requests.get(f"{URL}scoring")
logging.info(f"Prediction Response: {scoringRes.json()}")

summaryRes = requests.get(f"{URL}summarystats")
logging.info(f"Prediction Response: {summaryRes.json()}")

diagnosticsRes = requests.get(f"{URL}diagnostics")
logging.info(f"Prediction Response: {diagnosticsRes.json()}")

# combine all API responses
combinedRes = (
    predictionRes.json() | scoringRes.json() | summaryRes.json() | diagnosticsRes.json()
)

# write the responses to your workspace
with open(os.path.join(output_model_path, "apireturns.txt"), "w") as convert_file:
    convert_file.write(json.dumps(combinedRes))
