import requests
import json

URL = "http://127.0.0.1:8000/"

response1 = requests.post(f"{URL}prediction", json = { "lastmonth_activity": 14, "lastyear_activity": 2145, "number_of_employees": 99})
print(response1.json())
response2 = requests.get(f"{URL}scoring") 
print(response2.json())
response3 = requests.get(f"{URL}summarystats") 
print(response3.json())
response4 = requests.get(f"{URL}diagnostics") 
print(response4.json())

#combine all API responses
# responses = response1.json() | response2.json() | response3.json() | response3.json()

#write the responses to your workspace
# with open('apireturns.txt', 'w') as convert_file:
#      convert_file.write(json.dumps(responses))



