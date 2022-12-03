import requests

url = 'http://127.0.0.1:8000/fit'
json_body = {"keyword": "Testiiiiii"}
resp = requests.post(url=url, json=json_body)
print(resp.json())