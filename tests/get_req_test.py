import requests

url = 'http://127.0.0.1:8000/status'
resp = requests.get(url=url)
print(resp.json())

url = 'http://127.0.0.1:8000/get_classes'
resp = requests.get(url=url)
print(resp.json())