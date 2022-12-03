import requests

url = 'http://127.0.0.1:8000/status'
resp = requests.get(url=url)
print(resp.json())