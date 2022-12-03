import requests

test_img_path = 'C:/Users/Dima/Pictures/bf3 1-29-2019 3-08-09 PM.mp4 (0_00_00) 000000.bmp'
url = 'http://127.0.0.1:8000/predict'
file = {'file': open(test_img_path, 'rb')}
resp = requests.post(url=url, files=file)
print(resp.json())