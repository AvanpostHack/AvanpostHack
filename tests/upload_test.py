import requests

test_img_path = 'C:/Users/Dima/Pictures/bf3 1-29-2019 3-08-09 PM.mp4 (0_00_00) 000000.bmp'
test_img_path_2 = 'C:/Users/Dima/Pictures/f680891ffd4934ca7a61dcdbae7fe15c.png'

url = 'http://127.0.0.1:8000/predict'
file = {'files': open(test_img_path, 'rb')}

# from https://stackoverflow.com/questions/18179345/uploading-multiple-files-in-a-single-request-using-python-requests-module
files_two = [('files', open(test_img_path, 'rb')), ('files', open(test_img_path_2, 'rb'))]

# resp = requests.post(url = url, files = file) # one file
# print(resp.json())

resp = requests.post(url = url, files = files_two) # two files
print(resp.json())