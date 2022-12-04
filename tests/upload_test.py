import requests
import os
import base64


test_img_path = 'C:/Users/Dima/Pictures/bf3 1-29-2019 3-08-09 PM.mp4 (0_00_00) 000000.bmp'
test_img_path_2 = 'C:/Users/Dima/Pictures/f680891ffd4934ca7a61dcdbae7fe15c.png'

url = 'http://127.0.0.1:8000/predict'
file = {'files': (test_img_path.split('/')[-1], base64.b64encode(open(test_img_path, 'rb').read()))}

# from https://stackoverflow.com/questions/18179345/uploading-multiple-files-in-a-single-request-using-python-requests-module
files_two = [('files', (test_img_path.split('/')[-1], base64.b64encode(open(test_img_path, 'rb').read()))), \
             ('files', (test_img_path_2.split('/')[-1] ,base64.b64encode(open(test_img_path_2, 'rb').read())))]

resp = requests.post(url = url, files = file) # one file
print(resp.json())

resp = requests.post(url = url, files = files_two) # two files
print(resp.json())

# multiple images
img_test_dir_path = 'F:/AvanPost/Dataset/10/Subset'
img_pathes = [f'{img_test_dir_path}/{file}' for file in os.listdir(img_test_dir_path)]

files_multiple = [('files', (path.split('/')[-1], base64.b64encode(open(path, 'rb').read()))) for path in img_pathes]

resp = requests.post(url = url, files = files_multiple)
print(resp.json())
