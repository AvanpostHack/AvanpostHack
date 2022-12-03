import json
import os
import pathlib

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import get_predict_from_model, start_training
from typing import Union, List

import uvicorn

import yaml

app = FastAPI(debug=True)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Keyword(BaseModel):
    keyword: str
    epoch: Union[int, None] = None
    accuracy: Union[float, None] = None

def clean_folders(file_path: str):
    for filepath in pathlib.Path(file_path).glob('**/*'):
        if not '.gitkeep' in str(filepath):
            os.remove(filepath)

clean_folders('./downloaded_images')
clean_folders('./uploaded_images')
def get_status_config():
    with open('./config.yaml') as f:
        return yaml.safe_load(f)

@app.get("/status")
def get_status():
    config = get_status_config()

    return {"status": config['status'], "model_status": config['model_status'], "code": config['code']}

@app.get("/get_classes")
def get_classes():
    with open('config.json', 'r') as f:
        class_list = json.load(f)
        return class_list

@app.post("/predict")
def get_predict(files: List[UploadFile] = File(...)):

    file_names = []
    file_data = []

    for i, item in enumerate(files):
        try:
            contents = item.file.read()
            file_names.append(item.filename)
            with open('uploaded_images/' + item.filename, 'wb') as f:
                f.write(contents)
                file_data.append(contents)
        except Exception:
            return {"error": "There was an error uploading the file"}
        finally:
            item.file.close()

    model_predicts = get_predict_from_model(file_data)

    response_arr = []
    for i, predict in enumerate(model_predicts):
        file_name = ''
        try:
            file_name = file_names[i]
        except:
            file_name = ''

        response_arr.append({"predicted_class": predict[0], "сonfidence": predict[1], "filename": file_name})

    return {"preds_arr": response_arr, "error": ""}


@app.post("/fit")
def fit_model(keyword: Keyword):
    # print(keyword.keyword)
    start_training(keyword.keyword)
    config = get_status_config()

    return {"status": config['model_status'], "error": ""}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)