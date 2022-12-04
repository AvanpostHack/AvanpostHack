import json
import os
import pathlib

from fastapi import FastAPI, File, UploadFile, Form, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import get_predict_from_model, start_training
from typing import Union, List

import uvicorn
import base64
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

def clean_folders(file_path: str):
    for filepath in pathlib.Path(file_path).glob('**/*'):
        if not '.gitkeep' in str(filepath):
            os.remove(filepath)

clean_folders('./downloaded_images')
clean_folders('./uploaded_images')
clean_folders('./datasets/test')
def get_status_config():
    with open('./config.yaml') as f:
        return yaml.safe_load(f)

@app.get("/status")
def get_status():
    config = get_status_config()

    return {"status": config['status'], "model_status": config['model_status'], "code": config['code']}

@app.get("/get_classes")
def get_classes():
    with open('./classes.json', 'r') as f:
        class_list = json.load(f)
        return class_list

@app.post("/predict")
def get_predict(images: dict = Body(...)):

    file_names = []
    file_data = []

    for key, value in images.items():
        try:
            contents = base64.b64decode(value)
            file_names.append(key)
            with open('datasets/test/' + key, 'wb') as f:
                f.write(contents)
                file_data.append(contents)
        except Exception:
            return {"error": "There was an error uploading the file"}

    model_predicts = get_predict_from_model(file_data)
    clean_folders('./datasets/test')
    return model_predicts

    # response_arr = []
    # for i, predict in enumerate(model_predicts):
    #     file_name = ''
    #     try:
    #         file_name = file_names[i]
    #     except:
    #         file_name = ''
    #
    #     response_arr.append({"predicted_class": predict[0], "сonfidence": predict[1], "filename": file_name})
    #
    # return {"preds_arr": response_arr, "error": ""}


@app.post("/fit")
def fit_model(keyword: Keyword):
    start_training(keyword.keyword)
    config = get_status_config()

    return {"status": config['model_status'], "error": ""}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)