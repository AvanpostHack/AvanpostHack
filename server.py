from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import get_predict_from_model, start_training
from typing import Union

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

def get_status_config():
    with open('./config.yaml') as f:
        return yaml.safe_load(f)

@app.get("/status") # ОТКУДА брать инфу о статусе пока не ясно
def get_status():
    config = get_status_config()

    return {"status": config.status, "model_status": config.model_status, "code": config.code}

@app.post("/predict")
def get_predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open('uploaded_images/' + file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"error": "There was an error uploading the file"}
    finally:
        file.file.close()

    class_, conf = get_predict_from_model(image_data = contents)
    return {"predicted_class": class_, "сonfidence": conf, "error": " "}

@app.post("/fit")
def fit_model(keyword: Keyword):
    # print(keyword.keyword)
    start_training(keyword.keyword)
    config = get_status_config()
    return {"status": config.model_status, "error": ""}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)