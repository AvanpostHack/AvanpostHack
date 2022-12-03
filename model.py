# Тут предполагается код модели
from __future__ import annotations
import io
from time import sleep
from PIL import Image
import yaml

# from image_downloader import start_image_downloads # поставить пакет и откоментировать

def get_predict_from_model(image_data: list[bytes]) -> list[tuple[int, float]]:
    images_list = [] # List of PIL Images
    for item in image_data:
        image = Image.open(io.BytesIO(item))
        images_list.append(image)
        # print(image.width, image.height)

    # convert to numpy / tensor whatever
    # run model to predict

    return [(10, 0.9), (10, 0.9)] # return class index and confidence

def model_fit():
    print('go to sleep')
    sleep(1)
    # нужно сделать чтобы модель записывала статус обучения

def start_train_config():
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config['model_status'] = 'waiting'
    with open('./config.yaml', 'w') as f:
        yaml.dump(config, f)

def end_train_config():
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config['model_status'] = 'fitted'
    with open('./config.yaml', 'w') as f:
        yaml.dump(config, f)

def start_training(keyword: str):
    start_train_config()
    # загружаем картинки из интернета
    # start_image_downloads(keyword, './downloaded_images')

    model_fit() # запускаем дообучение модели
    end_train_config()