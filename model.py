# Тут предполагается код модели
from __future__ import annotations
import io
from time import sleep
from PIL import Image
import yaml

# from image_downloader import start_image_downloads # поставить пакет и откоментировать

from ac_model import ACLoaderDataset, ACModel

def get_predict_from_model(image_data: list[bytes]) -> list[tuple[int, float]]:
    images_list = [] # List of PIL Images
    for item in image_data:
        image = Image.open(io.BytesIO(item))
        images_list.append(image)

    # run model to predict (for images in dirs)

    # Прогнозирование по существующей моделе, Важно чтобы size_img был точно таким же как и при обучении
    load_ac_model = ACModel(size_img=200, check_gpu=False)
    # load_ac_model = ACModel(size_img=200, batch_size = 10, num_epochs=1, check_gpu=False, num_workers=0)
    load_ac_model.load_model()
    detected_files = load_ac_model.predict_imagefiles()
    # print(detected_files)

    return detected_files # return class index and confidence

def model_fit():
    # Загрузка нового класса изображений
    ac_loader = ACLoaderDataset(size_img=128, batch_size=10, num_workers=0, use_gpu=False)
    path_load = ac_loader.add_new_class(classname="skateboard", num=800)

    # При добавлении нового класса делается автоматом
    # Подразумевается что в каталоге dataset/train расположены тернировочные датасеты
    # Но для моедли надо отдельно выделить валидационные данные
    # Формирвоание Валидацинных данных. Если в dataset/val нет каталога, который есть в dataset/train,
    # то производим добавление соответсвтующего класса и переносим часть данных как валидационных
    # ac_loader = ACLoaderDataset(size_img=200, use_gpu=False)
    # ac_loader.create_val_dataset(val_size=0.1)

    # Начальное формирование модели по тем классам по которым есть данные
    ac_model = ACModel(size_img=128, batch_size=10, num_epochs=1, check_gpu=False, num_workers=0)
    # Модель формируется на основе предобученной ResNet50 но уже с новыми данными
    ac_model.new_model()
    # Сохраняем модель + сохраняем классы по которым считались
    ac_model.save_model()

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
    # start_image_downloads(keyword, './downloaded_images')

    model_fit() # запускаем дообучение модели
    end_train_config()