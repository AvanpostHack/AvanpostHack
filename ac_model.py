import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

import json  # Для конфигурации используется json

import torchvision
from torchvision import datasets, models, transforms
from PIL import Image, ImageOps, ImageEnhance
from PIL import ImageFile

# Библиотека для скачивания изображений из Google
# !pip install Google-Images-Search
# !pip install windows-curses
from google_images_search import GoogleImagesSearch

# Из-за того что при загрузке данных могут попадаться файлы не только в RGB но и в RGBA, требуется установить доп. условие
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
from os import listdir
from os.path import isfile, join

from tqdm.autonotebook import tqdm, trange

# Структура каталога:
# datasets/
#     /train # В директории train хранятся данные по классам разбитые по директориям, где название это наименование класса
#     /val # В директории val хранятся валидационные данные по классам разбитые по директориям, где название это наименование класса
#     /test # тестовые файлы, хранятся кучей, перед тем как записывать новые файлы надо очистить от старых

PATH_DATASET = "datasets"
PATH_DATASET_TRAIN = PATH_DATASET + "/train"
PATH_DATASET_VAL = PATH_DATASET + "/val"
PATH_DATASET_TEST = PATH_DATASET + "/test"
FILE_CLASSES = 'classes.json'  # Здесь хранятся все классы к текущей модели
FILE_CONFIG = 'prj_config.json'  # Здесь хранятся ключи для загрузки изображений


class ACLoaderDataset():

    def __init__(self, size_img=244, batch_size=10, num_workers=2, use_gpu=False):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.size_img = size_img
        self.use_gpu = use_gpu

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(self.size_img),
                transforms.CenterCrop(self.size_img),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.size_img),
                transforms.CenterCrop(self.size_img),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def add_new_class(self, classname, num=200, resize=500):
        path_new_class = PATH_DATASET_TRAIN + f"/{classname}"
        for _ in range(3):
            try:
                self.search_download_img(classname, path_new_class=path_new_class, num=num, resize=resize)
                break
            except:
                time.sleep(10)
                continue
        self.create_val_dataset(val_size=0.1)
        return path_new_class

    def search_download_img(self, classname, path_new_class, num=200, resize=500):
        # Read keys from config
        with open(FILE_CONFIG, 'r') as f:
            config = json.load(f)
        google_developer_key = config['google_developer_key']
        google_custom_search_cx = config['google_custom_search_cx']

        gis = GoogleImagesSearch(google_developer_key, google_custom_search_cx)
        # Множественный выбор параметров пока недоступен
        _search_params = {
            'q': classname,
            'num': num,
            'fileType': 'jpg',
            'imgType': 'photo',
            'imgSize': 'medium',
            'imgColorType': 'color'
        }
        # this will search, download and resize:
        gis.search(search_params=_search_params, path_to_dir=(PATH_DATASET_TRAIN + f"/{classname}"), width=resize,
                   height=resize)
        return path_new_class

    def create_val_dataset(self, val_size=0.1):
        val_dirs = [dir for dir in listdir(PATH_DATASET_VAL) if not isfile(join(PATH_DATASET_VAL, dir))]
        for exist_dir in [dir for dir in listdir(PATH_DATASET_TRAIN) if not isfile(join(PATH_DATASET_TRAIN, dir))]:
            train_dir = PATH_DATASET_TRAIN + f"/{exist_dir}"
            # Если для тернировочных данных нет валидационных формируем их
            if not exist_dir in val_dirs:
                val_dir = PATH_DATASET_VAL + f"/{exist_dir}"
                os.mkdir(val_dir)
                # Переносим часть данных из train в val
                train_files = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
                val_files = random.sample(train_files, k=int(len(train_files) * val_size))
                for move_file in val_files:
                    os.replace(train_dir + f"/{move_file}", val_dir + f"/{move_file}")
        return val_dirs

    def get_classes(self):
        self.classes = [dir for dir in listdir(PATH_DATASET_TRAIN) if not isfile(join(PATH_DATASET_TRAIN, dir))]
        return self.classes

    def get_datasets(self):
        # Преобразование обучающих данных для расширения обучающей выборки и её нормализация
        # Для валидационной (тестовой) выборки только нормализация

        image_datasets = {x: datasets.ImageFolder(os.path.join(PATH_DATASET, x),
                                                  self.data_transforms[x])
                          for x in ['train', 'val']}
        # загрузка данных в виде батчей
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size,
                                                           shuffle=True, num_workers=self.num_workers)
                            for x in ['train', 'val']}
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes
        self.count_classes = len(self.class_names)
        return self.dataloaders


class ACModel():
    def __init__(self, size_img=150, batch_size=10, num_epochs=3, check_gpu=False, num_workers=0):
        if check_gpu:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = False
        self.size_img = size_img
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.filename_state_model = 'ResNet_extractor.pth'

    def new_model(self):
        self.model_extractor = self.load_basemodel()
        self.ac_loader = self.get_datasets()
        self.count_classes = self.ac_loader.count_classes
        self.dataloaders = self.ac_loader.dataloaders
        self.change_output_classes()
        self.train()

    def get_datasets(self):
        self.ac_loader = ACLoaderDataset(size_img=self.size_img, batch_size=self.batch_size,
                                         num_workers=self.num_workers, use_gpu=self.use_gpu)
        self.ac_loader.get_datasets()
        self.class_names = self.ac_loader.class_names
        return self.ac_loader

    def load_basemodel(self):
        self.model_extractor = models.resnet50(weights="DEFAULT")
        return self.model_extractor

    def save_model(self):
        torch.save(self.model_extractor, self.filename_state_model)
        self.save_classes()
        return self.filename_state_model

    def save_classes(self):
        # save classname
        config = {}
        config['CLASSES'] = self.class_names
        with open(FILE_CLASSES, 'w') as f:
            json.dump(config, f, indent=2)
        return self.class_names

    def load_model(self):
        self.model_extractor = torch.load(self.filename_state_model)
        self.load_classes()
        return self.model_extractor

    def load_classes(self):
        # load classname
        with open(FILE_CLASSES, 'r') as f:
            config = json.load(f)
        self.class_names = config['CLASSES']
        return self.class_names

    def predict_imagefiles(self):
        ac_loader = ACLoaderDataset(size_img=self.size_img, batch_size=self.batch_size)
        data_transforms = ac_loader.data_transforms["val"]
        detected_files = {}
        for image_file in [file for file in listdir(PATH_DATASET_TEST) if isfile(join(PATH_DATASET_TEST, file))]:
            filename = PATH_DATASET_TEST + "/" + image_file
            img = Image.open(filename)
            # Для png RGBA Необходимо исключать прозрачный слой за счет перевода в RGB
            img = img.convert('RGB')
            x = data_transforms(img)
            x = x.unsqueeze(0)
            predict_classname = load_ac_model.model_extractor(x)
            detect_classname = self.class_names[torch.argmax(predict_classname, -1)]
            detected_files[image_file] = detect_classname
        return detected_files

    def change_output_classes(self):
        # замораживаем параметры (веса)
        for param in self.model_extractor.parameters():
            param.requires_grad = False
        # num_features -- это размерность вектора фич, поступающего на вход FC-слою
        num_features = 2048  # У ResNet50 это 2048 фич
        # Заменяем Fully-Connected слой на наш линейный классификатор
        self.model_extractor.fc = nn.Linear(num_features, self.count_classes)

    def train_model(self, model, criterion, optimizer, ac_loader, num_epochs=10):
        dataloaders = ac_loader.dataloaders
        dataset_sizes = ac_loader.dataset_sizes
        since = time.time()

        best_model_wts = model.state_dict()
        best_acc = 0.0

        # Ваш код здесь
        losses = {'train': [], "val": []}

        pbar = trange(num_epochs, desc="Epoch:")
        for epoch in pbar:
            # for epoch in range(num_epochs):

            # каждя эпоха имеет обучающую и тестовую стадии
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train(True)  # установаить модель в режим обучения
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # итерируемся по батчам
                # for data in tqdm(self.dataloaders[phase], leave=False, desc=f"{phase} iter:"):
                for data in self.dataloaders[phase]:
                    # получаем картинки и метки
                    inputs, labels = data

                    # оборачиваем в переменные
                    if self.use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # инициализируем градиенты параметров
                    if phase == "train":
                        optimizer.zero_grad()

                    # forward pass
                    if phase == "eval":
                        with torch.no_grad():
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                    preds = torch.argmax(outputs, -1)
                    loss = criterion(outputs, labels)

                    # backward pass + оптимизируем только если это стадия обучения
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # статистика
                    running_loss += loss.item()
                    running_corrects += int(torch.sum(preds == labels.data))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                losses[phase].append(epoch_loss)

                pbar.set_description('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc
                ))

                # если достиглось лучшее качество, то запомним веса модели
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # загрузим лучшие веса модели
        model.load_state_dict(best_model_wts)
        return model, losses

    def train(self):
        # Использовать ли GPU
        if self.use_gpu:
            self.model_extractor = self.model_extractor.cuda()
            # В качестве cost function используем кросс-энтропию
        loss_fn = nn.CrossEntropyLoss()
        # Обучаем только классификатор
        optimizer = optim.Adam(self.model_extractor.fc.parameters(), lr=1e-4)
        self.model_extractor, losses = self.train_model(self.model_extractor, criterion=loss_fn, optimizer=optimizer,
                                                        ac_loader=self.ac_loader, num_epochs=self.num_epochs)
        return self.model_extractor, losses


if __name__ == "__main__":
    # Загрузка нового класса изображений
    ac_loader = ACLoaderDataset(size_img=200, batch_size=10, num_workers=0, use_gpu=False)
    path_load = ac_loader.add_new_class(classname="skateboard", num=800)

    # При добавлении нового класса делается автоматом
    # Подразумевается что в каталоге dataset/train расположены тернировочные датасеты
    # Но для моедли надо отдельно выделить валидационные данные
    # Формирвоание Валидацинных данных. Если в dataset/val нет каталога, который есть в dataset/train,
    # то производим добавление соответсвтующего класса и переносим часть данных как валидационных
    # ac_loader = ACLoaderDataset(size_img=200, use_gpu=False)
    # ac_loader.create_val_dataset(val_size=0.1)

    # Начальное формирование модели по тем классам по которым есть данные
    ac_model = ACModel(size_img=200, batch_size=10, num_epochs=2, check_gpu=False, num_workers=0)
    # Модель формируется на основе предобученной ResNet50 но уже с новыми данными
    ac_model.new_model()
    # Сохраняем модель + сохраняем классы по которым считались
    ac_model.save_model()

    # Прогнозирование по существующей моделе, Важно чтобы size_img был точно таким же как и при обучении
    load_ac_model = ACModel(size_img=200, check_gpu=False)
    # load_ac_model = ACModel(size_img=200, batch_size = 10, num_epochs=1, check_gpu=False, num_workers=0)
    load_ac_model.load_model()
    detected_files = load_ac_model.predict_imagefiles()
    print(detected_files)
