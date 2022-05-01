import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

import re
# import torch.nn.functional as F
# from torch import optim
from torchvision.io import read_image

# from torchvision import models
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TrainSet(Dataset):

    def __init__(self, dir):
        self.train_dir = Path(dir)
        self.train_files_path = sorted(list(self.train_dir.rglob('*.jpg')))  # пути к файлам трейна
        self.train_class_names = [path.parent.name for path in self.train_files_path]  # имена классов для файлов трейна

        class_paths = sorted(list(self.train_dir.glob('*')))
        classes = [path.name for path in class_paths]  # список имен классов
        self.class_id = {classes[i]: i for i in range(len(classes))}  # словарик с класс айди

    def __len__(self):
        return len(self.train_files_path)

    def __getitem__(self, index):
        img = read_image(str(self.train_files_path[index]))
        img = img / 255.0
        img = transform(img)

        name = self.train_class_names[index]
        img_y = self.class_id[name]

        return img, img_y


class ValSet(Dataset):

    def __init__(self, dir):
        self.train_dir = Path(dir)
        self.test_dir = Path('simpsons/kaggle_simpson_testset')
        self.test_files_path = sorted(list(self.test_dir.rglob('*.jpg')))  # пути к файлам теста
        test_classes = [path.name for path in self.test_files_path]
        self.test_class_names = [re.sub(r"_\d+.jpg", "", path) for path in test_classes]  # имена классов для теста

        class_paths = sorted(list(self.train_dir.glob('*')))
        classes = [path.name for path in class_paths]  # список имен классов
        self.class_id = {classes[i]: i for i in range(len(classes))}  # словарик с класс айди

    def __len__(self):
        return len(self.test_files_path)

    def __getitem__(self, index):
        img = read_image(str(self.test_files_path[index]))
        img = img / 255.0
        img = transform(img)

        name = self.test_class_names[index]
        img_y = self.class_id[name]

        return img, img_y


def loss_batch(model, loss_func, xb, yb, opt=None):
    predict = model(xb)
    loss = loss_func(predict, yb)
    predict = [torch.argmax(pred) for pred in predict]
    predict = torch.stack(predict, 0)
    trueCount = 0
    for i in range(len(yb)):
        if predict[i] == yb[i]:
            trueCount += 1

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), trueCount / len(yb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    acc_val = []
    loss_val = []
    acc_train = []
    for epoch in range(epochs):
        model.train()
        _, numsT, acc_train = zip(*[loss_batch(model, loss_func, xb.cuda(), yb.cuda(), opt) for xb, yb in train_dl])

        model.eval()
        with torch.no_grad():
            losses, nums, acc_val = zip(
                *[loss_batch(model, loss_func, xb.cuda(), yb.cuda()) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        accuracy_val = np.sum(np.multiply(acc_val, nums)) / np.sum(nums)
        accuracy_train = np.sum(np.multiply(acc_train, numsT)) / np.sum(numsT)

        acc_val.append(accuracy_val)
        loss_val.append(val_loss)
        acc_train.append(accuracy_train)

    return acc_val, loss_val, acc_train
