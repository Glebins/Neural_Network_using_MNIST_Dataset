import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from mnist import MNIST

dataset = MNIST('C:/Users/79386/Programming/Data_science/Datasets/mnist')
train_images, train_labels = dataset.load_training()
test_images, test_labels = dataset.load_testing()
train_images, train_labels, test_images, test_labels = \
    torch.tensor(train_images), torch.tensor(train_labels), torch.tensor(test_images), torch.tensor(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(-1, 1, 28, 28)

batch_size = 64

train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=batch_size, shuffle=True)

custom_model = nn.Sequential(
    nn.BatchNorm2d(1),

    nn.Conv2d(1, 4, 3, padding=1),
    nn.SELU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(4, 12, 3, padding=1),
    nn.SELU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(12, 24, 3, padding=1),
    nn.SELU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(24,48, 3, padding=1),
    nn.SELU(),

    nn.Flatten(),
    nn.Linear(48 * 3 * 3, 240),
    nn.SELU(),

    nn.Linear(240, 120),
    nn.SELU(),

    nn.Linear(120, 10)
)

def get_param_number(model):
    param_number = 0
    for param in model.parameters():
        param_number += torch.tensor(param.shape).prod()

    return param_number

print(get_param_number(custom_model))

def train(model: nn.Sequential, train_loader: DataLoader):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.train()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_loss = 0
    train_accuracy = 0

    for X_batch, y_batch in train_loader:
        X_batch.to(device)
        y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(X_batch)
        loss_value = criterion(predictions, y_batch)

        loss_value.backward()
        optimizer.step()





