import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from PIL import Image

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dataset = MNIST('C:/Users/nedob/Programming/Data Science/Datasets/MNIST_handwritten_numbers/archive')
# dataset = MNIST('C:/Users/79386/Programming/Data_science/Datasets/mnist')

train_images, train_labels = dataset.load_training()
test_images, test_labels = dataset.load_testing()
train_images, train_labels, test_images, test_labels = \
    torch.tensor(train_images), torch.tensor(train_labels), torch.tensor(test_images), torch.tensor(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(-1, 1, 28, 28)

batch_size = 64
use_transforms = True

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.08, 0.08)),
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])


class TransformTensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]

        return img, label

if use_transforms:
    train_set = TransformTensorDataset(train_images, train_labels, train_transform)
    test_set = TransformTensorDataset(test_images, test_labels, test_transform)

    train_loader_mnist = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader_mnist = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

else:
    train_loader_mnist = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader_mnist = DataLoader(TensorDataset(test_images, test_labels), batch_size=batch_size, shuffle=False, pin_memory=True)

path_to_weights = "./nn_weights" if use_transforms else "./nn_weights_without_transforms"

custom_model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Dropout(0.1),
    nn.Linear(96 * 3 * 3, 256), nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 10)
)


def get_param_number(model):
    param_number = 0
    for param in model.parameters():
        param_number += torch.tensor(param.shape).prod().item()

    return param_number


print(get_param_number(custom_model))


def get_test_accuracy(model: nn.Sequential, test_loader: DataLoader):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    right_answers = 0

    model = model.to(device)
    model.eval()

    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predicts = model(X_batch)

        right_answers += (predicts.argmax(1) == y_batch).sum().item()

    return right_answers / len(test_loader.dataset)


def train(model: nn.Sequential, train_loader: DataLoader, test_loader: DataLoader, num_epochs=30):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        right_answers = 0
        current_loss = 0

        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss_value = criterion(predictions, y_batch)

            loss_value.backward()
            optimizer.step()

            right_answers += (predictions.argmax(1) == y_batch).sum().item()
            current_loss += loss_value.item() * len(y_batch)

        model.eval()
        with torch.no_grad():
            current_train_accuracy = right_answers / len(train_loader.dataset)
            current_train_loss = current_loss / len(train_loader.dataset)
            current_test_accuracy = get_test_accuracy(model, test_loader)

            train_loss_history.append(current_train_loss)
            train_accuracy_history.append(current_train_accuracy)
            test_accuracy_history.append(current_test_accuracy)

            print(f"#{epoch}. {current_train_loss}: {current_train_accuracy}, {current_test_accuracy}")

            path_to_save_weights = path_to_weights
            torch.save(custom_model.state_dict(), f"{path_to_save_weights}/{epoch}.pth")

    return train_loss_history, train_accuracy_history, test_accuracy_history


is_training = False

if is_training:
    loss, train_acc, test_acc = train(custom_model, train_loader_mnist, test_loader_mnist, num_epochs=50)

    fig, (axes_1, axes_2) = plt.subplots(1, 2)

    axes_1.plot(loss)
    axes_1.grid()

    axes_2.plot(train_acc, label="Train")
    axes_2.plot(test_acc, label="Test")
    axes_2.legend()
    axes_2.grid()

    plt.tight_layout()
    plt.show()

else:
    custom_model.load_state_dict(torch.load(f"{path_to_weights}/49.pth", weights_only=True))

    test_acc = get_test_accuracy(custom_model, test_loader_mnist)
    print(test_acc)
