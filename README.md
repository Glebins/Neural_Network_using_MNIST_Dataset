This application recognizes the numbers you draw.
The models were trained on the MNIST dataset.
No additional preprocessing techniques were applied.

|              Model              | Max accuracy, % |
|:-------------------------------:|:---------------:|
| My perceptron with 2[^1] layers |      85.3       |
| My perceptron with 3[^1] layers |      96.3       |
|        KNN (3 neighbors)        |      97.3       |
|               SVM               |      97.9       |
|    Bagging (100 estimators)     |      96.9       |
|    XGBoost (500 estimators)     |      98.1       |
|          Random Forest          |      94.3       |
|          Custom DNN 1           |      99.4       |
|          Custom DNN 2           |      99.6       |
|            Resnet18             |      99.5       |

[^1]: an input layer was taken into account

# Custom DNN architecture

```python
import torch.nn as nn

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
```

DNN 1 has no applied transformations. DNN 2 has the following ones:

```python
from torchvision.transforms import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.08, 0.08)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

During training, the images are randomly rotated by up to 10 degrees and shifted horizontally or vertically by as much
as 8% of their width or height.
Both the training and test images are normalized, using the mean and standard deviation computed across all 60,000
training samples (see analyze_dataset.ipynb for details).
These augmentation and normalization steps help reduce overfitting by effectively increasing the variety of training
examples.

The graphs below show the training results for DNN versions 1 and 2.

![DNN without transforms](training_graphs_and_images/dnn_without_transforms.png (Without transforms))
![DNN with transforms](training_graphs_and_images/dnn_with_transfroms.png (With transforms))

Data transformations help prevent the model from overfitting.
Interestingly, the perceptron shows a nearly monotonic decrease in loss with no signs of overfitting.
This is likely due to its relatively small number of parameters.

![Perceptron](training_graphs_and_images/3l_perceptron.png (3-l perceptron))

For educational purposes, a naked resnet18 model was used. Though it has 200 times more parameters, it behaves the same
as DNN 2 model. Transforms were applied.

![Resnet18](training_graphs_and_images/resnet18_with_transfroms.png (Resnet18))

# Alignment

The problem appeared while I was drawing numbers and giving them to the model. As you may know, numbers in the MNIST
database are aligned at the center. But it's not unlikely that users would draw them in a corner. Despite several
convolution layers in the architecture, the net sometimes recognizes "0" drawn near the top as a "9". This problem can
be solved pretty easily: we should resize the drawn number to match the parameters our net is used to.
Averaged numbers throughout the dataset:

![Averaged numbers](training_graphs_and_images/averaged_numbers.png (Averaged numbers))

For alignment, we will apply the following steps:
- Crop the digit to its bounding box.
- Resize it to 28×28 pixels and add padding.
- Center the image based on its center of mass.
