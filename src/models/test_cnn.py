import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN

# --------------------------------------------------
# 1. Load CIFAR-100
# --------------------------------------------------

dataset = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

images, labels = next(iter(loader))


# --------------------------------------------------
# 2. Select one random image
# --------------------------------------------------

index = torch.randint(len(dataset), (1,)).item()

image, label = dataset[30]


# [3, 32, 32] -> [1, 3, 32, 32]
x = image.unsqueeze(0)


# --------------------------------------------------
# 3. Create model
# --------------------------------------------------

model = SimpleCNN()
loss = nn.CrossEntropyLoss()

optim = torch.optim.SGD(params=model.parameters(), lr=.001)

output_logit = model(images)

optim.zero_grad()

loss_val = loss(output_logit, labels)



# --------------------------------------------------
# 4. Forward pass, one layer at a time
# --------------------------------------------------

with torch.no_grad():

    # Conv 1
    conv1 = model.layers[0](x)
    relu1 = model.layers[1](conv1)
    pool1 = model.layers[2](relu1)

    # Conv 2
    conv2 = model.layers[3](pool1)
    relu2 = model.layers[4](conv2)
    pool2 = model.layers[5](relu2)

    # Conv 3
    conv3 = model.layers[6](pool2)
    relu3 = model.layers[7](conv3)

    # Global average pooling
    representation = relu3.mean(dim=(2, 3))

    # Final classifier
    logits = model.classifiers(representation)

    # Prediction
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()

