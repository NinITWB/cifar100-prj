import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_dataset = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.CIFAR100(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=True
)
