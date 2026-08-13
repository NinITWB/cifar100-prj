import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

dataset = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

loader = DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True
)

images, labels = next(iter(loader))
