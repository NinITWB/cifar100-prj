from torchvision import datasets

dataset = datasets.CIFAR100(
    root="data",
    train=True,
    download=True
)

print("Number of images:", len(dataset))