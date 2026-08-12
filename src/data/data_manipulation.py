from torchvision import datasets

dataset = datasets.CIFAR100(
    root="data",
    train=True,
    download=True
)

img, label = dataset[0]

print(img.shape)