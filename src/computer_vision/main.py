from models import SimpleCNN
from models import SimpleMLP
from torch import nn
from data.data_manipulation import *
import torch
from tqdm import tqdm

def train_cnn(cnn, features: torch.Tensor, labels, loss_func, optimizer):
    cnn.train()
    optimizer.zero_grad()
    feat_return = cnn(features)
    logits, representation = feat_return
    loss_val = loss_func(logits, labels)

    loss_val.backward()
    optimizer.step()
    return loss_val.item()


def main():
    cnn = SimpleCNN()

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        params=cnn.parameters(),
        lr=0.01
    )
    epoch = 10
    for i in range(epoch):
        total_loss = 0.0
        for idx, (features, labels) in tqdm(enumerate(train_loader)):
            loss = train_cnn(cnn, features, labels, loss_func=loss_func, optimizer=optimizer)

            total_loss += loss
        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {i + 1}/{epoch} "
            f"Loss: {average_loss:.4f}"
        )

if __name__ == "__main__":
    main()
