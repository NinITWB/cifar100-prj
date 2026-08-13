import torch
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.feature_layers = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(256, 100)

    def forward(self, x: torch.Tensor):

        x = self.flatten(x)

        representation = self.feature_layers(x)

        logits = self.classifier(representation)

        return logits, representation
