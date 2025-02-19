import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, input_channels_number):
        super(RegressionModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_channels_number, out_features=input_channels_number * 4),
            nn.ReLU(),
            nn.Linear(in_features=input_channels_number * 4, out_features=1)
        )

    def forward(self, x):
        return self.classifier(x)
