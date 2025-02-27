from torch import nn
from non_local_embedded_gaussian import NONLocalBlock1D


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.nl_1 = NONLocalBlock1D(in_channels=32)
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.nl_2 = NONLocalBlock1D(in_channels=64)
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64000,out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        feature_1 = self.conv_1(x)

        nl_feature_1, w_1 = self.nl_1(feature_1)

        feature_2 = self.conv_2(nl_feature_1)
        nl_feature_2, w_2 = self.nl_2(feature_2)

        output = self.conv_3(nl_feature_2).view(batch_size, -1)
        output1 = self.fc1(output)
        output2 = self.fc2(output1)
        return output2, w_1, w_2


if __name__ == '__main__':
    import torch


