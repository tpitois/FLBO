from torch import nn
import torch.nn.functional as F

from src.models.layers import ACSConv


class ACSCNN(nn.Module):
    def __init__(self, n_desc, n_class):
        super().__init__()
        self.n_desc = n_desc
        self.n_class = n_class

        self.conv1 = ACSConv(n_desc, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = ACSConv(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = ACSConv(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = ACSConv(64, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = ACSConv(64, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = ACSConv(64, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, n_class)

    def forward(self, x, L):
        x = F.relu(self.bn1(self.conv1(x, L)))
        x = F.relu(self.bn2(self.conv2(x, L)))
        x = F.relu(self.bn3(self.conv3(x, L)))
        x = F.relu(self.bn4(self.conv4(x, L)))
        x = F.relu(self.bn5(self.conv5(x, L)))
        x = F.relu(self.bn6(self.conv6(x, L)))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)

        return self.fc3(x)