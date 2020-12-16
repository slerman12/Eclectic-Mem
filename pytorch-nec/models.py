import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
  def __init__(self, embedding_size):
    super(DQN, self).__init__()
    self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
    self.fc = nn.Linear(2592, 256)
    self.head = nn.Linear(256, embedding_size)

  def forward(self, x):
    out = F.relu((self.conv1(x)))
    out = F.relu(self.conv2(out))
    out = F.relu(self.fc(out.view(out.size(0), -1)))
    out = self.head(out)
    return out
