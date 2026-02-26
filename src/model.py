import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, 32, 32]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # -> [N, 16, 16, 16]
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # -> [N, 32, 8, 8]
        x = torch.flatten(x, 1)  # -> [N, 32*8*8]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
