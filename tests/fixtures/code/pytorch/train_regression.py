"""Regression training with PyTorch"""

import torch.nn as nn
import torch.optim as optim


class RegressionModel(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, train_data, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(train_data["X"])
        loss = criterion(predictions, train_data["y"])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

    return model
