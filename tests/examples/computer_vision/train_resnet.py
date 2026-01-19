import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms


class ResNetTrainer:
    def __init__(
        self, architecture="ResNet50", optimizer_name="Adam", learning_rate=0.001, epochs=10
    ):
        self.architecture = architecture
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        """Load and prepare CIFAR-10 dataset"""
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def create_model(self):
        """Initialize ResNet model"""
        model = models.resnet50(pretrained=True)
        num_classes = 10  # CIFAR-10 has 10 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)
        return model

    def train(self):
        """Train the model"""
        train_loader, test_loader = self.load_data()
        model = self.create_model()

        # Setup optimizer
        if self.optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for _batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_accuracy = 100.0 * correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for _batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            val_accuracy = 100.0 * correct / total

            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_loss / len(train_loader):.4f}, "
                f"Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss / len(test_loader):.4f}, "
                f"Val Acc: {val_accuracy:.2f}%"
            )

        return model


# Example usage
if __name__ == "__main__":
    trainer = ResNetTrainer(optimizer_name="Adam", learning_rate=0.001, epochs=10)
    model = trainer.train()
    torch.save(model.state_dict(), "resnet50_cifar10.pth")
