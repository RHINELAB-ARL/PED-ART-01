import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.metrics import classification_report, confusion_matrix

batch_size = 32
epochs = 10
learning_rate = 0.001

os.makedirs("data", exist_ok=True)

info = INFO["bloodmnist"]
DataClass = getattr(medmnist, info["python_class"])

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

train_data = DataClass(split="train", root="data", download=True, as_rgb=True, transform=transform)
test_data = DataClass(split="test", root="data", download=True, as_rgb=True, transform=transform)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


model = CNN(len(info["label"]))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for x, y in train_dataloader:
        y = y.squeeze().long()
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_dataloader):.6f}")

model.eval()
y_pred = []
y_test = []

with torch.no_grad():
    for x, y in test_dataloader:
        predicted = model(x).argmax(dim=1)
        y_pred.extend(predicted.numpy())
        y_test.extend(y.squeeze().numpy())

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(info["label"].values())))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))