import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

x, y = make_circles(n_samples=10000, noise=0.05, random_state=26)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=26)

class Data(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

input_dim = 2
hidden_dim = 10
output_dim = 1
batch_size = 64

train_data = Data(x_train, y_train)
test_data = Data(x_test, y_test)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        return torch.sigmoid(self.layer_2(x)).squeeze(1)

model = Net(input_dim, hidden_dim, output_dim)
print(model)

learning_rate = 0.1
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for x, Y in train_dataloader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader):.6f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, Y in test_dataloader:
        outputs = model(x)
        predicted = (outputs >= 0.5).float()
        correct += (predicted == Y).sum().item()
        total += Y.size(0)

print("Training Complete")
print(f"Accuracy of the network on {total} test instances: {100 * correct / total:.2f}%")