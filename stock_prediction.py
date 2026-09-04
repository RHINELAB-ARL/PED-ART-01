import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import datetime as dt
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


ticker = "AAPL"
start_date = "2010-01-01"
train_end_date = "2025-01-01"
prediction_days = 60
epochs = 30
batch_size = 32
learning_rate = 0.001


def preprocess_data(data, prediction_days):
    prices = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)

    x_train = []
    y_train = []

    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train = np.array(x_train).reshape(-1, prediction_days, 1)
    y_train = np.array(y_train).reshape(-1, 1)

    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), scaler


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(1, 50, batch_first=True)
        self.lstm2 = nn.LSTM(50, 50, batch_first=True)
        self.lstm3 = nn.LSTM(50, 50, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)

        x, _ = self.lstm2(x)
        x = self.dropout(x)

        x, _ = self.lstm3(x)
        x = self.dropout(x[:, -1, :])

        return self.fc(x)


def train_model(x_train, y_train):
    model = LSTM()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x, y in dataloader:
            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.6f}")

    return model


def predict_stock_prices(model, train_data, test_data, scaler):
    model.eval()

    train_prices = train_data["Close"].values.reshape(-1, 1)
    test_prices = test_data["Close"].values.reshape(-1, 1)
    prices = np.concatenate([train_prices, test_prices])
    scaled_prices = scaler.transform(prices)

    x_test = []

    for i in range(len(train_prices), len(scaled_prices)):
        x_test.append(scaled_prices[i - prediction_days:i, 0])

    x_test = np.array(x_test).reshape(-1, prediction_days, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(x_test).numpy()

    return scaler.inverse_transform(predictions).ravel()


def predict_tomorrow(model, data, scaler):
    model.eval()

    prices = data["Close"].values.reshape(-1, 1)
    last_days = scaler.transform(prices[-prediction_days:])
    x = torch.tensor(last_days.reshape(1, prediction_days, 1), dtype=torch.float32)

    with torch.no_grad():
        prediction = model(x).numpy()

    return scaler.inverse_transform(prediction)[0][0]


def plot_prices(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, color="black", label="Actual AAPL Prices")
    plt.plot(predicted, color="green", label="Predicted AAPL Prices")
    plt.title("AAPL Share Prices")
    plt.xlabel("Time")
    plt.ylabel("Share Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("AAPL_stock_prediction.png")
    plt.show()


print(f"Training {ticker} LSTM model...")

data_training = yf.download(ticker, start=start_date, end=train_end_date, auto_adjust=True)
data_testing = yf.download(ticker, start=train_end_date, end=dt.datetime.now().strftime("%Y-%m-%d"), auto_adjust=True)

if data_training.empty or data_testing.empty:
    raise ValueError("Could not download the stock data.")

x_train, y_train, scaler = preprocess_data(data_training, prediction_days)
model = train_model(x_train, y_train)

actual_prices = data_testing["Close"].values.ravel()
predicted_prices = predict_stock_prices(model, data_training, data_testing, scaler)

print(f"Actual range: {actual_prices.min():.2f} - {actual_prices.max():.2f}")
print(f"Predicted range: {predicted_prices.min():.2f} - {predicted_prices.max():.2f}")

plot_prices(actual_prices, predicted_prices)

prediction = predict_tomorrow(model, data_testing, scaler)
print(f"Predicted Price for Tomorrow: {prediction:.2f}")