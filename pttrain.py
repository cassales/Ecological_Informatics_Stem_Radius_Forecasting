import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Custom loss functions
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))

# Set parameters
epochs = 1
records_per_hour = 12
learning_rate = 0.001
forecast_hours = 24
lookback_hours = 10
shuffle = False
lookback = lookback_hours * records_per_hour
forecast = forecast_hours * records_per_hour
activation = 'tanh'
early_stop_patience = 20
interp_type = 'lin'
scion_sensor = ''
temporal_res = 5

input_directory = './exper-5min/'
interp_files_dir = './interp_sensors_5min/'

# Find min and max values among all sensors
mini = 9999999
maxi = -9999999
for f in os.listdir(interp_files_dir):
    sensorfile = f.split('-')[0]
    df = pd.read_csv(f"{interp_files_dir}/{f}", index_col=0)
    series = np.array(df['Value_c'].values)
    mini = min(mini, series.min())
    maxi = max(maxi, series.max())

print("got min and max among all sensors", mini, maxi)

# Use your utility function to create sequence file
import myutilstrain as mut
mut.create_sequence_file(interp_files_dir, input_directory, 'HALF', lookback, forecast, f'Ashley_HALF_{lookback}_{temporal_res}.csv', mini, maxi, temporal_res)

# Load the data
df = pd.read_csv('./exper-5min/Ashley_HALF_120_5.csv')
df.info()

# Prepare the data
dataLen = len(df)
X = np.array(df.iloc[:, :-1])
Y = np.array(df.iloc[:, -1].values)

print('X', X.shape, type(X), '\nY', Y.shape, type(Y))

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(2)  # Add a channel dimension
Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)  # Reshape to (num_samples, 1)

# Create DataLoader
batch_size = 256
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the PyTorch model
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * ((lookback - 4) // 4), 64)  # Adjust for Conv1D output size
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TimeSeriesCNN(input_channels=1)  # Input has 1 channel

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}')

# Save the model
torch.save(model.state_dict(), f'./CNN_Ashley_HALF_{lookback}_{temporal_res}min.pth')
