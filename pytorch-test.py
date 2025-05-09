import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim



# Set the random seed for reproducibility
# This is important for debugging and testing purposes.
# For testing and debugging we must remove all randomness and non-deterministic algorithms.
# See https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
RANDOM_SEED = 42
print("*********** TEST MODE ***********")
print("     disabling all randomness")
print(f"     RANDOM_SEED={RANDOM_SEED}")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)
# torch.use_deterministic_algorithms(True)  # 2 = error, 1 = warn, 0 = quiet (default)
# The above line does not work with PyTorch 1.4.0
# AttributeError: module 'torch' has no attribute 'use_deterministic_algorithms'
print("*********** TEST MODE ***********")


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 1
hidden_size = 50
output_size = 1
num_epochs = 100
learning_rate = 0.01

# Generate some dummy data
x_train = torch.linspace(0, 799, 800).reshape(-1, 1)
y_train = torch.sin(x_train * 0.01)

# Reshape data for LSTM [batch_size, sequence_length, input_size]
x_train = x_train.view(-1, 10, 1)
y_train = y_train.view(-1, 10, 1)

print(f"x_train:\n{x_train}")
print(f"y_train:\n{y_train}")

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print(f"pytorch version: {torch.__version__}")
for epoch in range(num_epochs):
    model.train()
    outputs = model(x_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train[:, -1, :])
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1:3d}/{num_epochs:3d} Loss: {loss.item():.8f}')

# Prediction loop
model.eval()
with torch.no_grad():
    test_input = torch.linspace(800, 899, 100).reshape(-1, 10, 1)
    test_output = model(test_input)
    torch.set_printoptions(precision=8)
    print(f"Test Output:\n{test_output}")