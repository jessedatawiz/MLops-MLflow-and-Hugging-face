import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Simple linear model

    def forward(self, x):
        return self.linear(x)

# Model, criterion, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy dataset
x_train = torch.tensor([[1.], [2.], [3.]])
y_train = torch.tensor([[2.], [4.], [6.]])

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

import mlflow.pytorch
LOCAL_PORT = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(LOCAL_PORT)
# Log the model 
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, "model")
