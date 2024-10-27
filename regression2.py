import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Assuming all necessary imports are done correctly

# Utility function to load and average data
def average_random_csvs(path="./results/result_all/result_all"):
       # Generate random indices to pick from each group
    adolescent_idx = random.randint(1, 10)
    adult_idx = random.randint(1, 10)
    child_idx = random.randint(1, 10)
    
    
    # Create filenames based on the indices
    adolescent_file = f"{path}/adolescent#{adolescent_idx:03d}.csv"
    adult_file = f"{path}/adult#{adult_idx:03d}.csv"
    child_file = f"{path}/child#{child_idx:03d}.csv"
    
    # Read the CSVs into DataFrames
    adolescent_df = pd.read_csv(adolescent_file)
    adult_df = pd.read_csv(adult_file)
    child_df = pd.read_csv(child_file)
    
    # Ensure that the DataFrames have the same shape
    if not (adolescent_df.shape == adult_df.shape == child_df.shape):
        raise ValueError("The CSV files do not have the same shape. Ensure all CSVs have identical columns and rows.")
    adolescent_df.drop(adolescent_df.tail(1).index,inplace=True)
    adult_df.drop(adult_df.tail(1).index,inplace=True)
    child_df.drop(child_df.tail(1).index,inplace=True)
    adolescent_df = adolescent_df.drop("Time", axis = 1)
    adult_df = adult_df.drop("Time", axis = 1)
    child_df = child_df.drop("Time", axis = 1)

    # Compute the average of the DataFrames
    average_df = (adolescent_df + adult_df + child_df) / 3
    # average_df = 
    
    return average_df
# Load and prepare dataset
df = average_random_csvs()
X = df[['BG', 'CGM', 'CHO']].values  # Using values directly for transformation
y = df['insulin'].values

# Polynomial transformation
degree = 5
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)
X_poly_tensor = torch.tensor(X_poly, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the model
class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 3)  # Output Kp, Ki, Kd

    def forward(self, x):
        return self.linear(x)

# Define PID controller using tensor operations
class PIDController(nn.Module):
    def __init__(self):
        super().__init__()
        self.integral = torch.zeros(1, dtype=torch.float32)
        self.prev_error = torch.zeros(1, dtype=torch.float32)

    def forward(self, kp, ki, kd, x):
        error = 100 - x
        P = kp * error
        self.integral += error
        I = ki * self.integral
        D = kd * (error - self.prev_error) if self.prev_error is not None else torch.zeros_like(error)
        self.prev_error = error
        return P + I + D

# Model setup
model = PolynomialRegressionModel(X_train.shape[1])
pid = PIDController()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    kp_ki_kd = model(X_train)
    insulin_preds = pid(kp_ki_kd[:, 0], kp_ki_kd[:, 1], kp_ki_kd[:, 2], X_train[:, 0])  # Assuming x input as 'BG' only
    loss = criterion(insulin_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    kp_ki_kd = model(X_test)
    insulin_preds = [pid(kp_ki_kd[i, 0], kp_ki_kd[i, 1], kp_ki_kd[i, 2], X_test[i, 0]) for i in range(len(X_test))]
    mse = mean_squared_error(y_test, insulin_preds)
    print(f"Mean Squared Error on Test Set: {mse}")
