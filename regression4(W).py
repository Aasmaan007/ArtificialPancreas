import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def average_random_csvs(path="./results/result_all/result_all"):
    adolescent_idx = random.randint(1, 10)
    adult_idx = random.randint(1, 10)
    child_idx = random.randint(1, 10)

    adolescent_file = f"{path}/adolescent#{adolescent_idx:03d}.csv"
    adult_file = f"{path}/adult#{adult_idx:03d}.csv"
    child_file = f"{path}/child#{child_idx:03d}.csv"

    adolescent_df = pd.read_csv(adolescent_file)
    adult_df = pd.read_csv(adult_file)
    child_df = pd.read_csv(child_file)

    adolescent_df.drop(adolescent_df.tail(1).index, inplace=True)
    adult_df.drop(adult_df.tail(1).index, inplace=True)
    child_df.drop(child_df.tail(1).index, inplace=True)
    adolescent_df = adolescent_df.drop("Time", axis=1)
    adult_df = adult_df.drop("Time", axis=1)
    child_df = child_df.drop("Time", axis=1)

    average_df = (adolescent_df + adult_df + child_df) / 3
    return average_df

df = average_random_csvs()
X = df[['BG', 'CGM', 'CHO']].values
y = df['insulin'].values

degree = 5
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

X_poly_tensor = torch.tensor(X_poly, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X_poly_tensor, y_tensor, test_size=0.2, random_state=42)

print("Any NaN in X_train:", np.isnan(X_train.numpy()).any())
print("Any NaN in y_train:", np.isnan(y_train.numpy()).any())


class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 3)  # Output Kp, Ki, Kd

    def forward(self, x):
        return self.linear(x)

class PIDController(nn.Module):
    def __init__(self):
        super().__init__()
        self.integral = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.prev_error = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def forward(self, kp, ki, kd, x):
        error = 100 - x
        P = kp * error
        self.integral =  self.integral+ error
        I = ki * self.integral
        D = kd * (error - self.prev_error)
        self.prev_error = error
        return P + I + D

model = PolynomialRegressionModel(X_train.shape[1])
pid = PIDController()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(100):
    for i in range(X_train.size(0)):
        optimizer.zero_grad()
        kp_ki_kd = model(X_train[i].unsqueeze(0))
        insulin_pred = pid(kp_ki_kd[0, 0], kp_ki_kd[0, 1], kp_ki_kd[0, 2], X_train[i, 0])
        loss = criterion(insulin_pred, y_train[i].unsqueeze(0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if i % 10 == 0:  # Print loss periodically for every 10 data points
            print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    mse = mean_squared_error(y_test, [pid(model(X_test[i].unsqueeze(0))[0, 0], model(X_test[i].unsqueeze(0))[0, 1], model(X_test[i].unsqueeze(0))[0, 2], X_test[i, 0]).item() for i in range(X_test.size(0))])
    print(f"Mean Squared Error on Test Set: {mse}")
