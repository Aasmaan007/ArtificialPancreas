import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random

# Function to average random CSVs
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
    
    adolescent_df = adolescent_df.iloc[:-1].drop("Time", axis=1)
    adult_df = adult_df.iloc[:-1].drop("Time", axis=1)
    child_df = child_df.iloc[:-1].drop("Time", axis=1)
    
    average_df = (adolescent_df + adult_df + child_df) / 3
    return average_df

# Polynomial Regression Model
class PolynomialRegression:
    def __init__(self, degree=2):
        self.poly = PolynomialFeatures(degree)
        self.weights = None

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.weights = np.linalg.pinv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
        
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return X_poly.dot(self.weights)

# PID Controller
class PIDController:
    def __init__(self):
        self.integral = 0
        self.prev_error = 0
    
    def compute(self, kp, ki, kd, x):
        error = 100 - x  # Target is 100
        self.integral += error
        derivative = error - self.prev_error
        output = kp * error + ki * self.integral + kd * derivative
        self.prev_error = error
        return output

# Load data
df = average_random_csvs()
X = df[['BG', 'CGM', 'CHO']].values
y = df['insulin'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial regression
degree = 5
model = PolynomialRegression(degree)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# Simulate PID Controller with the model
pid = PIDController()
kp, ki, kd = 0.1, 0.01, 0.005  # Example PID coefficients
for i in range(len(X_test)):
    bg = X_test[i][0]  # Assume BG is the first feature
    control_output = pid.compute(kp, ki, kd, bg)
    print(f"Control Output for Test {i+1}: {control_output}")
