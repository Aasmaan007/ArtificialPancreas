import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

# df = average_random_csvs()    

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.output_layer = nn.Linear(64, 3)  # Output Kp, Ki, Kd
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        return self.output_layer(x)

class PIDController(nn.Module):
    def __init__(self):
        super().__init__()
        self.integral = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.prev_error = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def forward(self, kp, ki, kd, p , i , d):
        # error = 100 - x

        P = kp * p
        # self.integral = self.integral + error
        I = ki * i
        D = kd * d
        temp = P+I+D
       
        # self.prev_error = error
        return P + I + D

model = NeuralNetworkModel(3)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001 , momentum = 0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

path="./results/result_all/result_all"

for i in tqdm(range(1, 11)):
    adolescent_file = f"{path}/adolescent#{i:03d}.csv"
    adult_file = f"{path}/adult#{i:03d}.csv"
    child_file = f"{path}/child#{i:03d}.csv"

    adolescent_df = pd.read_csv(adolescent_file)
    adult_df = pd.read_csv(adult_file)
    child_df = pd.read_csv(child_file)
    flag = 0
    for df in [adolescent_df, adult_df, child_df]:
        flag = (flag+1)%3
        X = df[['BG', 'CGM', 'CHO']].values
        y = df['insulin'].values

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42 , shuffle=False)
        P_train = 100-X_train[:,0]
        I_train = torch.cumsum(100-X_train[:,0], dim=0)
        D_train = X_train[:-1 ,0]-X_train[1: ,0] 


# Optionally, add a zero row at the start if you want to keep the same shape and indexing
        # D_train = D_train.unsqueeze(1)  # Make D_train two-dimensional

# Add a zero row at the start to keep the same shape as X_train
        zeros = torch.zeros(1)  # Ensure the zero tensor is also two-dimensional
        D_train = torch.cat([zeros, D_train], dim=0)
        X_train1,X_train2,X_train3,y_train1,y_train2,y_train3 = X_train[:512], X_train[512:1024], X_train[1024:], y_train[:512], y_train[512:1024], y_train[1024:]
        X_train_final = [X_train1,X_train2,X_train3]
        y_train_final = [y_train1,y_train2,y_train3]
        P_train1,P_train2,P_train3 = P_train[:512], P_train[512:1024], P_train[1024:]
        I_train1,I_train2,I_train3 = I_train[:512], I_train[512:1024], I_train[1024:]
        D_train1,D_train2,D_train3 = D_train[:512], D_train[512:1024], D_train[1024:]
        P_train_final = [P_train1,P_train2,P_train3]
        I_train_final = [I_train1,I_train2,I_train3]
        D_train_final = [D_train1,D_train2,D_train3]

        for epoch in tqdm(range(1000)):
            pid = PIDController()
            for j in range(3):
                optimizer.zero_grad()
                kp_ki_kd = model(X_train_final[j])
                # print(kp_ki_kd.shape)
                # print(P_train_final[j].shape)
                insulin_pred = pid(kp_ki_kd[:, 0], kp_ki_kd[:, 1], kp_ki_kd[:, 2], P_train_final[j] , I_train_final[j], D_train_final[j] )
                # print(insulin_pred.shape)
                # print(y_train_final[j].shape)
                loss = criterion(insulin_pred, y_train_final[j])
                loss.backward()
                total_norm_before = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
                # print(f'Epoch {epoch + 1}, Total gradient norm before clipping: {total_norm_before}')



                total_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                # print(f'Epoch {epoch + 1}, Total gradient norm after clipping: {total_norm_after}')
                optimizer.step()
            # print(f"Index: {i}, file: {flag}, Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    print(f"Index: {i}, Loss: {loss.item():.4f}")
# model.eval()
# with torch.no_grad():
#     mse = mean_squared_error(y_test, [pid(model(X_test[i].unsqueeze(0))[0, 0], model(X_test[i].unsqueeze(0))[0, 1], model(X_test[i].unsqueeze(0))[0, 2], X_test[i, 0]).item() for i in range(X_test.size(0))])
#     print(f"Mean Squared Error on Test Set: {mse}")
torch.save(model.state_dict(), 'model_parameters.pth')
torch.save(model, 'complete_model.pth')


