import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Define the model architecture as previously done
class NeuralNetworkModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.output_layer = torch.nn.Linear(64, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        return self.output_layer(x)

# Initialize and load the model
model = NeuralNetworkModel(3)
model.load_state_dict(torch.load('model_parameters.pth'))
model.eval()

path = "./results/result_all/result_all"

for i in tqdm(range(1, 2)):
    files = [f"{path}/adolescent#{i:03d}.csv", f"{path}/adult#{i:03d}.csv", f"{path}/child#{i:03d}.csv"]
    for file in files:
        df = pd.read_csv(file)
        X = df[['BG', 'CGM', 'CHO']].values
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Predict kp, ki, kd using the model
        with torch.no_grad():
            kp_ki_kd = model(X_tensor)

        # Plot distributions of kp, ki, kd
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.histplot(kp_ki_kd[:, 0].numpy(), kde=True, ax=axes[0], color='blue')
        axes[0].set_title(f'Distribution of KP for {file.split("/")[-1]}')
        
        sns.histplot(kp_ki_kd[:, 1].numpy(), kde=True, ax=axes[1], color='green')
        axes[1].set_title(f'Distribution of KI for {file.split("/")[-1]}')
        
        sns.histplot(kp_ki_kd[:, 2].numpy(), kde=True, ax=axes[2], color='red')
        axes[2].set_title(f'Distribution of KD for {file.split("/")[-1]}')
        
        plt.tight_layout()
        plt.show()
