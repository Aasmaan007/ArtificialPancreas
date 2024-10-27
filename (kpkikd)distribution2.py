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

# Assuming model and class are defined and loaded as above
model = NeuralNetworkModel(3)
model.load_state_dict(torch.load('model_parameters.pth'))
model.eval()

path = "./results/result_all/result_all"
categories = ['adolescent', 'adult', 'child']

# Iterate through each category and plot KP distributions
for category in categories:
    kd_values_by_category = []  # This will store KP values for all files in this category

    for i in tqdm(range(1, 11)):  # Assuming 10 datasets per category
        file = f"{path}/{category}#{i:03d}.csv"
        df = pd.read_csv(file)
        X = df[['BG', 'CGM', 'CHO']].values
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Predict kp, ki, kd using the model
        with torch.no_grad():
            kp_ki_kd = model(X_tensor)
        
        # Collect KP values only
        kd_values = kp_ki_kd[:, 2].numpy()
        kd_values_by_category.append(kd_values)

    # Now plot all KP distributions for this category
    fig, axes = plt.subplots(10, 1, figsize=(10, 20), sharex=True)
    fig.suptitle(f'KP Value Distributions for {category.capitalize()}')

    for idx, kp_values in enumerate(kd_values_by_category):
        sns.histplot(kp_values, kde=True, ax=axes[idx], color='skyblue')
        axes[idx].set_title(f'{category.capitalize()} Dataset {idx + 1}')
        axes[idx].set_xlabel('KP Values')
        axes[idx].set_ylabel('Density')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make room for the main title
    plt.show()
