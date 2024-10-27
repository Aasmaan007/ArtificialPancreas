import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

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

# Initialize the model
model = NeuralNetworkModel(3)
model.load_state_dict(torch.load('model_parameters.pth'))
model.eval()

path = "./results/result_all/result_all"
categories = ['adult', 'adolescent', 'child']

for category in categories:
    nanpercentiles_data = []
    for i in tqdm(range(1, 11)):  # Assuming datasets are numbered from 1 to 10
        file = f"{path}/{category}#{i:03d}.csv"
        df = pd.read_csv(file)
        X = df[['BG', 'CGM', 'CHO']].values
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            kp_ki_kd = model(X_tensor)

        kp_values = kp_ki_kd[:, 0].numpy()
        ki_values = kp_ki_kd[:, 1].numpy()
        kd_values = kp_ki_kd[:, 2].numpy()

        # Compute statistics ignoring NaNs
        kp_mean = np.nanmean(kp_values)
        ki_mean = np.nanmean(ki_values)
        kd_mean = np.nanmean(kd_values)

        kp_20th = np.nanpercentile(kp_values, 25)
        kp_80th = np.nanpercentile(kp_values, 75)
        ki_20th = np.nanpercentile(ki_values, 25)
        ki_80th = np.nanpercentile(ki_values, 75)
        kd_20th = np.nanpercentile(kd_values, 25)
        kd_80th = np.nanpercentile(kd_values, 75)
        kp_50th = np.nanpercentile(kp_values, 50)
        ki_50th = np.nanpercentile(ki_values, 50)
        kd_50th = np.nanpercentile(kd_values, 50)

        nanpercentiles_data.append({
            'Patient ID': f'{category}_{i}',
            'KP 20th Percentile': kp_20th,
            'KP 80th Percentile': kp_80th,
            'KI 20th Percentile': ki_20th,
            'KI 80th Percentile': ki_80th,
            'KD 20th Percentile': kd_20th,
            'KD 80th Percentile': kd_80th,
            'KP 50th Percentile': kp_50th,
            'KI 50th Percentile': ki_50th,
            'KD 50th Percentile': kd_50th,
            'KP mean': kp_mean,
            'KI mean': ki_mean,
            'KD mean': kd_mean
        })

    # Calculate mean values for the entire category
    category_mean = pd.DataFrame(nanpercentiles_data).mean().to_dict()
    category_mean['Patient ID'] = f'{category}_mean'
    nanpercentiles_data.append(category_mean)

    # Save to CSV
    nanpercentiles_df = pd.DataFrame(nanpercentiles_data)
    nanpercentiles_df.to_csv(f'{category}_kpkikd.csv', index=False)
