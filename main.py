import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
#LSTM
import torch.nn as nn
import torch.optim as optim
#IG
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
#Occ
from captum.attr import Occlusion
#evaluation
from captum.metrics import sensitivity_max, infidelity

# downloading data, S&P 500 and the fear index (VIX), from 2019 to 2024
tickers = ["^GSPC", "^VIX"]
data = yf.download(tickers, start="2019-01-01", end="2024-01-01")['Close'].dropna()

# scaling (model sensitive to large numbers)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 30-day lookbak window
seq_length = 30
xs, ys = [], []

for i in range(len(scaled_data) - seq_length):
    xs.append(scaled_data[i:(i + seq_length)])
    ys.append(scaled_data[i + seq_length, 0])

    """
    # Attempt to fix the nuanced naive persistence predictor: the target is the difference between tomorrow's price and today's price
    today_price = scaled_darta[i + seq_length - 1, 0]
    tomorrow_price = scaled_data[i + seq_length, 0]
    y = tomorrow_price - today_price 
    """

X, y = np.array(xs), np.array(ys)

# train test split (80% train, 20% test)
# we don't shuffle stock data because the order matters and past cannot be predicted with the future
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# convert to pytorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float().unsqueeze(1)
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float().unsqueeze(1)

print(f"Data prep done")
print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Testing shapes: X={X_test.shape}, y={y_test.shape}")



#----------Long Short-Term Memory (LSTM)

# define the LSTM
class StockPredictorLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# initialize the model, loss function, and optimizer
model = StockPredictorLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training loop
epochs = 100

for epoch in range(epochs):
    model.train() # set model to training mode
    
    # forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print progress
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {round(loss.item(), 6)}')

# test evaluation
model.eval() # set model to evaluation mode
print(f"Final Test Loss: {round(criterion(model(X_test), y_test).item(),6)}")



#----------Integrated Gradients (IG)

# pick a single test sample to explain
sample_idx = -1
# shape becomes (1, 30, 2)
input_seq = X_test[sample_idx].unsqueeze(0).requires_grad_()

# define the baseline
# sequence of all zeros, since our data is scaled [0,1]
baseline = torch.zeros_like(input_seq)

# initialize the IG explainer
ig = IntegratedGradients(model)

# calculate attributions
# target=0 because our output layer only has 1 neuron at index 0
attributions, delta = ig.attribute(inputs=input_seq, baselines=baseline, target=0, return_convergence_delta=True)

prediction = model(input_seq).item()
print(f"Model prediction for this sample: {round(prediction, 4)}")

# visualize the attributions
# convert the tensor to a numpy array for matplotlib
attr_np = attributions.squeeze(0).detach().numpy()

plt.figure(figsize=(12, 6))
# plot S&P 500 attributions (Index 0)
plt.plot(attr_np[:, 0], label='S&P 500 Importance', color='blue', marker='o')
# plot VIX attributions (Index 1)
plt.plot(attr_np[:, 1], label='VIX Importance', color='red', marker='x')

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title(f"Integrated Gradients")
plt.xlabel("Days in Lookback Window (0 = 30 days ago, 29 = Yesterday)")
plt.ylabel("Attribution score (impact on prediction)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



#----------Occlusion

occlusion = Occlusion(model)

# sliding window = (5, 1) to support the temporal structure
# strides = 1 ensures we check every day and every feature
attr_occ = occlusion.attribute(
    input_seq, 
    sliding_window_shapes=(5, 1), 
    strides=1, 
    target=0
)

attr_occ_np = attr_occ.squeeze(0).detach().numpy()

plt.figure(figsize=(12, 6))
plt.plot(attr_occ_np[:, 0], label='S&P 500 Importance', color='blue', marker="o")
plt.plot(attr_occ_np[:, 1], label='VIX Importance', color='red', marker="x")

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Occlusion")
plt.xlabel("Days in Lookback Window (0 = 30 days ago, 29 = Yesterday)")
plt.ylabel("Attribution score (impact on prediction)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# check the raw values
print(f"IG Shape: {attr_np.shape} | Occ Shape: {attr_occ_np.shape}")

# compare the first few values of S&P 500 (showing they are not identical)
print("First 3 values of S&P 500 (IG): ", attr_np[:3, 0])
print("First 3 values of S&P 500 (Occ):", attr_occ_np[:3, 0])



#----------evaluation
subset_size = 50
X_subset = X_test[-subset_size:]

# helper function for occlusion compatibility
def occlusion_attr_fn(inputs, **kwargs):
    return occlusion.attribute(inputs, sliding_window_shapes=(5, 1), strides=1, **kwargs)

# recalculate attributions
attr_ig_subset = ig.attribute(X_subset, target=0)
attr_occ_subset = occlusion.attribute(X_subset, sliding_window_shapes=(5, 1), strides=1, target=0)

# calculate metrics for each sample in the subset
# sensitivity: measure how much the explanation changes with tiny input noise (for robustness)
ig_sens_points = sensitivity_max(ig.attribute, X_subset, target=0)
occ_sens_points = sensitivity_max(occlusion_attr_fn, X_subset, target=0)

# infidelity: measure how well the explanation predicts the model's change (for faithfulness)
def perturb_fn(inputs):
    noise = torch.randn_like(inputs) * 0.01 
    return noise, inputs - noise

ig_infid_points = infidelity(model, perturb_fn, X_subset, attr_ig_subset, target=0)
occ_infid_points = infidelity(model, perturb_fn, X_subset, attr_occ_subset, target=0)

# plotting
plt.figure(figsize=(10, 6))

plt.scatter(ig_sens_points.detach().numpy(), ig_infid_points.detach().numpy(), 
            color='blue', label='Integrated Gradients (IG)')
plt.scatter(occ_sens_points.detach().numpy(), occ_infid_points.detach().numpy(), 
            color='red', label='Occlusion (Occ)')

plt.xlabel('Sensitivity (Robustness)')
plt.ylabel('Infidelity (Faithfulness)')
plt.title('Local Evaluations (lower is better on both axes)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

print(f"IG Mean - Sensitivity: {ig_sens_points.mean().item():.6f}, Infidelity: {ig_infid_points.mean().item():.6f}")
print(f"Occ Mean - Sensitivity: {occ_sens_points.mean().item():.6f}, Infidelity: {occ_infid_points.mean().item():.6f}")
