
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    


results_df = pd.read_csv("./results/mlp_ReLUMLP_dropout0.5_2.csv", header=0)
predictions = results_df["Predicted"].to_numpy()
actual_values = results_df["Actual"].to_numpy()

mse = np.mean((predictions - actual_values) ** 2)
mae = np.mean(np.abs(predictions - actual_values))

print(f"\nTest Results:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

print(actual_values.shape)
print(predictions.shape)
# Create scatter plots
fig, axis = plt.subplots(1, 3, figsize=(12, 4))

actual_values_unique = np.unique(actual_values)
bias = []
sd = []

for true_value in actual_values_unique:
    mask = actual_values==true_value
    cur_predictions = predictions[mask]
    cur_true = actual_values[mask]
    cur_bias =  np.mean(cur_predictions - cur_true)
    cur_sd = np.std(cur_predictions, ddof=1)
    bias.append(cur_bias)
    sd.append(cur_sd)
bias = np.array(bias)
sd = np.array(sd)
rmse = (sd**2 + bias**2)**(1/2)
    

axis[0].scatter(actual_values_unique, sd, color="blue", alpha=0.5)
axis[0].axhline(y=np.min(sd), color='k', linestyle='--', alpha=0.3)
axis[0].set_xlabel('True Values')
axis[0].set_ylabel('Variability')

axis[1].scatter(actual_values_unique, bias, color="red", alpha=0.5)
axis[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axis[1].set_xlabel('True Values')
axis[1].set_ylabel('Average Bias')

axis[2].scatter(actual_values_unique, rmse, color="orange", alpha=0.5)
axis[2].set_xlabel('True Values')
axis[2].set_ylabel('RMSE')

fig.suptitle(f"mlp_ReLUMLP_dropout0.5_2", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'./plots/mlp_ReLUMLP_dropout0.5_2.png')
plt.close()
