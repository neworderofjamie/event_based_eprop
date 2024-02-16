import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plot_settings
import seaborn as sns 

from pandas import NamedAgg

from itertools import product

from data_utils import load_data_frame

BAR_WIDTH = 1.0
BAR_PAD = 1.1
GROUP_PAD = 2.5

df = load_data_frame("params_*shd*.json", ["surrogate_gradient", "quantization_scale",
                                           "log_quantization", "log_quantization_min"], 
                      20, load_error_transitions=True, load_train=True)

# Filter out non-log
df = df[df["log_quantization"] == True]

# Group data by config and number of layers (later just to ensure column is retained)
df = df.groupby(["surrogate_gradient", "quantization_scale", "log_quantization", "log_quantization_min"], as_index=False, dropna=False)
df = df.agg(mean_num_error_transitions=NamedAgg(column="num_error_transitions", aggfunc=np.mean),
            mean_train_accuracy=NamedAgg(column="train_accuracy", aggfunc=np.mean))

print(f"{len(df)} configurations")

# Get unique quantisation levels
quant_scales = df["quantization_scale"].unique()
quant_mins = df["log_quantization_min"].unique()

# Calculate bar padding
bar_x = np.arange(len(quant_scales)) * BAR_PAD

# Plot accuracy
accuracy_fig, accuracy_axes = plt.subplots(len(quant_mins), sharex=True)
error_rate_fig, error_rate_axes = plt.subplots(len(quant_mins), sharex=True)
for i, q in enumerate(quant_mins):
    accuracy_axes[i].set_title(f"Quantization min {q}")
    error_rate_axes[i].set_title(f"Quantization min {q}")
    
    q_df = df[df["log_quantization_min"] == q]
    
    accuracy_bar_y = []
    error_rate_bar_y = []
    for q in quant_scales:
        frame = q_df[q_df["quantization_scale"] == q]
        if frame.shape[0] == 1:
            accuracy_bar_y.append(frame["mean_train_accuracy"].iloc[0])
            error_rate_bar_y.append(frame["mean_num_error_transitions"].iloc[0])
        else:
            accuracy_bar_y.append(0.0)
            error_rate_bar_y.append(0.0)
    
    
    # Plot bars
    accuracy_axes[i].bar(bar_x, accuracy_bar_y,
                         width=BAR_WIDTH)
    accuracy_axes[i].set_xticks(bar_x, quant_scales.astype(str))
    accuracy_axes[i].set_ylabel("Accuracy [%]")
    
    # Plot bars
    error_rate_axes[i].bar(bar_x, error_rate_bar_y,
                           width=BAR_WIDTH)
    error_rate_axes[i].set_xticks(bar_x, quant_scales.astype(str))
    error_rate_axes[i].set_ylabel("Error rate")

accuracy_axes[-1].set_xlabel("Quantization scale")
error_rate_axes[-1].set_xlabel("Quantization scale")
plt.show()
