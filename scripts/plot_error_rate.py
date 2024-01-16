import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plot_settings
import seaborn as sns 

from pandas import DataFrame, NamedAgg

from glob import glob
from itertools import product
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import concat, read_csv
from tqdm.auto import tqdm

BAR_WIDTH = 1.0
BAR_PAD = 1.1
GROUP_PAD = 2.5

# Dictionary to hold data
data = {"surrogate_gradient": [], "quant_level": [], "quant_algo": [],
        "seed": [], "num_error_transitions": []}

# Get list of test files (signifies complete experiments) and obtain latest modification time
# **TODO** proper path
test_files = glob(os.path.join("..", "classifier", "test_output_*.csv"))
latest_modification_time = max(os.path.getmtime(f) for f in test_files)

# If pre-processed error rate data exists and is newer than latest test data file
if os.path.exists("mnist_error_rates.csv") and os.path.getmtime("mnist_error_rates.csv") > latest_modification_time:
    df = read_csv("mnist_error_rates.csv", dtype={"surrogate_gradient": str, "quant_level": float, "quant_algo": str,
                                                  "seed": str, "num_error_transitions": float})
else:
    # Loop through test files (signifies complete experiments)
    for name in tqdm(test_files):
        # Split name of test filename into components seperated by _
        name_components = os.path.splitext(os.path.basename(name))[0].split("_")

        # **YUCK** dvs-gesture should probably not be _ delimited - stops this generalising
        num_components = len(name_components)
        assert num_components == 18

        # Start config string with surrogate gradient function in use
        surrogate_gradient = name_components[12]

        # If quantisation is enabled, add type of encoding
        quant_level = None if name_components[9] == "None" else int(name_components[9])
        quant_algo = "Log" if name_components[10] == "True" else "Linear"

        # Load train error file and calculate average number of transitions per output neuron per trial
        train_error = np.load(os.path.join("..", "classifier", f"train_e_{'_'.join(name_components[2:])}.npy"))
        num_error_transitions = np.sum(train_error[:,1:,:] != train_error[:,:-1,:]) / (train_error.shape[0] * train_error.shape[2])

        # Add data to intermediate dictionary
        if name_components[7] == "mnist":
            data["surrogate_gradient"].append(surrogate_gradient)
            data["quant_level"].append(quant_level)
            data["quant_algo"].append(quant_algo)
            data["seed"].append(int(name_components[8]))
            data["num_error_transitions"].append(num_error_transitions)

    # Build dataframe from dictionary and save
    df = DataFrame(data=data)
    df.to_csv("mnist_error_rates.csv")

# Group data by config and number of layers (later just to ensure column is retained)
df = df.groupby(["surrogate_gradient", "quant_level", "quant_algo"], as_index=False, dropna=False)
df = df.agg(mean_num_error_transitions=NamedAgg(column="num_error_transitions", aggfunc=np.mean))

print(f"{len(df)} configurations")

# Get unique quantisation levels
quant_levels = df["quant_level"].unique()

# Build map of surrogate gradient + quantization algorithm permutations
pal = sns.color_palette()
colours = {q: pal[i] for i, q in enumerate((df["surrogate_gradient"] + "-" + df["quant_algo"]).unique())}

# Loop through quantization levels and plot group of bars
quant_fig, quant_axis = plt.subplots(figsize=(plot_settings.double_column_width, 2.0))
x = 0.0
tick_x = []
tick_label = []
for q in quant_levels:
    if np.isnan(q):
        label="None"
        q_df = df[df["quant_level"].isna()]
    else:
        label = f"{q}\nlevels"
        q_df = df[df["quant_level"] == q]

    # Calculate bar padding
    bar_x = x + (np.arange(len(q_df)) * BAR_PAD)
    x = bar_x[-1] + GROUP_PAD

    # Add label and
    tick_label.append(label)
    tick_x.append(np.average(bar_x))

    # Plot bars
    bar_colours = [colours[q] for q in (q_df["surrogate_gradient"] + "-" + q_df["quant_algo"]).to_list()]
    quant_axis.bar(bar_x, q_df["mean_num_error_transitions"],# yerr=df["sd_train_accuracy"],
                   color=bar_colours, width=BAR_WIDTH)
quant_axis.set_xticks(tick_x, tick_label)
quant_axis.set_xlabel("Error quantisation")
quant_axis.set_ylabel("Number of error events\nper output per trial")
quant_axis.set_yscale("log")
quant_fig.legend([patches.Patch(color=c) for c in colours.values()], colours.keys(),
                 loc="lower center", ncol=len(colours), frameon=False)
quant_fig.tight_layout(pad=0, rect=[0.0, 0.15, 1.0, 1.0])

# Save figures
if not plot_settings.presentation and not plot_settings.poster:
    quant_fig.savefig("../figures/mnist_error_rate.pdf")
plt.show()
