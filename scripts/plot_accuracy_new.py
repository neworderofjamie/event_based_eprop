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

def plot_accuracy(fig, axis, df, accuracy_column):
    # Get unique quantisation levels
    quant_scales = df["quant_scale"].unique()

    # Build map of surrogate gradient + quantization algorithm permutations
    pal = sns.color_palette()
    colours = {q: pal[i] for i, q in enumerate((df["surrogate_gradient"] + "-" + df["quant_algo"]).unique())}

    x = 0.0
    tick_x = []
    tick_label = []
    for q in quant_scales:
        if np.isnan(q):
            label="None"
            q_df = df[df["quant_scale"].isna()]
        else:
            label = f"{q}\nlevels"
            q_df = df[df["quant_scale"] == q]

        # Calculate bar padding
        bar_x = x + (np.arange(len(q_df)) * BAR_PAD)
        x = bar_x[-1] + GROUP_PAD

        # Add label and
        tick_label.append(label)
        tick_x.append(np.average(bar_x))

        # Plot bars
        bar_colours = [colours[q] for q in (q_df["surrogate_gradient"] + "-" + q_df["quant_algo"]).to_list()]
        axis.bar(bar_x, q_df[accuracy_column],# yerr=df["sd_train_accuracy"],
                 color=bar_colours, width=BAR_WIDTH)
    axis.set_xticks(tick_x, tick_label)
    axis.set_xlabel("Error quantisation")
    axis.set_ylim((0, 100))
    fig.legend([patches.Patch(color=c) for c in colours.values()], colours.keys(),
               loc="lower center", ncol=len(colours), frameon=False)

BAR_WIDTH = 1.0
BAR_PAD = 1.1
GROUP_PAD = 2.5

# Dictionary to hold data
data = {"surrogate_gradient": [], "quant_scale": [], "quant_algo": [],
        "seed": [], "train_accuracy": []}

# Loop through train files (signifies complete experiments)
# **TODO** proper path
for name in glob(os.path.join("..", "classifier", "train_output_?_False_None_False_512_50_shd_1*.csv")):
    # Split name of test filename into components seperated by _
    name_components = os.path.splitext(os.path.basename(name))[0].split("_")

    # **YUCK** dvs-gesture should probably not be _ delimited - stops this generalising
    num_components = len(name_components)
    assert num_components == 20

    # Start config string with surrogate gradient function in use
    surrogate_gradient = name_components[14]

    # If quantisation is enabled, add type of encoding
    quant_scale = None if name_components[11] == "None" else float(name_components[11])
    quant_algo = "Log" if name_components[12] == "True" else "Linear"

    # Read train output CSV
    train_data = read_csv(name, delimiter=",")

    last_epoch = int(name_components[7]) - 1
    last_epoch_train_data = train_data[train_data["Epoch"] == last_epoch]
    assert last_epoch_train_data.shape[0] == 1

    # Add data to intermediate dictionary
    data["surrogate_gradient"].append(surrogate_gradient)
    data["quant_scale"].append(quant_scale)
    data["quant_algo"].append(quant_algo)
    data["seed"].append(int(name_components[10]))
    data["train_accuracy"].append((100.0 * (last_epoch_train_data["Number correct"] / last_epoch_train_data["Num trials"])).iloc[0])

# Build dataframe from dictionary and sort by config
df = DataFrame(data=data)

# Group data by config and number of layers (later just to ensure column is retained)
df = df.groupby(["surrogate_gradient", "quant_scale", "quant_algo"], as_index=False, dropna=False)
df = df.agg(mean_train_accuracy=NamedAgg(column="train_accuracy", aggfunc=np.mean),
            sd_train_accuracy=NamedAgg(column="train_accuracy", aggfunc=np.std))

print(f"{len(df)} configurations")


# Plot training accuracy
train_quant_fig, train_quant_axis = plt.subplots(figsize=(plot_settings.double_column_width, 2.0))
plot_accuracy(train_quant_fig, train_quant_axis, df, "mean_train_accuracy")
train_quant_axis.set_ylabel("Training accuracy [%]")
train_quant_fig.tight_layout(pad=0, rect=[0.0, 0.15, 1.0, 1.0])

# Save figures
if not plot_settings.presentation and not plot_settings.poster:
    train_quant_fig.savefig("../figures/shd_train_accuracy.pdf")
plt.show()
