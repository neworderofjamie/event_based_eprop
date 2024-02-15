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
data = {"surrogate_gradient": [], "quant_scale": [], "quant_algo": [],
        "seed": [], "num_error_transitions": []}

# Get list of test files (signifies complete experiments) and obtain latest modification time
# **TODO** proper path
test_files = glob(os.path.join("..", "classifier", "test_output_*.csv"))
latest_modification_time = max(os.path.getmtime(f) for f in test_files)

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
    
    # Load record of number of error transitions and scale by batch size and num neurons
    batch_size = int(name_components[6])
    num_error_transitions = np.load(os.path.join("..", "classifier", f"train_error_transitions_{'_'.join(name_components[2:])}.npy"))
    num_error_transitions = np.divide(num_error_transitions, batch_size * 20, dtype=float)

    # Add data to intermediate dictionary
    data["surrogate_gradient"].append(surrogate_gradient)
    data["quant_scale"].append(quant_scale)
    data["quant_algo"].append(quant_algo)
    data["seed"].append(int(name_components[10]))
    data["num_error_transitions"].append(np.average(num_error_transitions))

# Build dataframe from dictionary and save
df = DataFrame(data=data)

# Group data by config and number of layers (later just to ensure column is retained)
df = df.groupby(["surrogate_gradient", "quant_scale", "quant_algo"], as_index=False, dropna=False)
df = df.agg(mean_num_error_transitions=NamedAgg(column="num_error_transitions", aggfunc=np.mean))

print(f"{len(df)} configurations")

# Get unique quantisation levels
quant_scales = df["quant_scale"].unique()

# Build map of surrogate gradient + quantization algorithm permutations
pal = sns.color_palette()
colours = {q: pal[i] for i, q in enumerate((df["surrogate_gradient"] + "-" + df["quant_algo"]).unique())}

# Loop through quantization levels and plot group of bars
quant_fig, quant_axis = plt.subplots(figsize=(plot_settings.double_column_width, 2.0))
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
    quant_axis.bar(bar_x, q_df["mean_num_error_transitions"],# yerr=df["sd_train_accuracy"],
                   color=bar_colours, width=BAR_WIDTH)
quant_axis.set_xticks(tick_x, tick_label)
quant_axis.set_xlabel("Error quantisation")
quant_axis.set_ylabel("Number of error events\nper output per trial")
quant_axis.set_yscale("log")
quant_axis.set_ylim((1.0, np.amax(df["mean_num_error_transitions"])))
quant_fig.legend([patches.Patch(color=c) for c in colours.values()], colours.keys(),
                 loc="lower center", ncol=len(colours), frameon=False)
quant_fig.tight_layout(pad=0, rect=[0.0, 0.15, 1.0, 1.0])

# Save figures
if not plot_settings.presentation and not plot_settings.poster:
    quant_fig.savefig("../figures/shd_error_rate.pdf")
plt.show()
