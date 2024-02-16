import os
import numpy as np

from pandas import DataFrame

from glob import glob
from json import load
from pandas import read_csv

def load_data_frame(params_wildcard, keys, num_outputs,
                    load_train=False, load_error_transitions=False):
    # Build dictionary to hold data
    data = {k: [] for k in keys}
    
    # If we should load training data, add list
    if load_train:
        data["train_accuracy"] = []
    
    # If we should load error transitions, add list
    if load_error_transitions:
        data["num_error_transitions"] = []
    
    # Loop through parameter files
    for name in glob(os.path.join("..", "classifier", params_wildcard)):
        # Get title from file
        title = os.path.splitext(os.path.basename(name))[0]

        # Load parameters
        with open(name) as fp:
            params = load(fp)

        if load_error_transitions:
            assert "batch_size" in params
            error_transition_filename = os.path.join("..", "classifier", f"train_error_transitions_{title[7:]}.npy")
            if not os.path.exists(error_transition_filename):
                print(f"ERROR: missing '{error_transition_filename}'")
                continue
            else:
                num_error_transitions = np.load(error_transition_filename)
                num_error_transitions = np.divide(num_error_transitions, params["batch_size"] * num_outputs, dtype=float)
            
        if load_train:
            assert "num_epochs" in params
            train_filename = os.path.join("..", "classifier", f"train_output_{title[7:]}.csv")
            if not os.path.exists(train_filename):
                print(f"ERROR: missing '{train_filename}'")
                continue
            else:
                train_data = read_csv(train_filename, delimiter=",")

                last_epoch_train_data = train_data[train_data["Epoch"] == (params["num_epochs"] - 1)]
                assert last_epoch_train_data.shape[0] == 1
        
        if load_error_transitions:
            data["num_error_transitions"].append(np.average(num_error_transitions))
        if load_train:
            data["train_accuracy"].append((100.0 * (last_epoch_train_data["Number correct"] / last_epoch_train_data["Num trials"])).iloc[0])

        # Add parameters to dictionary
        for k in keys:
            data[k].append(params[k])

    # Build dataframe from dictionary and save
    return DataFrame(data=data)
