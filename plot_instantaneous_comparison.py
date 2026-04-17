import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from helper_functions import animate_comparison, calculate_metrics, find_fire_scar

import matplotlib.pyplot as plt

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,            # Default font size for everything
    'axes.titlesize': 23,       # Title size
    'axes.labelsize': 23,       # X and Y labels
    'xtick.labelsize': 23,      # X tick labels
    'ytick.labelsize': 23,      # Y tick labels
    'legend.fontsize': 23,      # Legend
    'figure.titlesize': 18,     # Figure title
})

# -----------------------------------------------------------------------------
# 1) Prepare data
# -----------------------------------------------------------------------------
data_root_folder = './model_outputs/'

folders = ['Baseline','Baseline_with_Otsu',
           'Baseline_4fold_uniformweight_ensemblerollout','Baseline_4fold_uniformweight_ensemblerollout_otsu',
           'Baseline_4fold_4xsmaller_uniformweight_ensemblerollout_otsu',
           'Baseline_4fold_25epochs_uniformweight_ensemblerollout_otsu']
model_labels = ['Baseline','Otsu','4-fold','4-fold Otsu','4-fold Otsu 4x smaller','4fold Otsu 25 epochs']

xif_filename = 'xif_after_comp.npy'
pred_filename = 'pred_after_comp.npy'
last_snapshot_filename = 'last_test_snapshot.npy'
df_test = pd.read_csv(data_root_folder + 'test_original.csv')
n_predict = 60

# Load ground truth
xif_after_comp = np.load(data_root_folder + xif_filename)   # shape: (num_test_cases, n_predict, H, W)

# For each folder, load predictions into a list
all_predictions = []
for folder in folders:
    pred_after_comp = np.load((data_root_folder + folder) + ('/' + pred_filename))
    # Some models might have shape issues
    if folder == 'Thomas Dubail':
        pred_after_comp = pred_after_comp[:, :, 0, :, :].squeeze()
    all_predictions.append(pred_after_comp)

# User‐selectable parameters
desired_u     = 10     # e.g. 10 m/s
desired_alpha = 30   # e.g. 30 degrees
selected_time = 60    # whatever frame you want 1 ≤ t ≤ n_predict
selected_time_idx = selected_time - 1 # change to an index

# find the matching row in df_test
mask = (df_test['u'] == desired_u) & (df_test['alpha'] == desired_alpha)
if not mask.any():
    raise ValueError(f"No case found with u={desired_u}, alpha={desired_alpha}")
# if there are duplicates, just take the first
selected_case_idx = mask.idxmax()

# -----------------------------------------------------------------------------
# 2) Define "fire scar" vs "line"
# -----------------------------------------------------------------------------
def compute_scar_or_line_data(xif_data, pred_data, line_or_scar, last_snapshot_fname):
    """
    If line_or_scar == 'scar', run find_fire_scar() on both xif_data and pred_data.
    Otherwise, return xif_data and pred_data as-is.
    """
    if line_or_scar == 'scar':
        # shape: (num_test_cases, n_predict, H, W)
        num_cases = xif_data.shape[0]
        xif_scarred = np.zeros_like(xif_data)
        pred_scarred = np.zeros_like(pred_data)
        
        last_test_snapshots = np.load(last_snapshot_fname)  # shape: (num_test_cases, H, W)
        
        for j in range(num_cases):
            initial_fire_line = last_test_snapshots[j, :, :].squeeze()
            xif_scarred[j, ...] = find_fire_scar(
                xif_data[j, ...].squeeze(), 
                initial_fire_line
            )
            pred_scarred[j, ...] = find_fire_scar(
                pred_data[j, ...].squeeze(), 
                initial_fire_line
            )
        return xif_scarred, pred_scarred
    else:
        # 'line'
        return xif_data, pred_data

line_or_scar = 'line'  # 'line' or 'scar'

# -----------------------------------------------------------------------------
# 3) Compute final ground-truth data and predictions
# -----------------------------------------------------------------------------
all_scar_or_line_preds = []
for i, folder in enumerate(folders):
    xif_processed, pred_processed = compute_scar_or_line_data(
        xif_after_comp, all_predictions[i],
        line_or_scar, last_snapshot_filename
    )
    all_scar_or_line_preds.append(pred_processed)

# We'll define once the final ground truth for all test cases
xif_final, _ = compute_scar_or_line_data(
    xif_after_comp, all_predictions[0],  # dummy pred
    line_or_scar, last_snapshot_filename
)

def plot_models_comparison_static(xif_data, list_of_preds,
                                  case_idx, time_idx, model_labels,
                                  outfile=None):
    """
    Stack of images (ground truth + N models) for one (case, time).

    * Each subplot gets y‑axis label “y [m]”.
    * Bottom subplot also shows x‑axis with label “x [m]”.
    * Pixel size is 8 m ⇒ axes scaled accordingly.
    """
    num_models = len(list_of_preds)
    nrows      = num_models + 1

    H, W = xif_data.shape[2:]          # original array shape (H rows, W cols)
    dx   = 8.0                         # metres per pixel
    extent_rot = [0, H*dx, 0, W*dx]    # because we’ll plot arr.T  (see below)

    fig, axes = plt.subplots(nrows, 1,
                             figsize=(5, 1.5*nrows), dpi=300)
    plt.tight_layout()

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, axes = plt.subplots(nrows, 1, figsize=(5, 1.5*nrows), dpi=300, constrained_layout=True)
    
    for row, ax in enumerate(axes):
        if row == 0:
            arr = xif_data[case_idx, time_idx].T
            title = "Ground Truth"
        else:
            arr = list_of_preds[row-1][case_idx, time_idx].T
            title = model_labels[row-1]
    
        im = ax.imshow(arr, cmap='hot', vmin=0, vmax=1,
                       origin='lower', extent=extent_rot, aspect='auto')
    
        ax.set_ylabel('y [m]')
    
        if row == nrows - 1:
            ax.set_xlabel('x [m]')
            ax.tick_params(axis='x', which='both', labelbottom=True)
    
            cax = inset_axes(ax,
                             width="95%",    # bar length relative to the axes width
                             height="10%",   # bar thickness relative to the axes height
                             loc='lower center',
                             bbox_to_anchor=(0, -1.2, 1, 1),  # push below ticks; increase -0.55 if needed
                             bbox_transform=ax.transAxes,
                             borderpad=0)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_label(r'$\xi_f$')
        else:
            ax.tick_params(axis='x', which='both', labelbottom=False)
    
        ax.set_xlim((100, 800))
        ax.set_xticks([250, 500, 750])
    
    if outfile:
        fig.savefig(outfile, dpi=600)
    plt.show()

    
output_dir = "figures/instantaneous_comparisons"
os.makedirs(output_dir, exist_ok=True)
outfile = os.path.join(
    output_dir,
    f"u{desired_u}_alpha{desired_alpha}_t{selected_time_idx}.png"
)

plot_models_comparison_static(
    xif_data   = xif_final,
    list_of_preds = all_scar_or_line_preds,
    case_idx   = selected_case_idx,
    time_idx   = selected_time_idx,
    model_labels  = model_labels,
    outfile    = outfile
)

print(f"Done! Figure saved to {outfile}")
