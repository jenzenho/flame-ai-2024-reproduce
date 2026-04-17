import numpy as np
import os
import pandas as pd
from helper_functions import animate_comparison, calculate_metrics, find_fire_scar

from pathlib import Path
import matplotlib.pyplot as plt

# This script compares model predictions against ground truth wildfire spread maps,
# computes time-resolved error metrics, and exports summary figures for either:
# (1) the original competition models, or
# (2) baseline modification models.
#
# Expected folder structure:
#   model_outputs/
#       test_original.csv
#       xif_after_comp.npy
#       last_test_snapshot.npy
#       <model_folder>/pred_after_comp.npy
#
# Optional outputs:
#   - per-case animations saved under each model folder
#   - vector figures saved under ./figures/error_metrics/

## These are the testing velocities, slopes, and starting times
# U_TEST_ = [10, 14, 18, 22]
# RAMP_TEST_ = [2.5, 7.5, 12.5, 30]
# SPECIFIC_TIMES = [0,30,60]

ROOT_FOLDER = Path("./model_outputs")

# Choose which family of models to compare:
# - 'competition': original competition entries
# - 'baseline_modifications': variants of the baseline model
COMPETITION_OR_BASELINEMODS = 'competition' 

if COMPETITION_OR_BASELINEMODS == 'competition':
    folders = ['Line','Simulation ROS','Baseline','Ajay Asaithambi','Jobayer Hossain','Rafal Pawlowski','Zhuoqun Li','Thomas Dubail']
    model_label = ['Linear Regression','Simulation ROS','Baseline ML Model','Mixed','Latent Loop','SwinUNet','MultiResUNet','Conv-TT-LSTM']
elif COMPETITION_OR_BASELINEMODS == 'baseline_modifications':
    folders = ['Baseline','Baseline_with_Otsu',
               'Baseline_4fold_uniformweight_ensemblerollout','Baseline_4fold_uniformweight_ensemblerollout_otsu',
               'Baseline_4fold_4xsmaller_uniformweight_ensemblerollout_otsu',
               'Baseline_4fold_25epochs_uniformweight_ensemblerollout_otsu']
    model_label = ['Baseline','Otsu','4-fold','4-fold Otsu',
                   '4-fold Otsu 4x smaller',
                   '4-fold Otsu 25 epochs']

assert len(folders) == len(model_label)

# Input files:
# - xif_after_comp.npy: ground-truth time series for all test cases
# - pred_after_comp.npy: model predictions for a single model folder
# - last_test_snapshot.npy: initial fire-line state used for fire-scar metrics
XIF_FILENAME = 'xif_after_comp.npy'
PRED_FILENAME = 'pred_after_comp.npy'
LAST_SNAPSHOT_FILENAME = 'last_test_snapshot.npy'

# Metadata for each test case. This is mainly used to generate readable filenames
# for optional animation outputs.
df_test = pd.read_csv(ROOT_FOLDER / "test_original.csv")
xif_after_comp = np.load(ROOT_FOLDER / XIF_FILENAME)
last_test_snapshots = np.load(ROOT_FOLDER / LAST_SNAPSHOT_FILENAME)

# Runtime options:
# - generate_animations: if True, save side-by-side truth/prediction animations for each case
# - line_or_scar:
#       'line' -> compare instantaneous fire-line fields
#       'scar' -> compare cumulative burned-area / fire-scar fields
generate_animations = False
line_or_scar = 'line'

#%%
# Accumulate model-averaged metrics in two forms:
# - *_all: time-resolved averages over all test cases
# - *_one_val: single scalar averages over both cases and time
mse_all = []
ssim_all = []
jaccard_all = []
fire_loc_all = []
perc_fire_loc_all = []
mean_loc_all = []
perc_mean_loc_all = []
spread_loc_all = []
perc_spread_loc_all = []
spread_mean_all = []
perc_spread_mean_all = []

mse_one_val = []
ssim_one_val = []
jaccard_one_val = []
fire_loc_one_val = []
mean_loc_one_val = []
perc_fire_loc_one_val = []
perc_mean_loc_one_val = []

# Process one model at a time: load its predictions, optionally convert to fire-scar
# representation, optionally generate animations, then compute aggregate error metrics.
for folder in folders:
    pred_after_comp = np.load(ROOT_FOLDER / folder / PRED_FILENAME)
    
    # Thomas Dubail's predictions are stored with an extra singleton dimension,
    # so they need to be reshaped to match the standard (n_cases, n_t, n_x, n_y) format.
    if folder=='Thomas Dubail':
        pred_after_comp = pred_after_comp[:,:,0,:,:].squeeze()
    
    xif_analysis = np.zeros(np.shape(xif_after_comp))
    pred_analysis = np.zeros(np.shape(pred_after_comp))
    
    # Convert instantaneous fire-line fields into cumulative fire-scar fields if requested.
    # In 'line' mode, use the raw arrays directly.
    if line_or_scar == 'scar':
        for j in range(np.shape(xif_after_comp)[0]):
            initial_fire_line = last_test_snapshots[j,:,:].squeeze()
            xif_analysis[j,:,:,:] = find_fire_scar(xif_after_comp[j,:,:,:].squeeze(),initial_fire_line)
            pred_analysis[j,:,:,:] = find_fire_scar(pred_after_comp[j,:,:,:].squeeze(),initial_fire_line)
    else:
        xif_analysis = xif_after_comp
        pred_analysis = pred_after_comp
                
    # Optionally save one animation per case inside each model folder.
    # Filenames encode the wind speed, slope, and initial time for traceability.
    if generate_animations:
        for j in range(np.shape(pred_analysis)[0]):
            if line_or_scar == 'line':
                if not os.path.exists(os.path.join(ROOT_FOLDER,folder,'animations_line')):
                    os.mkdir(os.path.join(ROOT_FOLDER,folder,'animations_line'))
                outfile = os.path.join(ROOT_FOLDER,folder,'animations_line','u'+str(df_test['u'][j])+'_slope'+str(df_test['alpha'][j])+'_tinit'+str(df_test['t_initial'][j])+'.mp4')
            elif line_or_scar == 'scar':
                if not os.path.exists(os.path.join(ROOT_FOLDER,folder,'animations_scar')):
                    os.mkdir(os.path.join(ROOT_FOLDER,folder,'animations_scar'))
                outfile = os.path.join(ROOT_FOLDER,folder,'animations_scar','u'+str(df_test['u'][j])+'_slope'+str(df_test['alpha'][j])+'_tinit'+str(df_test['t_initial'][j])+'.mp4')
            animate_comparison(xif_analysis,pred_analysis,case_index=j,output_file=outfile)
            
    # Calculate error metrics
    mse_case = []
    ssim_case = []
    jaccard_case = []
    fire_loc_case = []
    perc_fire_loc_case = []
    mean_loc_case = []
    perc_mean_loc_case = []
    spread_loc_case = []
    perc_spread_loc_case = []
    spread_mean_case = []
    perc_spread_mean_case = []
    for j in range(xif_analysis.shape[0]):
        initial_fire_line = last_test_snapshots[j,:,:].squeeze()
        
        mse_values, ssim_values, jaccard_values, fire_loc_error, perc_fire_loc_error, mean_loc_error, perc_mean_loc_error, spread_loc_error, perc_spread_loc_error, spread_mean_error, perc_spread_mean_error = calculate_metrics(xif_analysis[j],pred_analysis[j],initial_fire_line,initial_fire_line)
        
        mse_case.append(mse_values)
        ssim_case.append(ssim_values)
        jaccard_case.append(jaccard_values)
        fire_loc_case.append(fire_loc_error)
        perc_fire_loc_case.append(perc_fire_loc_error)
        mean_loc_case.append(mean_loc_error)
        perc_mean_loc_case.append(perc_mean_loc_error)
        spread_loc_case.append(spread_loc_error)
        perc_spread_loc_case.append(perc_spread_loc_error)
        spread_mean_case.append(spread_mean_error)
        perc_spread_mean_case.append(perc_spread_mean_error)
    
    mse_all.append(np.nanmean(np.array(mse_case),0))
    ssim_all.append(np.nanmean(np.array(ssim_case),0))
    jaccard_all.append(np.nanmean(np.array(jaccard_case),0))
    fire_loc_all.append(np.nanmean(np.array(fire_loc_case),0))
    perc_fire_loc_all.append(np.nanmean(np.array(perc_fire_loc_case),0))
    mean_loc_all.append(np.nanmean(np.array(mean_loc_case),0))
    perc_mean_loc_all.append(np.nanmean(np.array(perc_mean_loc_case),0))
    spread_loc_all.append(np.nanmean(np.array(spread_loc_case),0))
    perc_spread_loc_all.append(np.nanmean(np.array(perc_spread_loc_case),0))
    spread_mean_all.append(np.nanmean(np.array(spread_mean_case),0))
    perc_spread_mean_all.append(np.nanmean(np.array(perc_spread_mean_case),0))
    
    mse_one_val.append(np.nanmean(np.array(mse_case)))
    ssim_one_val.append(np.nanmean(np.array(ssim_case)))
    jaccard_one_val.append(np.nanmean(np.array(jaccard_case)))
    fire_loc_one_val.append(np.nanmean(np.array(fire_loc_case)))
    mean_loc_one_val.append(np.nanmean(np.array(mean_loc_case)))
    perc_fire_loc_one_val.append(np.nanmean(np.array(perc_fire_loc_case)))
    perc_mean_loc_one_val.append(np.nanmean(np.array(perc_mean_loc_case)))
    
mse_all = np.array(mse_all)
ssim_all = np.array(ssim_all)
jaccard_all = np.array(jaccard_all)
fire_loc_all = np.array(fire_loc_all)
perc_fire_loc_all = np.array(perc_fire_loc_all)
mean_loc_all = np.array(mean_loc_all)
perc_mean_loc_all = np.array(perc_mean_loc_all)
spread_loc_all = np.array(spread_loc_all)
perc_spread_loc_all = np.array(perc_spread_loc_all)
spread_mean_all = np.array(spread_mean_all)
perc_spread_mean_all = np.array(perc_spread_mean_all)

# Shared plotting style used for all exported metric figures.
plt.rcParams["font.family"] = "Times New Roman"

FONT_LABEL = 16
FONT_TICK = 14
FONT_LEGEND = 9

VEC_FMT = "pdf"   # change to "eps" if needed
OUTDIR = Path("./figures/error_metrics")
OUTDIR.mkdir(parents=True, exist_ok=True)

CUSTOM_COLORS = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Green
    "#8C564B",  # Brown
    "#D55E00",  # Red-orange
    "#CC79A7",  # Purple
    "#000000",  # Black (8th model)
]

def style_axis(ax, ylabel=None, show_legend=False):
    """Apply consistent formatting to a time-series metric axis."""
    ax.set_xlabel("t [s]", fontsize=FONT_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)

    # Shade the first 20 s and draw a marker at t = 20 s
    ax.axvspan(0, 20, color="0.5", alpha=0.15, zorder=0)
    ymin, ymax = ax.get_ylim()
    ax.axvline(20, color="black", lw=1)

    # Force lower bound to zero for these metrics
    ax.set_ylim(bottom=0)

    ax.minorticks_on()
    ax.tick_params(direction="in", which="both")
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK)

    if show_legend:
        ax.legend(fontsize=FONT_LEGEND, frameon=True, ncol=2)

def plot_metric_series(data, model_labels, ylabel, filename, show_legend=False):
    """
    Plot one metric versus time for all models and save as vector figure.

    Parameters
    ----------
    data : ndarray, shape (n_models, n_times)
        Metric values for each model across time.
    model_labels : list[str]
        Names of the models.
    ylabel : str
        Y-axis label.
    filename : str
        Output filename stem without extension.
    show_legend : bool
        Whether to draw the legend.
    """
    fig, ax = plt.subplots(dpi=300, figsize=(6, 4.5))

    for i in range(data.shape[0]):
        linestyle = "--" if i < 3 else "-"
        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        ax.plot(
            np.arange(data.shape[1]) + 1,
            data[i, :],
            linestyle=linestyle,
            color=color,
            label=model_labels[i],
        )

    style_axis(ax, ylabel=ylabel, show_legend=show_legend)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{filename}.{VEC_FMT}", bbox_inches="tight")
    plt.close(fig)

def make_all_plots(
    mse_all,
    ssim_all,
    jaccard_all,
    fire_loc_all,
    perc_fire_loc_all,
    mean_loc_all,
    perc_mean_loc_all,
    model_labels,
    line_or_scar="line",
):
    """Generate all metric figures used in the paper."""
    plot_metric_series(mse_all, model_labels, r"MSE $\downarrow$", "metric_mse", show_legend=True)
    plot_metric_series(ssim_all, model_labels, r"SSIM $\uparrow$", "metric_ssim")
    plot_metric_series(jaccard_all, model_labels, r"Jaccard $\uparrow$", "metric_jaccard")

    if line_or_scar == "line":
        plot_metric_series(
            fire_loc_all,
            model_labels,
            "Furthest fire location error [m]",
            "fire_furthest_loc_err",
            show_legend=True,
        )
        plot_metric_series(
            perc_fire_loc_all,
            model_labels,
            "Furthest fire location percentage error",
            "fire_furthest_loc_pct_err",
        )
        plot_metric_series(
            mean_loc_all,
            model_labels,
            "Mean fire location error [m]",
            "fire_mean_loc_err",
        )
        plot_metric_series(
            perc_mean_loc_all,
            model_labels,
            "Mean fire location percentage error",
            "fire_mean_loc_pct_err",
        )
        
make_all_plots(
    mse_all=mse_all,
    ssim_all=ssim_all,
    jaccard_all=jaccard_all,
    fire_loc_all=fire_loc_all,
    perc_fire_loc_all=perc_fire_loc_all,
    mean_loc_all=mean_loc_all,
    perc_mean_loc_all=perc_mean_loc_all,
    model_labels=model_label,
    line_or_scar=line_or_scar,
)