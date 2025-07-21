#!/usr/bin/env python3

import argparse
import os
import itertools
import yaml
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib as mpl

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run grid search over spatial hyperparameters (crop size, pixel res, min pixels, cam size, etc)."
    )
    parser.add_argument('--input_train', type=str, required=True)
    parser.add_argument('--input_val', type=str, required=True)
    parser.add_argument('--input_mask', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--backbone', type=str, default="unet")
    parser.add_argument('--hyperparam_yaml', type=str, help="YAML file with parameter grid")
    parser.add_argument('--hyperparam_csv', type=str, help="CSV file with explicit parameter sets (advanced)")
    return parser.parse_args()

def load_param_grid(args):
    """Return a list of dicts, each a single hyperparam config"""
    if args.hyperparam_csv:
        df = pd.read_csv(args.hyperparam_csv)
        configs = df.to_dict(orient="records")
        return configs

    # Expanded grid if YAML not provided
    grid_dict = {
        'crop_size': [32, 40, 48, 56, 64, 72, 80, 96],
        'pixel_res': [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90],
        'min_pixels': [3, 5, 8, 10, 12, 15, 20],
        'cam_size': [256, 480, 720, 960, 1280],
    }
    if args.hyperparam_yaml:
        with open(args.hyperparam_yaml, "r") as f:
            grid_dict = yaml.safe_load(f)
    keys, values = zip(*grid_dict.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return configs

def run_pipeline(config, args, config_idx, run_idx, outdir):
    config_name = (
        f"crop{config['crop_size']}_pix{config['pixel_res']}_minpix{config['min_pixels']}_cam{config['cam_size']}_run{run_idx+1}"
    )
    curr_outdir = os.path.join(outdir, config_name)
    os.makedirs(curr_outdir, exist_ok=True)
    config_file = os.path.join(curr_outdir, "run_config.yaml")

    config_yaml = {
        'name': config_name,
        'epochs': args.epochs,
        'augment': 5,
        'batchsize': 108,
        'size': config['crop_size'],
        'cam_size': config['cam_size'],
        'backbone': args.backbone,
        'features': 32,
        'train_samples_per_epoch': 50000,
        'num_workers': 4,
        'projhead': 'minimal_batchnorm',
        'classhead': 'minimal',
        'input_train': [args.input_train],
        'input_val': [args.input_val],
        'input_mask': [args.input_mask],
        'split_train': [[0.0, 1.0]],
        'split_val': [[0.0, 1.0]],
        'outdir': curr_outdir,
        'gpu': "0",
        'seed': args.seed + run_idx,
        'pixel_resolution': config['pixel_res'],
        'tensorboard': False,
        'write_final_cams': False,
        'binarize': False,
        'min_pixels': config['min_pixels'],
        'config_yaml': config_file,
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_yaml, f)

    pipeline_cmd = [
        "python3", "Workflow/01_fine-tune.py",
        "--config", config_file
    ]
    print(f"Running: {' '.join(pipeline_cmd)}")
    subprocess.run(pipeline_cmd, check=True)
    metrics_csv = os.path.join(curr_outdir, "pipeline_metrics.csv")
    return metrics_csv, curr_outdir

def multipanel_plot(df, param, outdir):
    import matplotlib as mpl

    # Publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'axes.edgecolor': '#222222',
        'axes.linewidth': 1.1,
        'axes.titlesize': 18,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'font.family': 'DejaVu Sans',
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'axes.grid': True,
        'grid.alpha': 0.32,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # Parameters for lines (number of unique non-varied configs)
    param_values = np.sort(df[param].unique())
    other_params = [p for p in ['crop_size', 'pixel_res', 'min_pixels', 'cam_size'] if p != param]
    combo_cols = other_params

    num_lines = df.groupby(combo_cols).ngroups
    cmap = mpl.cm.get_cmap('Blues', num_lines+2)
    line_colors = [cmap(i) for i in range(2, num_lines+2)]

    # -- HORIZONTAL PAIRED PANELS --
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, gridspec_kw={'wspace': 0.18})

    # Panel 1: Test Accuracy (LEFT)
    ax1 = axs[0]
    for idx, (key, grp) in enumerate(df.groupby(combo_cols)):
        x = grp[param].values
        y = grp['val_acc'].values
        if len(x) < 4:  # spline requires at least 4 points
            ax1.plot(x, y, color=line_colors[idx], linewidth=2.2, alpha=0.93, zorder=2)
            ax1.scatter(x, y, color=line_colors[idx], s=45, edgecolor='white', zorder=3)
        else:
            xnew = np.linspace(x.min(), x.max(), 100)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(xnew)
            ax1.plot(xnew, y_smooth, color=line_colors[idx], linewidth=2.2, alpha=0.93, zorder=2)
            ax1.scatter(x, y, color=line_colors[idx], s=45, edgecolor='white', zorder=3)
    # Find and mark the best (max) accuracy point
    max_idx = df['val_acc'].idxmax()
    best_x = df.loc[max_idx, param]
    best_y = df.loc[max_idx, 'val_acc']
    ax1.scatter([best_x], [best_y], color='red', s=95, edgecolor='black', zorder=10)
    ax1.axvline(best_x, color='red', linestyle=':', linewidth=1.7, alpha=0.55, zorder=0)
    ax1.set_title(f"Test Set Accuracy\nvs. {param.replace('_', ' ').title()}", weight='bold', pad=8)
    ax1.set_ylabel("Test Accuracy", labelpad=8)
    ax1.set_xticks(param_values)
    ax1.set_ylim(0.60, 1.02)
    ax1.tick_params(axis='both', which='both', length=5)
    ax1.set_axisbelow(True)

    # Panel 2: Test Loss (RIGHT)
    ax2 = axs[1]
    for idx, (key, grp) in enumerate(df.groupby(combo_cols)):
        x = grp[param].values
        y = grp['val_loss'].values
        if len(x) < 4:
            ax2.plot(x, y, color=line_colors[idx], linewidth=2.2, alpha=0.93, zorder=2)
            ax2.scatter(x, y, color=line_colors[idx], s=45, edgecolor='white', zorder=3)
        else:
            xnew = np.linspace(x.min(), x.max(), 100)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(xnew)
            ax2.plot(xnew, y_smooth, color=line_colors[idx], linewidth=2.2, alpha=0.93, zorder=2)
            ax2.scatter(x, y, color=line_colors[idx], s=45, edgecolor='white', zorder=3)
    # Find and mark the best (min) loss point
    min_idx = df['val_loss'].idxmin()
    best_x2 = df.loc[min_idx, param]
    best_y2 = df.loc[min_idx, 'val_loss']
    ax2.scatter([best_x2], [best_y2], color='red', s=95, edgecolor='black', zorder=10)
    ax2.axvline(best_x2, color='red', linestyle=':', linewidth=1.7, alpha=0.55, zorder=0)
    ax2.set_title(f"Test Set Loss\nvs. {param.replace('_', ' ').title()}", weight='bold', pad=8)
    ax2.set_xlabel(param.replace('_', ' ').title(), labelpad=8)
    ax2.set_ylabel("Test Loss", labelpad=8)
    ax2.set_xticks(param_values)
    ax2.set_xlim(param_values.min()-2, param_values.max()+2)
    ax2.tick_params(axis='both', which='both', length=5)
    ax2.set_axisbelow(True)

    plt.tight_layout(pad=1.2)
    save_path = os.path.join(outdir, f"multipanel_{param}.png")
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"Saved multipanel plot: {save_path}")

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    configs = load_param_grid(args)
    print(f"Total grid points: {len(configs)}")
    all_results = []
    for config_idx, config in enumerate(configs):
        for run_idx in range(args.num_runs):
            try:
                metrics_csv, outdir = run_pipeline(config, args, config_idx, run_idx, args.outdir)
                result = config.copy()
                result['run'] = run_idx+1
                if os.path.exists(metrics_csv):
                    metrics = pd.read_csv(metrics_csv)
                    last_row = metrics.iloc[-1]
                    for col in metrics.columns:
                        result[col] = last_row[col]
                all_results.append(result)
            except Exception as e:
                print(f"[!] Failed run: config {config}, run {run_idx+1} | Error: {e}")

    # Save summary CSV
    results_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(args.outdir, "grid_search_summary.csv")
    results_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    # Multi-panel plots for each hyperparameter
    for param in ['crop_size', 'pixel_res', 'min_pixels', 'cam_size']:
        if param not in results_df.columns:
            continue
        multipanel_plot(results_df, param, args.outdir)

if __name__ == '__main__':
    main()
