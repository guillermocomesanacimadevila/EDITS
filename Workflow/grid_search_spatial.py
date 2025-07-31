# grid_search_spatial.py
import os
import subprocess
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

# ========== USER-DEFINED GRID ==========
sizes = [32, 48, 64]
min_pixels_list = [5, 10]
pixel_resolutions = [0.5, 0.65]
crops_per_image_list = [500, 1000]
cam_size_list = [64, 96, 128]
n_folds = 5

# ========== PATHS ==========
DATA_ROOT = "../Data"
INPUT_TRAIN = os.path.join(DATA_ROOT, "021221_C16-1_8bit_PFFC-BrightAdj.tif")
INPUT_VAL = os.path.join(DATA_ROOT, "021221_C16-1_8bit_PFFC-BrightAdj.tif")
INPUT_MASK = os.path.join(DATA_ROOT, "021221_C16-1_8bit_PFFC-BrightAdj_binary_allevents.tif")

results = []

# Use a dummy split over 100 "samples" for reproducible folds
N_SAMPLES = 100
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
splits = list(kf.split(range(N_SAMPLES)))

# ========== SPATIAL GRID SEARCH WITH 5-FOLD CV ==========
for size, min_pixels, pixel_res, crops_per_image, cam_size in product(
    sizes, min_pixels_list, pixel_resolutions, crops_per_image_list, cam_size_list
):
    fold_losses = []
    fold_dirs = []
    print(f"\n=== Running config: size={size}, min_pixels={min_pixels}, pixel_res={pixel_res}, "
          f"crops_per_image={crops_per_image}, cam_size={cam_size} ===\n")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        split_train = (min(train_idx)/N_SAMPLES, (max(train_idx)+1)/N_SAMPLES)
        split_val = (min(val_idx)/N_SAMPLES, (max(val_idx)+1)/N_SAMPLES)
        outdir = f"runs/grid_size{size}_minpx{min_pixels}_pixres{pixel_res}_crops{crops_per_image}_camsz{cam_size}_fold{fold_idx}"
        cmd = [
            "python3", "01_fine-tune.py",
            "--input_train", INPUT_TRAIN,
            "--input_val", INPUT_VAL,
            "--input_mask", INPUT_MASK,
            "--split_train", f"{split_train[0]:.4f}", f"{split_train[1]:.4f}",
            "--split_val", f"{split_val[0]:.4f}", f"{split_val[1]:.4f}",
            "--size", str(size),
            "--epochs", "200",
            "--batchsize", "108",
            "--backbone", "unet",
            "--classhead", "linear",
            "--projhead", "minimal_batchnorm",
            "--min_pixels", str(min_pixels),
            "--outdir", outdir,
            "--seed", "42",
            "--pixel_resolution", str(pixel_res),
            "--crops_per_image", str(crops_per_image),
            "--cam_size", str(cam_size)
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        # Find model dir made by 01_fine-tune.py (timestamped inside outdir)
        model_dirs = [os.path.join(outdir, d) for d in os.listdir(outdir) if "run_backbone_unet" in d]
        model_dirs.sort()
        if not model_dirs:
            print(f"WARNING: No model directory found in {outdir}")
            fold_losses.append(float("inf"))
            fold_dirs.append(None)
            continue
        model_dir = model_dirs[-1]
        fold_dirs.append(model_dir)
        metrics_path = os.path.join(model_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            loss = df["val_loss"].iloc[-1]
            fold_losses.append(loss)
        else:
            print(f"WARNING: No metrics.csv found in {model_dir}")
            fold_losses.append(float("inf"))

    # Compute metrics
    mean_val_loss = np.mean(fold_losses)
    std_val_loss = np.std(fold_losses)
    best_fold = int(np.argmin(fold_losses))
    best_model_dir = fold_dirs[best_fold]

    print(f"--> Mean val_loss for config: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"    Best fold: {best_fold} (val_loss={fold_losses[best_fold]:.4f})")
    print(f"    Model directory: {best_model_dir}")

    results.append({
        "size": size,
        "min_pixels": min_pixels,
        "pixel_res": pixel_res,
        "crops_per_image": crops_per_image,
        "cam_size": cam_size,
        "mean_val_loss": mean_val_loss,
        "std_val_loss": std_val_loss,
        "best_fold": best_fold,
        "best_model_dir": best_model_dir
    })

    print(f"\n=== Finished config size={size}, min_pixels={min_pixels}, pixel_res={pixel_res}, "
          f"crops_per_image={crops_per_image}, cam_size={cam_size} ===\n")

print("==== ALL CONFIGURATIONS COMPLETE! ====")

# ========== Save results ==========
results_df = pd.DataFrame(results)
os.makedirs("spatial_search_results", exist_ok=True)
results_csv = "spatial_search_results/spatial_grid_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"Saved grid search summary CSV to: {results_csv}")

# ========== Retrain best config on full data 10x ==========
best_idx = results_df["mean_val_loss"].idxmin()
best_row = results_df.iloc[best_idx]
print("Best spatial hyperparam config:", best_row)

final_run_dirs = []
final_losses = []
for repeat_seed in range(10):
    outdir = f"runs/best_spatial_full_train_seed{repeat_seed}"
    cmd = [
        "python3", "01_fine-tune.py",
        "--input_train", INPUT_TRAIN,
        "--input_val", INPUT_VAL,
        "--input_mask", INPUT_MASK,
        "--split_train", "0.0", "1.0",
        "--split_val", "0.0", "1.0",
        "--size", str(best_row["size"]),
        "--epochs", "200",
        "--batchsize", "108",
        "--backbone", "unet",
        "--classhead", "linear",
        "--projhead", "minimal_batchnorm",
        "--min_pixels", str(best_row["min_pixels"]),
        "--outdir", outdir,
        "--seed", str(repeat_seed),
        "--pixel_resolution", str(best_row["pixel_res"]),
        "--crops_per_image", str(best_row["crops_per_image"]),
        "--cam_size", str(best_row["cam_size"])
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # Find model dir
    model_dirs = [os.path.join(outdir, d) for d in os.listdir(outdir) if "run_backbone_unet" in d]
    model_dirs.sort()
    if not model_dirs:
        print(f"WARNING: No model directory found in {outdir}")
        final_losses.append(float("inf"))
        final_run_dirs.append(None)
        continue
    model_dir = model_dirs[-1]
    final_run_dirs.append(model_dir)
    metrics_path = os.path.join(model_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        loss = df["val_loss"].iloc[-1]
        final_losses.append(loss)
    else:
        print(f"WARNING: No metrics.csv found in {model_dir}")
        final_losses.append(float("inf"))

# ========== Select best seed/model ==========
final_best_idx = int(np.argmin(final_losses))
final_best_dir = final_run_dirs[final_best_idx]
final_best_seed = final_best_idx
final_best_loss = final_losses[final_best_idx]

print(f"==== FINAL BEST MODEL ====")
print(f"Seed: {final_best_seed}, Model dir: {final_best_dir}, Loss: {final_best_loss:.4f}")

# ========== Continue pipeline with best model ==========
# ---- 02_data_prep.py ----
preprocessed_data_dir = os.path.join(final_best_dir, "preprocessed_data")
cmd_02 = [
    "python3", "02_data_prep.py",
    "--input_frame", INPUT_TRAIN,
    "--input_mask", INPUT_MASK,
    "--data_save_dir", preprocessed_data_dir,
    "--size", str(best_row["size"]),
    "--frames", "2",
    "--min_pixels", str(best_row["min_pixels"]),
    "--crops_per_image", str(best_row["crops_per_image"]),
    "--subsample", "1",
    "--data_seed", str(final_best_seed)
]
print(" ".join(cmd_02))
subprocess.run(cmd_02, check=True)

# ---- 03_event_classification.py GRID SEARCH ----
event_cls_dir = os.path.join(final_best_dir, "event_classification_models")
cmd_03_grid = [
    "python3", "03_event_classification.py", "grid",
    "--size", str(best_row["size"]),
    "--frames", "2",
    "--crops_per_image", str(best_row["crops_per_image"]),
    "--model_seed", str(final_best_seed),
    "--data_seed", str(final_best_seed),
    "--data_save_dir", preprocessed_data_dir,
    "--TAP_model_load_path", final_best_dir,
    "--model_save_dir", event_cls_dir,
    # add any additional arguments needed for grid search!
]
print(" ".join(cmd_03_grid))
subprocess.run(cmd_03_grid, check=True)

# ---- 03_event_classification.py FINAL TRAIN WITH BEST CLASSIFIER CONFIG ----
print("\n==== FINAL CLASSIFIER HEAD TRAINING WITH BEST PARAMS ====")
grid_csv = os.path.join(event_cls_dir, "grid_search_results.csv")
df_grid = pd.read_csv(grid_csv)
# If you want a specific metric, change "f1"
best_row_cls = df_grid.sort_values("f1", ascending=False).iloc[0]

def safeval(row, key, fallback=""):
    return str(row[key]) if key in row else str(row.get(key, fallback))

cmd_03_final = [
    "python3", "03_event_classification.py",
    "--size", safeval(best_row_cls, "size", best_row['size']),
    "--frames", "2",
    "--batchsize", safeval(best_row_cls, "batchsize", 32),
    "--training_epochs", safeval(best_row_cls, "epochs", 10),
    "--crops_per_image", safeval(best_row_cls, "crops_per_image", best_row['crops_per_image']),
    "--model_seed", str(final_best_seed),
    "--data_seed", str(final_best_seed),
    "--num_runs", "1",
    "--data_save_dir", preprocessed_data_dir,
    "--model_save_dir", event_cls_dir,
    "--model_id", "final_best_classifier",
    "--TAP_model_load_path", final_best_dir,
    "--cls_head_arch", safeval(best_row_cls, "cls_head_arch", "linear"),
    "--TAP_init", "loaded",
    "--balancing_method", safeval(best_row_cls, "balancing_method", "balanced"),
    "--balanced_sample_size", "500000"
]
print(" ".join([str(a) for a in cmd_03_final]))
subprocess.run(cmd_03_final, check=True)

# ---- 04_examine_mistaken_predictions.py ----
mistake_pred_dir = os.path.join(final_best_dir, "mistake_preds")
test_data_load_path = os.path.join(preprocessed_data_dir, "preprocessed_image_crops.pth")
cmd_04 = [
    "python3", "04_examine_mistaken_predictions.py",
    "--mistake_pred_dir", mistake_pred_dir,
    "--masks_path", INPUT_MASK,
    "--num_egs_to_show", "10",
    "--TAP_model_load_path", final_best_dir,
    "--patch_size", str(best_row["size"]),
    "--test_data_load_path", test_data_load_path,
    "--combined_model_load_dir", event_cls_dir,
    "--model_id", "final_best_classifier",
    "--cls_head_arch", safeval(best_row_cls, "cls_head_arch", "linear"),
    "--is_true_positive",
    "--is_true_negative",
    "--save_data"
]
print(" ".join(cmd_04))
subprocess.run(cmd_04, check=True)

# ========== PLOTTING ==========
print("=== Plotting grid search results ===")
plot_dir = "spatial_search_results/plots"
os.makedirs(plot_dir, exist_ok=True)

# --- Multipanel line plots ---
hyperparams = ["size", "min_pixels", "pixel_res", "crops_per_image"]
titles = ["Patch Size", "Min Pixels", "Pixel Resolution", "Crops Per Image"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
for i, param in enumerate(hyperparams):
    row, col = divmod(i, 2)
    ax = axes[row, col]
    grouped = results_df.groupby(param)["mean_val_loss"]
    means = grouped.mean()
    stds = grouped.std()
    xs = means.index.values
    ax.plot(xs, means.values, marker='o', lw=2, color='#005f73', label="Mean val_loss")
    ax.fill_between(xs, means-stds, means+stds, alpha=0.18, color='#94d2bd', label="±1 std")
    best_idx = xs[np.argmin(means.values)]
    ax.axvline(best_idx, color="#ae2012", linestyle="--", lw=2, label="Best")
    ax.set_title(f"{titles[i]}", fontsize=13)
    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(axis='both', labelsize=10)
plt.tight_layout(rect=[0,0.14,1,0.97])

# --- Separate cam_size plot ---
fig2 = plt.figure(figsize=(12, 3), dpi=120)
ax5 = fig2.add_subplot(111)
grouped = results_df.groupby("cam_size")["mean_val_loss"]
means = grouped.mean()
stds = grouped.std()
xs = means.index.values
ax5.plot(xs, means.values, marker='o', lw=2, color='#005f73', label="Mean val_loss")
ax5.fill_between(xs, means-stds, means+stds, alpha=0.18, color='#94d2bd', label="±1 std")
best_idx = xs[np.argmin(means.values)]
ax5.axvline(best_idx, color="#ae2012", linestyle="--", lw=2, label="Best")
ax5.set_title("CAM Size", fontsize=13)
ax5.set_xlabel("cam_size", fontsize=11)
ax5.set_ylabel("Validation Loss", fontsize=11)
ax5.legend(fontsize=9)
ax5.tick_params(axis='both', labelsize=10)
plt.tight_layout()
fig2.savefig(f"{plot_dir}/val_loss_vs_camsize.png")
plt.close(fig2)

fig.savefig(f"{plot_dir}/multipanel_lineplots.png")
plt.close(fig)
print(f"Saved: {plot_dir}/multipanel_lineplots.png")
print(f"Saved: {plot_dir}/val_loss_vs_camsize.png")

# --- Heatmap of pixel_res vs crops_per_image ---
heatmap_df = results_df.groupby(["pixel_res", "crops_per_image"])["mean_val_loss"].mean().unstack()
plt.figure(figsize=(7.2,5.5), dpi=120)
ax = sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu",
            cbar_kws={'label': 'Mean val_loss'},
            linewidths=0.6, linecolor='grey', annot_kws={'fontsize':11})
plt.title("Mean Validation Loss: Pixel Res. vs. Crops Per Image", fontsize=14)
plt.xlabel("Crops Per Image", fontsize=12)
plt.ylabel("Pixel Resolution", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
min_idx = np.unravel_index(np.nanargmin(heatmap_df.values), heatmap_df.shape)
rect = plt.Rectangle(
    (min_idx[1], min_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3, clip_on=False
)
ax.add_patch(rect)
plt.tight_layout()
plt.savefig(f"{plot_dir}/heatmap_pixelres_vs_cropsperimage.png")
plt.close()
print(f"Saved: {plot_dir}/heatmap_pixelres_vs_cropsperimage.png")
print("=== All plots saved in:", plot_dir)

# --- DONE ---
