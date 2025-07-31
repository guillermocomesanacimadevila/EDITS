#!/usr/bin/env python3

"""
CellFate Pipeline Report Generator & Dashboard
python 05_generate_report.py --config path/to/config.yaml --outdir output_folder
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

def safe_read_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def safe_read_json(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f)

def find_first(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""

def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def crawl_outputs(outdir):
    outputs = []
    for root, dirs, files in os.walk(outdir):
        rel_root = os.path.relpath(root, outdir)
        for f in sorted(files):
            rel_path = os.path.join(rel_root, f) if rel_root != "." else f
            if f.lower().endswith('.html'):  # Exclude previous report(s)
                continue
            outputs.append(rel_path)
    return sorted(outputs)

def group_files_by_type(files):
    groups = {
        "Config & Requirements": [],
        "Logs": [],
        "Results / Statistics": [],
        "Visualizations": [],
        "Model Weights": [],
        "Other": [],
    }
    for f in files:
        fname = f.lower()
        ext = os.path.splitext(f)[1].lower()
        if any(x in fname for x in ["requirement", "conda", "env", "config", "yaml", "yml"]) or ext in {".yml", ".yaml"}:
            groups["Config & Requirements"].append(f)
        elif ext == ".json" and ("config" in fname or "requirement" in fname):
            groups["Config & Requirements"].append(f)
        elif ext in {".log", ".txt"} and ("log" in fname or "stdout" in fname):
            groups["Logs"].append(f)
        elif ext in {".csv", ".tsv"} or ext == ".json" or "result" in fname or "metric" in fname:
            groups["Results / Statistics"].append(f)
        elif ext in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            groups["Visualizations"].append(f)
        elif ext in {".pth", ".h5", ".ckpt", ".npz"} or "weight" in fname or "model" in fname:
            groups["Model Weights"].append(f)
        else:
            groups["Other"].append(f)
    return groups

def preview_csv_table(csv_path, max_rows=150):
    try:
        df = pd.read_csv(csv_path)
        if len(df) > max_rows:
            df = df.head(max_rows)
        table = df.to_html(index=False, classes="data-table", border=0, escape=False)
        return f"<div style='overflow-x:auto'>{table}</div>"
    except Exception as e:
        return f"<div style='color:#d00'>Error reading CSV: {html_escape(str(e))}</div>"

def preview_text_file(txt_path, max_lines=40):
    try:
        with open(txt_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            preview = "".join(lines[:max_lines])
            if len(lines) > max_lines:
                preview += f"\n... (truncated, see file for more)"
        return f"<pre>{html_escape(preview)}</pre>"
    except Exception as e:
        return f"<div style='color:#d00'>Error reading file: {html_escape(str(e))}</div>"

def preview_image(img_path, rel_path):
    return f"""
    <div class="figure-wrapper" tabindex="0" role="button" aria-label="Zoom Image: {html_escape(os.path.basename(rel_path))}">
      <img src="{html_escape(rel_path)}" alt="{html_escape(os.path.basename(rel_path))}" loading="lazy"/>
      <div class="figure-caption">
        <span>{html_escape(os.path.basename(rel_path))}</span>
        <div class="download-group">
          <a href="{html_escape(rel_path)}" download class="download-btn">Download</a>
        </div>
      </div>
    </div>
    """

def preview_pdf(rel_path):
    return f"""
      <div class="figure-wrapper">
        <div class="figure-caption">
          <span>{html_escape(os.path.basename(rel_path))}</span>
          <div class="download-group">
            <a href="{html_escape(rel_path)}" download class="download-btn">Download PDF</a>
          </div>
        </div>
      </div>
    """

def preview_file_link(rel_path):
    return f'<a href="{html_escape(rel_path)}" download class="toggle-btn">{html_escape(os.path.basename(rel_path))}</a>'

def find_metrics_csv(outdir):
    for root, dirs, files in os.walk(outdir):
        for fname in files:
            if fname == "metrics.csv":
                return os.path.join(root, fname)
    return None

def find_summary_shadow_dir(outdir):
    for d in ["summary_shadow", "summary", "shadow_summary"]:
        candidate = os.path.join(outdir, d)
        if os.path.isdir(candidate):
            return candidate
    return None

def collect_summary_shadow_imgs(shadow_dir):
    imgs = []
    if shadow_dir and os.path.isdir(shadow_dir):
        for fname in sorted(os.listdir(shadow_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                imgs.append(os.path.join(os.path.basename(shadow_dir), fname))
    return imgs

# === TrainingCurvesPlotter class for automated curve plotting ===
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
class TrainingCurvesPlotter:
    def __init__(self, metrics_csv_list, outdir):
        self.outdir = os.path.join(str(outdir), "figures")
        os.makedirs(self.outdir, exist_ok=True)
        if isinstance(metrics_csv_list, str):
            metrics_csv_list = [metrics_csv_list]
        self.dfs = [pd.read_csv(f) for f in metrics_csv_list]

        # Detect if this is a per-epoch metrics CSV (from training) or a summary CSV (from pipeline)
        if 'epoch' in self.dfs[0].columns:
            self.epochs = self.dfs[0]['epoch'].values
            self.train_loss = np.stack([df['train_loss'].values for df in self.dfs])
            self.val_loss   = np.stack([df['val_loss'].values for df in self.dfs])
            # Try both train_acc/val_acc or accuracy/val_accuracy
            if 'train_acc' in self.dfs[0] and 'val_acc' in self.dfs[0]:
                self.train_acc = np.stack([df['train_acc'].values for df in self.dfs])
                self.val_acc   = np.stack([df['val_acc'].values for df in self.dfs])
            elif 'accuracy' in self.dfs[0] and 'val_accuracy' in self.dfs[0]:
                self.train_acc = np.stack([df['accuracy'].values for df in self.dfs])
                self.val_acc   = np.stack([df['val_accuracy'].values for df in self.dfs])
            else:
                self.train_acc = None
                self.val_acc = None
            self.is_per_epoch = True
        else:
            self.epochs = None
            self.train_loss = None
            self.val_loss = None
            self.train_acc = None
            self.val_acc = None
            self.is_per_epoch = False
            print("No 'epoch' column found in metrics CSVs. Per-epoch learning curves will be skipped.")

    def plot_all_multipanel(self, filename="curves_multipanel"):
        if not self.is_per_epoch:
            print("Skipping multipanel curves: no per-epoch data available.")
            return

        c_train = "#2386E6"
        c_val   = "#FC573B"
        lw = 2
        alpha_shade = 0.16
        fs_title = 17
        fs_labels = 15

        fig, axs = plt.subplots(1, 2, figsize=(16, 5.5))

        # --- LOSS PANEL ---
        ax = axs[0]
        mean_train = self.train_loss.mean(axis=0)
        std_train  = self.train_loss.std(axis=0)
        mean_val   = self.val_loss.mean(axis=0)
        std_val    = self.val_loss.std(axis=0)

        ax.plot(self.epochs, mean_train, label="Train Loss (mean)", color=c_train, linewidth=lw)
        ax.fill_between(self.epochs, mean_train-std_train, mean_train+std_train, color=c_train, alpha=alpha_shade, label="Train ± std")
        ax.plot(self.epochs, mean_val, label="Val Loss (mean)", color=c_val, linewidth=lw)
        ax.fill_between(self.epochs, mean_val-std_val, mean_val+std_val, color=c_val, alpha=alpha_shade, label="Val ± std")
        ax.set_xlabel("Epoch", fontsize=fs_labels)
        ax.set_ylabel("Loss", fontsize=fs_labels)
        ax.set_title("Training and Validation Loss", fontsize=fs_title)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.18)

        # --- ACCURACY PANEL ---
        ax = axs[1]
        if self.train_acc is not None and self.val_acc is not None:
            mean_train = self.train_acc.mean(axis=0)
            std_train  = self.train_acc.std(axis=0)
            mean_val   = self.val_acc.mean(axis=0)
            std_val    = self.val_acc.std(axis=0)

            ax.plot(self.epochs, mean_train, "--", label="Train Accuracy (mean)", color=c_train, linewidth=lw)
            ax.fill_between(self.epochs, mean_train-std_train, mean_train+std_train, color=c_train, alpha=alpha_shade, label="Train ± std")
            ax.plot(self.epochs, mean_val, "--", label="Val Accuracy (mean)", color=c_val, linewidth=lw)
            ax.fill_between(self.epochs, mean_val-std_val, mean_val+std_val, color=c_val, alpha=alpha_shade, label="Val ± std")
            ax.set_xlabel("Epoch", fontsize=fs_labels)
            ax.set_ylabel("Accuracy", fontsize=fs_labels)
            ax.set_ylim(0.5, 1.01)
            ax.set_title("Training and Validation Accuracy", fontsize=fs_title)
            ax.legend(fontsize=12)
            ax.grid(alpha=0.18)
        else:
            ax.text(0.5, 0.5, "No accuracy columns found", ha='center', va='center', fontsize=16)
            ax.set_axis_off()

        plt.tight_layout()
        fpath_png = os.path.join(self.outdir, f"{filename}.png")
        plt.savefig(fpath_png, dpi=320, bbox_inches='tight')
        plt.close()
        print(f"Multipanel loss/accuracy curves saved to {fpath_png}")

# === Main Script ===

parser = argparse.ArgumentParser(description="CellFate: Generate HTML report for a run")
parser.add_argument("--config", required=True, help="YAML config file")
parser.add_argument("--outdir", required=True, help="Output directory for this run")
parser.add_argument("--batch_outdirs", nargs='*', help="Batch output dirs (optional, for batch/TAP mode)")
args = parser.parse_args()

config = safe_read_yaml(args.config)
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

metrics_json = find_first(
    os.path.join(outdir, "metrics.json"),
    os.path.join(outdir, "classifier_metrics.json"),
)
metrics = safe_read_json(metrics_json) if metrics_json else {}

model_id = config.get("name", "-")
crop_size = config.get("size", "-")
epochs = config.get("epochs", "-")
pixel_res = config.get("pixel_resolution", "-")
backbone = config.get("backbone", "-")
mask_file = "-"
if config.get("input_mask"):
    mask0 = config.get("input_mask")
    if isinstance(mask0, (list, tuple)) and mask0:
        mask_file = os.path.basename(mask0[0])
    elif isinstance(mask0, str):
        mask_file = os.path.basename(mask0)
config_file = os.path.basename(args.config)

all_files = crawl_outputs(outdir)
file_groups = group_files_by_type(all_files)

# === AUTOMATIC: Find and plot ALL learning curves for all runs ===
metrics_csvs = []
parent_dir = os.path.abspath(os.path.join(outdir, ".."))
for run_dir in sorted(glob.glob(os.path.join(parent_dir, "*run*"))):
    metrics_csvs.extend(glob.glob(os.path.join(run_dir, "*backbone_*", "metrics.csv")))
if not metrics_csvs:
    # fallback: just use current run if no others found
    metrics_csv = find_metrics_csv(outdir)
    if metrics_csv:
        metrics_csvs = [metrics_csv]

learning_curve_img = None
if metrics_csvs:
    plotter = TrainingCurvesPlotter(metrics_csvs, outdir)
    plotter.plot_all_multipanel(filename="curves_multipanel")
    learning_curve_img = "figures/curves_multipanel.png"

learning_curves_html = ""
if learning_curve_img and os.path.exists(os.path.join(outdir, learning_curve_img)):
    learning_curves_html += f"""
    <section>
      <h2><span class="icon">📈</span><span class="gradient-title">Learning Curves (All Runs)</span></h2>
      <img src="{learning_curve_img}" alt="Training and Validation Loss/Accuracy curves" style="max-width:97%;border-radius:18px;box-shadow:0 4px 28px #14cfff22;margin-bottom:1.6em;">
    </section>
    """

# --- SHADOW SUMMARY SECTION ---
summary_shadow_dir = find_summary_shadow_dir(outdir)
summary_imgs = collect_summary_shadow_imgs(summary_shadow_dir)
if summary_imgs:
    summary_shadow_html = """
    <section>
      <h2><span class="icon">🕳️</span><span class="gradient-title">Summary Shadow Plots (Across Runs)</span></h2>
      <div class="grid" style="margin-bottom:2em;">
    """
    for rel_path in summary_imgs:
        summary_shadow_html += preview_image(rel_path, rel_path)
    summary_shadow_html += "</div></section>\n"
else:
    summary_shadow_html = ""

# --- HTML for the rest of the groups
def render_section(title, files, outdir, show_preview=True):
    if not files:
        return ""
    html = f'<section aria-labelledby="{html_escape(title.lower().replace(" ", "-"))}-title" tabindex="0">\n'
    html += f'  <h2 id="{html_escape(title.lower().replace(" ", "-"))}-title"><span class="icon" aria-hidden="true">📁</span><span class="gradient-title">{html_escape(title)}</span></h2>\n'
    if title == "Visualizations":
        html += '<div class="grid" style="margin-bottom:2em;">\n'
        for rel_path in files:
            ext = os.path.splitext(rel_path)[1].lower()
            if ext in [".png", ".jpg", ".jpeg"]:
                html += preview_image(rel_path, rel_path)
            elif ext == ".pdf":
                html += preview_pdf(rel_path)
        html += "</div>\n"
        return html + "</section>\n"
    for rel_path in files:
        ext = os.path.splitext(rel_path)[1].lower()
        html += f'<div style="margin-bottom:2.3em;">\n'
        html += f'<h3 style="margin-bottom:0.1em;">{html_escape(rel_path)}</h3>\n'
        html += preview_file_link(rel_path) + "\n"
        if show_preview:
            full_path = os.path.join(outdir, rel_path)
            if ext in [".csv", ".tsv"]:
                html += preview_csv_table(full_path)
            elif ext in [".txt", ".log", ".json", ".yaml", ".yml"]:
                html += preview_text_file(full_path)
        html += "</div>\n"
    html += "</section>\n"
    return html

sections_html = ""
order = ["Config & Requirements", "Logs", "Results / Statistics", "Visualizations", "Model Weights", "Other"]
for group in order:
    preview = group not in ["Model Weights", "Other"]
    sections_html += render_section(group, file_groups[group], outdir, show_preview=preview)

def format_metric(met, key):
    val = met.get(key, None)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return f"{val:.3f}"

clf_accuracy = format_metric(metrics, "accuracy")
clf_precision = format_metric(metrics, "precision")
clf_recall = format_metric(metrics, "recall")
clf_f1 = format_metric(metrics, "f1")
clf_auc = format_metric(metrics, "auc")

summary_text = (
    f"Run completed for model <b>{html_escape(model_id)}</b>. "
    f"Backbone: <b>{html_escape(backbone)}</b>, Crop: <b>{html_escape(crop_size)}</b>, "
    f"Epochs: <b>{html_escape(epochs)}</b>, Pixel Res: <b>{html_escape(pixel_res)}</b>."
)
if clf_accuracy != "-":
    summary_text += f" Final accuracy: <b>{clf_accuracy}</b>."
all_outputs_zip = find_first(os.path.join(outdir, "all_outputs.zip"))
if all_outputs_zip:
    zip_rel = os.path.relpath(all_outputs_zip, outdir)
    summary_text += f" Download all outputs as a <a href='{zip_rel}' download>zip file</a>."

if all_outputs_zip:
    zip_button_html = f'<a href="{os.path.relpath(all_outputs_zip, outdir)}" download class="toggle-btn" aria-label="Download all outputs as ZIP">⬇️ Download All Outputs (.zip)</a>'
else:
    zip_button_html = '<span class="toggle-btn" style="opacity:0.5;pointer-events:none;cursor:not-allowed;">⬇️ Download All Outputs (.zip)</span>'

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CellFate Pipeline Report</title>
  <meta name="description" content="Pipeline report for CellFate — all statistics, figures, and results in one modern, professional dashboard.">
  <link rel="icon" href="https://avatars.githubusercontent.com/u/10752544?s=200&v=4" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet" crossorigin="anonymous" />
  <style>
    :root {{
      --font-sans: 'Inter', 'Segoe UI', Arial, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --accent: #38e1ff;
      --accent2: #38ffba;
      --accent-dark: #2f97c1;
      --accent-gradient: linear-gradient(90deg, #38e1ff 0%, #38ffba 100%);
      --bg: #f7fafd;
      --bg-dark: #181d23;
      --card-bg: rgba(255,255,255,0.90);
      --card-bg-dark: rgba(28,32,40,0.89);
      --header-bg: linear-gradient(110deg, #38e1ff 0%, #49e3c7 100%);
      --header-bg-dark: linear-gradient(110deg, #222c3c 0%, #1c3d4b 100%);
      --border-radius: 28px;
      --shadow: 0 8px 32px 0 rgba(40, 61, 110, 0.13), 0 1.5px 7px rgba(44,61,94,0.11);
      --shadow-strong: 0 16px 45px rgba(40, 61, 110, 0.13), 0 4px 15px rgba(44,61,94,0.12);
      --transition: all 0.34s cubic-bezier(.77,0,.18,1);
      --text: #222936;
      --text-dark: #e7edf5;
      --kpi-glow: #38e1ff60;
      --kpi-glow-dark: #38ffba88;
      --table-head: #f2fbfd;
      --table-head-dark: #233143;
      --table-border: #d5f1fb;
      --table-border-dark: #2c3c4e;
      --footer-bg: #f7fcfe;
      --footer-bg-dark: #1d2736;
      --modal-bg: rgba(0, 0, 0, 0.86);
      --avatar-bg: #1cd1e9;
    }}
    html {{ scroll-behavior: smooth; font-size: 16px; }}
    body {{
      font-family: var(--font-sans);
      margin: 0; min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      transition: var(--transition);
      overflow-x: hidden;
    }}
    .dark-mode {{
      background: var(--bg-dark);
      color: var(--text-dark);
    }}
    header {{
      position: relative;
      z-index: 30;
      padding: 2.3rem 0 0.4rem 0;
      background: var(--header-bg);
      border-bottom-left-radius: var(--border-radius);
      border-bottom-right-radius: var(--border-radius);
      box-shadow: var(--shadow);
      min-height: 145px;
      transition: background 0.4s;
      overflow: visible;
    }}
    .dark-mode header {{ background: var(--header-bg-dark); }}
    .hero-content {{
      display: flex; align-items: center; justify-content: space-between;
      max-width: 1220px; margin: 0 auto; padding: 0 2.4rem;
    }}
    .header-brand {{
      display: flex; align-items: center; gap: 1.5rem;
      font-weight: 900; font-size: 2.55rem; color: #fff;
      letter-spacing: 1.5px;
      filter: drop-shadow(0 2px 10px #1ec6ee22);
      user-select: none;
    }}
    .brand-avatar {{
      background: var(--avatar-bg);
      border-radius: 50%;
      width: 54px; height: 54px;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 3px 16px #36ecfc44;
      font-size: 2.1rem; font-weight: 800; color: #fff;
    }}
    .header-right {{
      display: flex; align-items: center; gap: 1.3rem;
    }}
    .company-tagline {{
      color: #d9fff9;
      font-size: 1.17rem;
      font-weight: 500;
      letter-spacing: 0.07em;
      margin-right: 0.4rem;
      opacity: 0.78;
      font-family: var(--font-mono);
    }}
    .toggle-btn {{
      background: linear-gradient(90deg,#fff 0%, #eafaff 100%);
      color: #299acf;
      border: none;
      padding: 0.58rem 1.38rem;
      border-radius: 2em;
      font-weight: 700;
      font-size: 1.15rem;
      cursor: pointer;
      box-shadow: 0 2.5px 18px rgba(0,160,240,0.10);
      transition: var(--transition);
      outline-offset: 3px;
      border: 1.3px solid #e6faff;
      user-select: none;
      display: flex; align-items: center; gap: 0.38em;
    }}
    .toggle-btn:hover {{ background: var(--accent-dark); color: #fff; }}
    .dark-mode .toggle-btn {{
      background: linear-gradient(90deg,#1f2c36 0%, #233243 100%);
      color: #c9f8ff;
      border: 1.3px solid #223b51;
      box-shadow: 0 0 20px #38eaff88;
    }}
    main {{
      max-width: 1200px;
      margin: -22px auto 3.8rem auto;
      padding: 0 1.7rem;
      user-select: text;
      z-index: 5;
      position: relative;
    }}
    .summary-box {{
      border-radius: 24px;
      box-shadow: var(--shadow-strong);
      background: rgba(215,250,255,0.68);
      backdrop-filter: blur(7px) saturate(180%);
      padding: 2.1em 2.7em;
      margin-bottom: 2.8em;
      font-size: 1.18em;
      font-weight: 700;
      color: #167199;
      border: 1.5px solid #caf4ff;
      position: relative;
      animation: fadeIn 1s cubic-bezier(.53,.09,.29,1) forwards;
    }}
    .dark-mode .summary-box {{
      background: rgba(20,34,42,0.80);
      color: #a4f7ff;
      border: 1.7px solid #285975;
      box-shadow: 0 6px 34px #1a4a7dbb;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(32px);}}
      to {{ opacity: 1; transform: translateY(0);}}
    }}
    section {{
      background: var(--card-bg);
      border-radius: var(--border-radius);
      box-shadow: var(--shadow-strong);
      margin-bottom: 3.1rem;
      padding: 2.4rem 2.2rem 2rem 2.2rem;
      position: relative;
      overflow: hidden;
      animation: fadeInUp 0.8s cubic-bezier(.33,1.15,.68,1) forwards;
      transition: background 0.5s, box-shadow 0.4s;
      backdrop-filter: blur(6px) saturate(170%);
    }}
    .dark-mode section {{ background: var(--card-bg-dark); }}
    @keyframes fadeInUp {{
      from {{ opacity: 0; transform: translateY(28px);}}
      to {{ opacity: 1; transform: translateY(0);}}
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 2.1em;
      margin-bottom: 1.5em;
      user-select: none;
    }}
    .kpi-card {{
      background: linear-gradient(120deg, #eafafd 50%, #bafcff 100%);
      border-radius: 1.9em;
      box-shadow: 0 5px 30px #43e4ff16;
      padding: 1.5em 1.7em;
      text-align: center;
      font-size: 1.25em;
      color: #185680;
      font-weight: 800;
      letter-spacing: 0.02em;
      display: flex; flex-direction: column; align-items: center;
      gap: 0.45em; transition: var(--transition);
      cursor: pointer; position: relative; overflow: hidden;
      border: 1.5px solid #c7f9ff88;
      backdrop-filter: blur(6px);
    }}
    .kpi-card .kpi-label {{
      font-size: 1.02em; color: #41d0fc; font-weight: 600;
      margin-bottom: 0.38em;
      letter-spacing: 0.04em;
    }}
    .kpi-card .kpi-icon {{
      font-size: 2.4em;
      margin-bottom: 0.21em;
      color: var(--accent-dark);
      filter: drop-shadow(0 3px 7px #37ebff41);
    }}
    .kpi-card:hover, .kpi-card:focus {{
      background: linear-gradient(110deg,#e1fff8 50%,#e1eaff 100%);
      color: var(--accent-dark);
      box-shadow: 0 10px 50px #1ad6fa4b;
      border-color: var(--accent);
      transform: translateY(-4px) scale(1.03);
    }}
    .dark-mode .kpi-card {{
      background: linear-gradient(120deg, #263849 50%, #232e39 100%);
      color: #cafaff;
      border: 1.5px solid #3df4ff33;
      box-shadow: 0 5px 32px #4eefff30;
    }}
    .dark-mode .kpi-card .kpi-label {{ color: #46e7ff; }}
    .dark-mode .kpi-card .kpi-icon {{ color: #38eaff; }}
    h2 {{
      margin-top: 0;
      font-size: 2rem;
      font-weight: 800;
      display: flex;
      align-items: center;
      gap: 0.78em;
      margin-bottom: 1.65rem;
      line-height: 1.2;
      letter-spacing: 0.04em;
      user-select: none;
      color: var(--accent-dark);
    }}
    .gradient-title {{
      background: var(--accent-gradient);
      background-clip: text;
      -webkit-background-clip: text;
      color: transparent;
      -webkit-text-fill-color: transparent;
      font-size: inherit;
      font-weight: inherit;
      letter-spacing: inherit;
      user-select: text;
    }}
    .icon {{ font-size: 1.5em; }}
    h2:after {{
      content: '';
      display: block;
      flex: 1 1 0%;
      height: 3.5px;
      margin-left: 1.1em;
      border-radius: 3.5px;
      background: var(--accent-gradient);
      min-width: 8vw;
      opacity: 0.4;
      margin-top: 0.18em;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 2.2rem;
      margin-bottom: 2.7rem;
    }}
    .figure-wrapper {{
      position: relative;
      border-radius: 1.7em;
      overflow: hidden;
      box-shadow: 0 7px 34px 0 #48e2ff1c;
      border: 3px solid #b5e6ff33;
      background: #f6fafdcc;
      cursor: pointer;
      transition: var(--transition);
      display: flex; flex-direction: column;
      min-height: 240px;
      user-select: none;
    }}
    .figure-wrapper:hover, .figure-wrapper:focus {{
      transform: scale(1.045) translateY(-6.5px);
      box-shadow: 0 18px 60px #3ae4fc2f;
      border-color: var(--accent-dark);
      z-index: 10;
    }}
    .figure-wrapper img {{
      width: 100%; height: auto; aspect-ratio: 16 / 9;
      object-fit: cover;
      border-radius: 1.7em 1.7em 0 0;
      transition: filter 0.25s;
      user-select: none;
      pointer-events: none;
      display: block;
    }}
    .figure-wrapper:hover img {{ filter: brightness(1.11); }}
    .figure-caption {{
      padding: 0.95em 1.2em 1.25em 1.2em;
      font-weight: 700;
      font-size: 1.08rem;
      color: #18406d;
      background: #def6ffcc;
      border-radius: 0 0 1.4em 1.4em;
      user-select: text;
      box-shadow: inset 0 1px 1px #9ad1fb80;
      transition: background 0.3s, color 0.3s;
      display: flex; align-items: center; gap: 0.7em;
    }}
    .download-group {{ display: flex; gap: 0.72em; }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      margin-bottom: 2em;
      font-size: 1.08em;
      border-radius: 18px;
      overflow: hidden;
      background: #fff;
      box-shadow: 0 4px 18px #91dfff33;
      user-select: text;
      transition: box-shadow 0.33s;
    }}
    table:hover {{ box-shadow: 0 10px 44px #1dcfff38; }}
    thead tr {{
      background: var(--table-head);
      font-weight: 700; font-size: 1.04em;
      border-bottom: 3px solid var(--table-border);
      position: sticky; top: 0; z-index: 5; user-select: none; cursor: pointer;
      transition: background-color 0.25s;
      box-shadow: 0 2.5px 6px #1edfff21;
    }}
    thead tr:hover {{ background-color: #c9f0ff; }}
    tbody tr:nth-child(even) {{ background: #f6fbff; }}
    tbody tr:hover {{ background: #d6f0ff; transition: background-color 0.22s; }}
    th, td {{
      padding: 1em 1.1em; text-align: center;
      border-bottom: 1.4px solid var(--table-border);
      vertical-align: middle;
      font-weight: 500;
      color: #155d90;
    }}
    th {{
      font-weight: 800; color: #007bbf; position: relative;
    }}
    th.sortable:hover {{ color: var(--accent-dark); cursor: pointer;}}
    th .sort-arrow {{
      position: absolute; right: 0.9em; top: 50%;
      transform: translateY(-50%);
      font-size: 0.82em; opacity: 0.33; transition: opacity 0.2s;
      user-select: none;
    }}
    th.sortable:hover .sort-arrow {{ opacity: 0.6;}}
    th.sorted-asc .sort-arrow {{ opacity: 1; transform: translateY(-50%) rotate(180deg); color: var(--accent-dark);}}
    th.sorted-desc .sort-arrow {{ opacity: 1; color: var(--accent-dark);}}
    .dark-mode table {{ background: #222f41; box-shadow: 0 6px 28px #1dcfff1a;}}
    .dark-mode thead tr {{ background: var(--table-head-dark); color: #90f8ff; border-bottom: 3px solid var(--table-border-dark);}}
    .dark-mode tbody tr:nth-child(even) {{ background: #223146;}}
    .dark-mode th, .dark-mode td {{ border-bottom: 1.4px solid var(--table-border-dark); color: #88e2f5;}}
    @media (max-width: 780px) {{
      table {{ font-size: 0.95em;}}
      th, td {{ padding: 0.7em 0.4em;}}
    }}
    .download-btn {{
      background: var(--accent);
      color: white;
      border: none;
      padding: 0.35em 0.8em;
      border-radius: 0.8em;
      font-weight: 700;
      font-size: 0.95em;
      cursor: pointer;
      text-decoration: none;
      user-select: none;
      transition: background 0.3s;
      box-shadow: 0 3px 14px #16b7ee3b;
      display: inline-flex; align-items: center; justify-content: center; gap: 0.2em;
    }}
    .download-btn:hover, .download-btn:focus-visible {{
      background: var(--accent-dark); box-shadow: 0 7px 24px #18eaff56; outline: none;
    }}
    pre, code {{
      font-family: var(--font-mono);
      background: #eafdff;
      color: #007bbf;
      padding: 1.1em 1.2em;
      font-size: 1.08em;
      border-radius: 0.9em;
      box-shadow: 0 3px 14px #00b8d60a;
      margin: 0.2em 0 1.1em 0;
      overflow-x: auto; line-height: 1.6; user-select: text;
    }}
    .dark-mode pre, .dark-mode code {{
      background: #1a3240; color: #31e5ff;
      border: none; box-shadow: 0 3px 22px #0ef1ffaa;
    }}
    #modal-zoom {{
      position: fixed; top:0; left:0; width:100vw; height:100vh;
      background: var(--modal-bg); display: none;
      align-items: center; justify-content: center; z-index: 9999;
      cursor: zoom-out; padding: 2em; user-select: none;
      animation: fadeInModal 0.28s ease forwards;
    }}
    #modal-zoom img {{
      max-width: 97vw; max-height: 96vh; border-radius: 2.1em;
      box-shadow: 0 0 50px #1be5ffdd;
      user-select: none; pointer-events: none;
      filter: drop-shadow(0 0 8px #00d8ffcc);
      transition: filter 0.3s;
    }}
    #modal-zoom:hover img {{ filter: drop-shadow(0 0 16px #00e1ff); }}
    @keyframes fadeInModal {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
    footer {{
      padding: 2.2rem 0 1.7rem 0;
      text-align: center;
      font-weight: 700;
      font-size: 1.11rem;
      color: #2673aa;
      user-select: none;
      letter-spacing: 0.04em;
      border-top: 2px solid #d1e9ff88;
      background: var(--footer-bg);
      transition: background 0.3s, color 0.3s;
      margin-top: 0.5rem;
    }}
    .dark-mode footer {{ background: var(--footer-bg-dark); color: #58c3ffcc; border-top-color: #1852a5cc;}}
    @media (max-width: 820px) {{
      main {{padding: 0 0.8rem;}}
      .hero-content {{padding: 0 1.1rem;}}
      section {{padding: 1.2rem 1rem 1rem 1rem;}}
      .kpi-grid {{grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1.1em;}}
      .figure-wrapper {{min-height: 160px;}}
      h2 {{font-size: 1.3rem;}}
    }}
  </style>
</head>
<body>
  <header>
    <h1>🔬 <span style="letter-spacing:0.06em">CellFate REPORT</span></h1>
    <button class="toggle-btn" id="theme-toggle" aria-label="Toggle dark mode" title="Toggle light/dark theme">🌙 Toggle Theme</button>
  </header>
  <main>
    <div class="summary-box" role="region" aria-live="polite" aria-atomic="true">
      <span style="font-size:1.24em; font-weight:800;">Summary:</span>
      {summary_text}
    </div>
    <div class="batch-zip" role="region">
      {zip_button_html}
    </div>
    <section aria-labelledby="run-config-title" tabindex="0">
      <h2 id="run-config-title"><span class="icon" aria-hidden="true">📋</span><span class="gradient-title">Run Configuration</span></h2>
      <div class="kpi-grid" role="list" aria-label="Run configuration key performance indicators">
        <div class="kpi-card" role="listitem"><div class="kpi-label">Model ID</div>{model_id}</div>
        <div class="kpi-card" role="listitem"><div class="kpi-label">Crop Size</div>{crop_size}</div>
        <div class="kpi-card" role="listitem"><div class="kpi-label">Epochs</div>{epochs}</div>
        <div class="kpi-card" role="listitem"><div class="kpi-label">Pixel Resolution</div>{pixel_res}</div>
        <div class="kpi-card" role="listitem"><div class="kpi-label">Backbone</div>{backbone}</div>
        <div class="kpi-card" role="listitem"><div class="kpi-label">Mask File</div><span style="font-size:0.98em; word-break: break-word;">{mask_file}</span></div>
        <div class="kpi-card" role="listitem"><div class="kpi-label">Config File</div><span style="font-size:0.98em; word-break: break-word;">{config_file}</span></div>
      </div>
    </section>
    {learning_curves_html}
    {summary_shadow_html}
    {sections_html}
  </main>
  <footer>
    <strong>CellFate</strong> &mdash; &copy; 2025
  </footer>
  <div id="modal-zoom" role="dialog" aria-modal="true" aria-label="Zoomed image view" tabindex="-1">
    <img src="" alt="" />
  </div>
    </section>
    {learning_curves_html}
    {sections_html}
  </main>
  <footer>
    <strong>CellFate</strong> &mdash; &copy; 2025
  </footer>
    <strong>CellFate</strong> &mdash; &copy; 2025
  </footer>
  <div id="modal-zoom" role="dialog" aria-modal="true" aria-label="Zoomed image view" tabindex="-1">
    <img src="" alt="" />
  </div>
  <script>
    (function() {{
      const themeBtn = document.getElementById('theme-toggle');
      if (!themeBtn) return;
      function updateButtonText() {{
        const isDark = document.body.classList.contains('dark-mode');
        themeBtn.textContent = isDark ? '☀️ Theme' : '🌙 Theme';
      }}
      themeBtn.addEventListener('click', function () {{
        document.body.classList.toggle('dark-mode');
        document.body.style.transition = "background 0.4s, color 0.4s";
        setTimeout(() => {{ document.body.style.transition = ""; }}, 650);
        const isDark = document.body.classList.contains('dark-mode');
        localStorage.setItem('cf_dark_mode', isDark ? '1' : '0');
        updateButtonText();
      }});
      const savedMode = localStorage.getItem('cf_dark_mode');
      if (savedMode === '1') document.body.classList.add('dark-mode');
      updateButtonText();
    }})();
    (function() {{
      const modal = document.getElementById('modal-zoom');
      const modalImg = modal.querySelector('img');
      document.querySelectorAll('.figure-wrapper img').forEach(img => {{
        img.addEventListener('click', () => {{
          modalImg.src = img.src;
          modalImg.alt = img.alt || '';
          modal.style.display = 'flex';
          modal.focus();
        }});
      }});
      document.querySelectorAll('.figure-wrapper').forEach(wrapper => {{
        wrapper.addEventListener('keydown', e => {{
          if (e.key === 'Enter' || e.key === ' ') {{
            e.preventDefault();
            const img = wrapper.querySelector('img');
            img.click();
          }}
        }});
      }});
      modal.addEventListener('click', e => {{
        if (e.target === modal) {{
          modal.style.display = 'none';
          modalImg.src = '';
          modalImg.alt = '';
        }}
      }});
      document.addEventListener('keydown', e => {{
        if (e.key === 'Escape' && modal.style.display === 'flex') {{
          modal.style.display = 'none';
          modalImg.src = '';
          modalImg.alt = '';
        }}
      }});
    }})();
    (function() {{
      function getCellValue(row, idx) {{
        return row.cells[idx].innerText || row.cells[idx].textContent;
      }}
      function comparer(idx, asc) {{
        return (a, b) => {{
          const v1 = getCellValue(a, idx).trim();
          const v2 = getCellValue(b, idx).trim();
          const num1 = parseFloat(v1.replace(/[^0-9.\-]/g, ''));
          const num2 = parseFloat(v2.replace(/[^0-9.\-]/g, ''));
          if (!isNaN(num1) && !isNaN(num2)) {{
            return (num1 - num2) * (asc ? 1 : -1);
          }}
          return v1.localeCompare(v2) * (asc ? 1 : -1);
        }};
      }}
      document.querySelectorAll('table').forEach(table => {{
        const thead = table.tHead;
        if (!thead) return;
        [...thead.rows[0].cells].forEach((th, idx) => {{
          if (th.classList.contains('no-sort')) return;
          th.classList.add('sortable');
          const arrow = document.createElement('span');
          arrow.classList.add('sort-arrow');
          arrow.innerHTML = '▲';
          th.appendChild(arrow);
          let asc = true;
          th.addEventListener('click', () => {{
            thead.querySelectorAll('th').forEach(header => {{
              header.classList.remove('sorted-asc', 'sorted-desc');
            }});
            const tbody = table.tBodies[0];
            const rows = Array.from(tbody.rows);
            rows.sort(comparer(idx, asc));
            rows.forEach(row => tbody.appendChild(row));
            th.classList.toggle('sorted-asc', asc);
            th.classList.toggle('sorted-desc', !asc);
            asc = !asc;
          }});
        }});
      }});
    }})();
  </script>
</body>
</html>
"""

## === Write report ===
html_out = HTML_TEMPLATE.format(
    summary_text=summary_text,
    zip_button_html=zip_button_html,
    model_id=html_escape(model_id),
    crop_size=html_escape(crop_size),
    epochs=html_escape(epochs),
    pixel_res=html_escape(pixel_res),
    backbone=html_escape(backbone),
    mask_file=html_escape(mask_file),
    config_file=html_escape(config_file),
    learning_curves_html=learning_curves_html,
    summary_shadow_html=summary_shadow_html,
    sections_html=sections_html,
)

report_path = os.path.join(outdir, "report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_out)

print(f"Report saved to {report_path}")
