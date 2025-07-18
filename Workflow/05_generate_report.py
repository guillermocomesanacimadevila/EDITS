import os
import sys
import yaml
import json
import glob
import argparse
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────
#   Utility functions: Safe readers for all file types
# ────────────────────────────────────────────────────────────────

def safe_read_yaml(yaml_path):
    """Read YAML or return empty dict if missing."""
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def safe_read_json(json_path):
    """Read JSON or return empty dict if missing."""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f)

def safe_read_csv(csv_path):
    """Read CSV or return empty DataFrame if missing."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)

def find_first(*paths):
    """Return the first path that exists from given paths, else empty string."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""

def html_escape(s):
    """HTML-escape a string."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ────────────────────────────────────────────────────────────────
#   Parse CLI args
# ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CELLFLOW: Generate HTML report for a run")
parser.add_argument("--config", required=True, help="YAML config file")
parser.add_argument("--outdir", required=True, help="Output directory for this run")
parser.add_argument("--batch_outdirs", nargs='*', help="Batch output dirs (optional, for batch/TAP mode)")
args = parser.parse_args()

# ────────────────────────────────────────────────────────────────
#   Load config and define paths
# ────────────────────────────────────────────────────────────────

config = safe_read_yaml(args.config)
outdir = args.outdir
figdir = os.path.join(outdir, "figures")
os.makedirs(figdir, exist_ok=True)

# Find metrics JSON file (classifier_metrics.json or metrics.json)
metrics_json = find_first(
    os.path.join(outdir, "metrics.json"),
    os.path.join(outdir, "classifier_metrics.json"),
)
metrics = safe_read_json(metrics_json) if metrics_json else {}

# ────────────────────────────────────────────────────────────────
#   Key config values and file references for the report
# ────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────
#   Figure and output file existence checks
# ────────────────────────────────────────────────────────────────

def fig_file(basename):
    """Return (png, pdf) paths for a figure if they exist, else empty string."""
    png = os.path.join(figdir, basename + ".png")
    pdf = os.path.join(figdir, basename + ".pdf")
    return (png if os.path.exists(png) else "", pdf if os.path.exists(pdf) else "")

epoch_curve, epoch_curve_pdf = fig_file("loss_acc_no_legend")
loss_curve, loss_curve_pdf = epoch_curve, epoch_curve_pdf
accuracy_curve, accuracy_curve_pdf = epoch_curve, epoch_curve_pdf

confusion_matrix_png, confusion_matrix_pdf = fig_file("confusion_matrix")
roc_curve_png, roc_curve_pdf = fig_file("roc_curve")
pr_curve_png, pr_curve_pdf = fig_file("pr_curve")
perclass_png, perclass_pdf = fig_file("per_class_metrics")

iou_hist_png, iou_hist_pdf = fig_file("iou_histogram")
seg_overlay_png, seg_overlay_pdf = fig_file("segmentation_overlay")
montage_png, montage_pdf = fig_file("montage_examples")
eventcount_png, eventcount_pdf = fig_file("event_count_per_frame")

# CSV and zip outputs
training_metrics_csv = find_first(os.path.join(outdir, "training_metrics.csv"))
confusion_matrix_csv = find_first(os.path.join(outdir, "confusion_matrix.csv"))
classifier_stats_csv = find_first(os.path.join(outdir, "classifier_metrics.csv"))
perclass_csv = find_first(os.path.join(outdir, "per_class_metrics.csv"))
tap_csv = find_first(os.path.join(outdir, "tap_metrics.csv"))
all_outputs_zip = find_first(os.path.join(outdir, "all_outputs.zip"))

# ────────────────────────────────────────────────────────────────
#   Classification KPIs (if present)
# ────────────────────────────────────────────────────────────────

clf_accuracy = f"{metrics.get('accuracy', np.nan):.3f}" if "accuracy" in metrics and metrics.get('accuracy') is not None else "-"
clf_precision = f"{metrics.get('precision', np.nan):.3f}" if "precision" in metrics and metrics.get('precision') is not None else "-"
clf_recall = f"{metrics.get('recall', np.nan):.3f}" if "recall" in metrics and metrics.get('recall') is not None else "-"
clf_f1 = f"{metrics.get('f1', np.nan):.3f}" if "f1" in metrics and metrics.get('f1') is not None else "-"
clf_auc = f"{metrics.get('auc', np.nan):.3f}" if "auc" in metrics and metrics.get('auc') is not None else "-"

# ────────────────────────────────────────────────────────────────
#   Per-class metrics (table HTML)
# ────────────────────────────────────────────────────────────────

perclass_rows = ""
if perclass_csv and os.path.exists(perclass_csv):
    pc_df = safe_read_csv(perclass_csv)
    for _, row in pc_df.iterrows():
        perclass_rows += (
            f"<tr><td>{html_escape(row.get('class',''))}</td>"
            f"<td>{row.get('count','')}</td>"
            f"<td>{row.get('precision','')}</td>"
            f"<td>{row.get('recall','')}</td>"
            f"<td>{row.get('f1','')}</td></tr>\n"
        )

# ────────────────────────────────────────────────────────────────
#   TAP batch metrics (table HTML)
# ────────────────────────────────────────────────────────────────

tap_rows = ""
if tap_csv and os.path.exists(tap_csv):
    tap_df = safe_read_csv(tap_csv)
    for _, row in tap_df.iterrows():
        tap_rows += (
            f"<tr><td>{html_escape(row.get('file',''))}</td>"
            f"<td>{row.get('TP','')}</td>"
            f"<td>{row.get('FP','')}</td>"
            f"<td>{row.get('FN','')}</td>"
            f"<td>{row.get('TAP_score','')}</td></tr>\n"
        )

# ────────────────────────────────────────────────────────────────
#   Sample predictions montage figure (if available)
# ────────────────────────────────────────────────────────────────

sample_predictions_html = ""
if montage_png and os.path.exists(montage_png):
    sample_predictions_html = f"""
    <figure>
      <img src="{os.path.relpath(montage_png, outdir)}" alt="Prediction Montage" />
      <figcaption>Prediction Examples</figcaption>
    </figure>
    """

# ────────────────────────────────────────────────────────────────
#   TAP overlays (if available)
# ────────────────────────────────────────────────────────────────

tap_overlays_html = ""
if seg_overlay_png and os.path.exists(seg_overlay_png):
    tap_overlays_html += f"""
    <figure>
      <img src="{os.path.relpath(seg_overlay_png, outdir)}" alt="Segmentation Overlay" />
      <figcaption>Segmentation Overlay</figcaption>
    </figure>
    """
if iou_hist_png and os.path.exists(iou_hist_png):
    tap_overlays_html += f"""
    <figure>
      <img src="{os.path.relpath(iou_hist_png, outdir)}" alt="IoU Histogram" />
      <figcaption>IoU Histogram</figcaption>
    </figure>
    """
if eventcount_png and os.path.exists(eventcount_png):
    tap_overlays_html += f"""
    <figure>
      <img src="{os.path.relpath(eventcount_png, outdir)}" alt="Event Count per Frame" />
      <figcaption>Events per Frame</figcaption>
    </figure>
    """

# ────────────────────────────────────────────────────────────────
#   Extra content for the future, or user extensions
# ────────────────────────────────────────────────────────────────

extra_content = ""  # Add more sections if needed

# ────────────────────────────────────────────────────────────────
#   Build a summary text for the top of the report
# ────────────────────────────────────────────────────────────────

summary_text = (
    f"Run completed for model <b>{html_escape(model_id)}</b>. "
    f"Backbone: <b>{html_escape(backbone)}</b>, Crop: <b>{html_escape(crop_size)}</b>, "
    f"Epochs: <b>{html_escape(epochs)}</b>, Pixel Res: <b>{html_escape(pixel_res)}</b>."
)
if clf_accuracy and clf_accuracy != "-":
    summary_text += f" Final accuracy: <b>{clf_accuracy}</b>."
if all_outputs_zip:
    summary_text += f" Download all outputs as a zip file."

# ────────────────────────────────────────────────────────────────
#   Determine pipeline mode for hiding/showing sections in HTML
# ────────────────────────────────────────────────────────────────

pipeline_mode = 0 if clf_accuracy != "-" else 1

# --- HTML Template (insert your HTML here or read from file) ---
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CELLFLOW Pipeline Report</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet" crossorigin="anonymous">
  <style>
    :root {
      --font-sans: 'Inter', 'Segoe UI', Arial, sans-serif;
      --accent: #38e1ff;
      --accent2: #38ffba;
      --accent-dark: #219ecb;
      --bg: #f7fafd;
      --bg-dark: #181d23;
      --card-bg: rgba(255,255,255,0.96);
      --card-bg-dark: rgba(28,32,40,0.96);
      --header-bg: linear-gradient(90deg, #48d5ff 0%, #3e73ff 100%);
      --header-bg-dark: linear-gradient(90deg, #202b3a 0%, #1a3446 100%);
      --border-radius: 22px;
      --shadow: 0 8px 32px 0 rgba(40, 61, 110, 0.11), 0 1.5px 6px rgba(44,61,94,0.08);
      --transition: all 0.25s cubic-bezier(.77,0,.18,1);
      --text: #222936;
      --text-dark: #e7edf5;
      --divider: #eaf2f7;
      --divider-dark: #223041;
      --metric-bg: #eafdff;
      --metric-bg-dark: #223041;
      --table-head: #f2fbfd;
      --table-head-dark: #233143;
      --table-border: #d5f1fb;
      --table-border-dark: #2c3c4e;
      --footer-bg: #f7fcfe;
      --footer-bg-dark: #1d2736;
    }
    html { scroll-behavior: smooth; }
    body {
      font-family: var(--font-sans); margin: 0;
      background: var(--bg); color: var(--text);
      transition: var(--transition); min-height: 100vh; letter-spacing: 0.01em;
    }
    .dark-mode { background: var(--bg-dark); color: var(--text-dark); }
    header {
      position: sticky; top: 0; z-index: 20;
      background: var(--header-bg); color: #fff;
      padding: 1.5rem 2rem 1.2rem 2rem;
      display: flex; justify-content: space-between; align-items: center;
      border-bottom-left-radius: var(--border-radius); border-bottom-right-radius: var(--border-radius);
      box-shadow: var(--shadow); transition: var(--transition);
    }
    .dark-mode header { background: var(--header-bg-dark); }
    header h1 {
      margin: 0; font-size: 2.15rem; font-weight: 800;
      letter-spacing: 1.5px; display: flex; align-items: center; gap: 0.6em;
      filter: drop-shadow(0 1px 5px rgba(30,200,250,0.08));
    }
    .toggle-btn {
      background: linear-gradient(90deg,#fff 0%, #eafaff 100%);
      color: #299acf; border: none; padding: 0.48rem 1.15rem;
      border-radius: 1.7em; font-weight: 700; font-size: 1.07rem; cursor: pointer;
      box-shadow: 0 2px 10px rgba(0,160,240,0.07);
      transition: background 0.22s, color 0.22s, box-shadow 0.22s;
      outline: none; border: 1.2px solid #e6faff;
    }
    .toggle-btn:focus {
      outline: 3px solid var(--accent);
      outline-offset: 2px;
    }
    .toggle-btn:hover {
      background: var(--accent-dark);
      color: #fff;
      box-shadow: 0 3px 18px rgba(54,180,250,0.18);
    }
    .dark-mode .toggle-btn {
      background: linear-gradient(90deg,#1f2c36 0%, #233243 100%);
      color: #93e7ff; border: 1.2px solid #223b51;
    }
    main {
      max-width: 1120px; margin: 2.7rem auto 0 auto; padding: 0 1.7rem 2rem 1.7rem;
    }
    section {
      background: var(--card-bg);
      border-radius: var(--border-radius);
      box-shadow: var(--shadow);
      margin-bottom: 2.5rem;
      padding: 2.25rem 2.15rem 1.6rem 2.15rem;
      transition: var(--transition);
      position: relative;
      overflow: hidden;
      animation: fadeInUp 0.7s cubic-bezier(.33,1.15,.68,1) both;
    }
    @keyframes fadeInUp {
      from { opacity:0; transform: translateY(20px);}
      to {opacity:1; transform: translateY(0);}
    }
    .dark-mode section { background: var(--card-bg-dark); }
    h2 {
      margin-top: 0; font-size: 1.62rem; font-weight: 700;
      display: flex; align-items: center; gap: 0.6em; margin-bottom: 1.1rem;
      line-height: 1.22;
    }
    .icon {
      font-size: 1.35em;
      margin-right: 0.11em;
      vertical-align: -0.12em;
    }
    .gradient-title {
      background: linear-gradient(90deg, #22b7d5 0, #33e4be 70%);
      background-clip: text;
      -webkit-background-clip: text;
      color: transparent;
      -webkit-text-fill-color: transparent;
      display: inline;
      font-size: inherit;
      font-weight: inherit;
      letter-spacing: inherit;
    }
    pre, code {
      font-family: "JetBrains Mono", "Fira Mono", "Menlo", monospace;
      background: var(--metric-bg);
      color: #0090ff;
      padding: 1em 1em;
      font-size: 1.06em;
      border-radius: 0.8em;
      box-shadow: 0 2px 8px rgba(42,126,230,0.03);
      margin: 0.1em 0 0.9em 0;
      overflow-x: auto;
      transition: var(--transition);
      line-height: 1.62;
      border: 1.2px solid var(--divider);
    }
    .dark-mode pre, .dark-mode code {
      background: var(--metric-bg-dark);
      color: #31e5ff;
      border: 1.2px solid var(--divider-dark);
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1.22em;
      margin-bottom: 1em;
    }
    .kpi-card {
      background: linear-gradient(125deg, #eafafd 0%, #f7fcff 100%);
      border-radius: 1.5em;
      box-shadow: 0 4px 22px rgba(10,190,240,0.07);
      padding: 1.09em 1.45em;
      text-align: center;
      font-size: 1.13em;
      color: #0e395a;
      font-weight: 600;
      transition: var(--transition);
      letter-spacing: 0.01em;
      border: 1.2px solid #d8f6ff;
      display: flex;
      flex-direction: column;
      gap: 0.22em;
      align-items: center;
      justify-content: center;
    }
    .dark-mode .kpi-card {
      background: linear-gradient(125deg, #222c38 0%, #262e3a 100%);
      color: #93eafd;
      border: 1.2px solid #233b4d;
    }
    .kpi-label {
      font-size: 0.98em;
      color: #5e7893;
      margin-bottom: 0.18em;
      font-weight: 500;
    }
    .dark-mode .kpi-label { color: #8fd1e3; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(305px, 1fr));
      gap: 1.55rem;
      margin-bottom: 1.18em;
    }
    img {
      width: 100%;
      border-radius: 1.2em;
      box-shadow: 0 3px 18px 0 rgba(53,138,215,0.08);
      border: 2.5px solid #e0f6ff;
      transition: transform 0.15s cubic-bezier(.75,0,.18,1), box-shadow 0.15s;
      cursor: pointer;
      background: #f6fafd;
    }
    img:hover {
      transform: scale(1.045) translateY(-2px);
      box-shadow: 0 8px 38px 0 rgba(36,196,245,0.12);
      border-color: #38e1ff;
      z-index: 2;
    }
    .dark-mode img {
      background: #212837;
      border-color: #253b51;
    }
    .divider {
      height: 1.4px;
      background: linear-gradient(90deg,#c6f6ff 0,#e5eefc 80%);
      border-radius: 2em;
      margin: 2.2em 0 2em 0;
      border: none;
    }
    .dark-mode .divider {
      background: linear-gradient(90deg,#2c3a47 0,#2b414e 80%);
    }
    .download-group { margin-top: 0.45em; }
    .download-btn { margin-right: 0.72em; }
    .no-data {
      color: #bbb;
      text-align: center;
      font-size: 1.09em;
      padding: 1.18em 0;
    }
    .batch-zip {
      margin: 0.8em 0 2em 0;
      text-align: right;
    }
    .summary-box {
      border-radius: 18px;
      box-shadow: 0 4px 18px #d6f3fc7a;
      background: linear-gradient(109deg,#d7f6ff 0,#f3fcff 100%);
      padding: 1.38em 2.18em;
      margin-bottom: 1.7em;
      font-size: 1.13em;
      font-weight: 500;
      color: #18849b;
      border: 1.5px solid #caf4ff;
    }
    .dark-mode .summary-box {
      background: linear-gradient(109deg,#223947 0,#1a3043 100%);
      color: #a3dfff;
      border: 1.5px solid #1d3a4a;
    }
    table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      margin-bottom: 1em;
      font-size: 1.04em;
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
      box-shadow: 0 2px 12px #e3f6ff65;
    }
    th, td {
      padding: 0.85em 0.95em;
      text-align: center;
    }
    thead tr {
      background: var(--table-head);
      font-weight: 700;
      font-size: 1em;
      border-bottom: 2.2px solid var(--table-border);
    }
    tbody tr:nth-child(even) { background: #f5fafc; }
    tbody tr:hover { background: #e7f7fb; }
    td, th { border-bottom: 1.1px solid var(--table-border);}
    table:last-child {margin-bottom:0.5em;}
    .dark-mode table {
      background: #232e3d;
      box-shadow: 0 2px 10px #1e2e3a80;
    }
    .dark-mode th, .dark-mode thead tr {
      background: var(--table-head-dark);
      color: #7fd2ea;
      border-bottom: 2px solid var(--table-border-dark);
    }
    .dark-mode td, .dark-mode tbody tr {
      border-bottom: 1.1px solid var(--table-border-dark);
    }
    .dark-mode tbody tr:nth-child(even) { background: #223146;}
    .dark-mode tbody tr:hover { background: #23303f;}
    footer {
      background: var(--footer-bg);
      color: #55b3e2;
      text-align: center;
      padding: 1.1rem 0 1.2rem 0;
      font-size: 1.1rem;
      letter-spacing: 0.01em;
      border-top-left-radius: 14px;
      border-top-right-radius: 14px;
      margin-top: 1.5rem;
      box-shadow: 0 0 10px #c3ecfa27;
      font-weight: 600;
    }
    .dark-mode footer {
      background: var(--footer-bg-dark);
      color: #38e1ff;
      box-shadow: 0 0 10px #10304545;
    }
    @media (max-width: 700px) {
      header {padding: 1rem;}
      section {padding: 1rem;}
      .summary-box {padding: 1em;}
      .kpi-card {font-size: 1em;}
      h2 {font-size: 1.12rem;}
      main {padding: 0 0.2rem;}
      table {font-size:0.99em;}
    }
  </style>
</head>
<body>
  <header>
    <h1>🔬 <span style="letter-spacing:0.06em">CELLFLOW REPORT</span></h1>
    <button class="toggle-btn" id="theme-toggle" aria-label="Toggle dark mode">🌙 Toggle Theme</button>
  </header>
  <main>
    <!-- Summary Section -->
    <div class="summary-box" role="region" aria-live="polite" aria-atomic="true">
      <span style="font-size:1.21em; font-weight:700;">Summary:</span>
      {summary_text}
    </div>
    <!-- Download All Outputs -->
    <div class="batch-zip" role="region">
      <a href="{all_outputs_zip}" download class="toggle-btn">⬇️ Download All Outputs (.zip)</a>
    </div>
    <!-- Run Configuration Section -->
    <section aria-labelledby="run-config-title">
      <h2 id="run-config-title"><span class="icon">📋</span><span class="gradient-title">Run Configuration</span></h2>
      <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">Model ID</div>{model_id}</div>
        <div class="kpi-card"><div class="kpi-label">Crop Size</div>{crop_size}</div>
        <div class="kpi-card"><div class="kpi-label">Epochs</div>{epochs}</div>
        <div class="kpi-card"><div class="kpi-label">Pixel Res</div>{pixel_res}</div>
        <div class="kpi-card"><div class="kpi-label">Backbone</div>{backbone}</div>
        <div class="kpi-card"><div class="kpi-label">Mask File</div><span style="font-size:0.98em">{mask_file}</span></div>
        <div class="kpi-card"><div class="kpi-label">Config File</div><span style="font-size:0.98em">{config_file}</span></div>
      </div>
    </section>
    <!-- Training Metrics Section -->
    <section aria-labelledby="training-metrics-title">
      <h2 id="training-metrics-title"><span class="icon">📈</span><span class="gradient-title">Training Metrics</span></h2>
      <div class="grid">
        <figure>
          <img src="{epoch_curve}" alt="Epoch Training Curve" title="Epoch Training Curve" />
          <figcaption style="margin-top:0.7em;">
            Epoch Curve
            <a href="{epoch_curve}" download class="toggle-btn download-btn">⬇️ PNG</a>
            <a href="{epoch_curve_pdf}" download class="toggle-btn">⬇️ PDF</a>
          </figcaption>
        </figure>
        <figure>
          <img src="{loss_curve}" alt="Loss Over Time" title="Loss Over Time" />
          <figcaption style="margin-top:0.7em;">
            Loss Curve
            <a href="{loss_curve}" download class="toggle-btn download-btn">⬇️ PNG</a>
            <a href="{loss_curve_pdf}" download class="toggle-btn">⬇️ PDF</a>
          </figcaption>
        </figure>
        <figure>
          <img src="{accuracy_curve}" alt="Accuracy Over Time" title="Accuracy Over Time" />
          <figcaption style="margin-top:0.7em;">
            Accuracy Curve
            <a href="{accuracy_curve}" download class="toggle-btn download-btn">⬇️ PNG</a>
            <a href="{accuracy_curve_pdf}" download class="toggle-btn">⬇️ PDF</a>
          </figcaption>
        </figure>
      </div>
      <div class="download-group" style="text-align:right;">
        <a href="{training_metrics_csv}" download class="toggle-btn">⬇️ Training Metrics CSV</a>
      </div>
    </section>
    <!-- Sample Predictions Section -->
    <section aria-labelledby="sample-predictions-title">
      <h2 id="sample-predictions-title"><span class="icon">🖼️</span><span class="gradient-title">Sample Predictions</span></h2>
      <div class="grid" id="sample-predictions" role="list">
        {sample_predictions_html}
      </div>
    </section>
    <!-- Confusion Matrix Section -->
    <section id="confusion-section" aria-labelledby="confusion-matrix-title">
      <h2 id="confusion-matrix-title"><span class="icon">📊</span><span class="gradient-title">Confusion Matrix</span></h2>
      <div style="max-width:400px;margin:auto;">
        <img src="{confusion_matrix}" alt="Confusion Matrix" title="Confusion Matrix" />
        <div class="download-group" style="text-align:center;">
          <a href="{confusion_matrix}" download class="toggle-btn download-btn">⬇️ PNG</a>
          <a href="{confusion_matrix_pdf}" download class="toggle-btn">⬇️ PDF</a>
          <a href="{confusion_matrix_csv}" download class="toggle-btn">⬇️ CSV</a>
        </div>
      </div>
    </section>
    <!-- Classifier Performance Section -->
    <section id="classifier-section" aria-labelledby="classifier-performance-title">
      <h2 id="classifier-performance-title"><span class="icon">🧠</span><span class="gradient-title">Classifier Performance</span></h2>
      <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">Accuracy</div>{clf_accuracy}</div>
        <div class="kpi-card"><div class="kpi-label">Precision</div>{clf_precision}</div>
        <div class="kpi-card"><div class="kpi-label">Recall</div>{clf_recall}</div>
        <div class="kpi-card"><div class="kpi-label">F1 Score</div>{clf_f1}</div>
        <div class="kpi-card"><div class="kpi-label">AUC</div>{clf_auc}</div>
      </div>
      <div class="download-group" style="text-align:right;">
        <a href="{classifier_stats_csv}" download class="toggle-btn">⬇️ Classifier Stats CSV</a>
      </div>
    </section>
    <!-- Per-Class Statistics Section -->
    <section id="perclass-section" aria-labelledby="per-class-stats-title">
      <h2 id="per-class-stats-title"><span class="icon">🔢</span><span class="gradient-title">Per-Class Statistics</span></h2>
      <table>
        <thead>
          <tr><th>Class</th><th>Count</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
        </thead>
        <tbody>
          {perclass_table_rows}
        </tbody>
      </table>
      <div class="download-group" style="text-align:right;">
        <a href="{perclass_csv}" download class="toggle-btn">⬇️ Per-Class Stats CSV</a>
      </div>
    </section>
    <!-- TAP Batch Metrics Section -->
    <section id="tap-section" aria-labelledby="tap-metrics-title">
      <h2 id="tap-metrics-title"><span class="icon">🧪</span><span class="gradient-title">TAP (Batch Mode) Metrics</span></h2>
      <table>
        <thead>
          <tr><th>File</th><th>TP</th><th>FP</th><th>FN</th><th>TAP Score</th></tr>
        </thead>
        <tbody>
          {tap_table_rows}
        </tbody>
      </table>
      <div class="download-group" style="text-align:right;">
        <a href="{tap_csv}" download class="toggle-btn">⬇️ TAP CSV</a>
      </div>
      <div class="grid" id="tap-overlays" role="list">
        {tap_overlays_html}
      </div>
    </section>
    <!-- Additional Outputs Section -->
    <section id="extra-section" aria-labelledby="additional-outputs-title">
      <h2 id="additional-outputs-title"><span class="icon">📁</span><span class="gradient-title">Additional Outputs</span></h2>
      <div>
        {extra_content}
      </div>
    </section>
  </main>
  <footer>
    <strong>CELLFLOW</strong> &mdash; &copy; 2025
  </footer>
  <script>
    // Theme toggle with transition and persistence
    (function() {
      const themeBtn = document.getElementById('theme-toggle');
      if (!themeBtn) return;
      
      function updateButtonText() {
        const isDark = document.body.classList.contains('dark-mode');
        themeBtn.textContent = isDark ? '☀️ Toggle Theme' : '🌙 Toggle Theme';
      }

      themeBtn.addEventListener('click', function () {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        localStorage.setItem('cf_dark_mode', isDark ? '1' : '0');
        updateButtonText();
      });

      const savedMode = localStorage.getItem('cf_dark_mode');
      if (savedMode === '1') {
        document.body.classList.add('dark-mode');
      }
      updateButtonText();
    })();

    // ------- Mode-dependent display --------
    var pipelineMode = {pipeline_mode}; // 0 = classifier, 1 = TAP batch mode
    function hideSection(id) {
      var el = document.getElementById(id);
      if(el) el.style.display = "none";
    }
    function showSection(id) {
      var el = document.getElementById(id);
      if(el) el.style.display = "";
    }
    window.onload = function() {
      if (pipelineMode === 0) {
        hideSection("tap-section");
        hideSection("extra-section");
      } else if (pipelineMode === 1) {
        hideSection("classifier-section");
        hideSection("confusion-section");
        hideSection("perclass-section");
      }
      // Hide empty dynamic grids
      const samplePredictions = document.getElementById('sample-predictions');
      if(samplePredictions && samplePredictions.children.length === 0)
        samplePredictions.innerHTML = '<div class="no-data">No sample predictions available.</div>';

      const tapOverlays = document.getElementById('tap-overlays');
      if(tapOverlays && tapOverlays.children.length === 0)
        tapOverlays.innerHTML = '<div class="no-data">No TAP overlays available.</div>';
    };
  </script>
</body>
</html>
"""

html_out = HTML_TEMPLATE.format(
    summary_text=summary_text,
    all_outputs_zip=os.path.relpath(all_outputs_zip, outdir) if all_outputs_zip else "#",
    model_id=html_escape(model_id),
    crop_size=html_escape(crop_size),
    epochs=html_escape(epochs),
    pixel_res=html_escape(pixel_res),
    backbone=html_escape(backbone),
    mask_file=html_escape(mask_file),
    config_file=html_escape(config_file),
    epoch_curve=os.path.relpath(epoch_curve, outdir) if epoch_curve else "",
    epoch_curve_pdf=os.path.relpath(epoch_curve_pdf, outdir) if epoch_curve_pdf else "",
    loss_curve=os.path.relpath(loss_curve, outdir) if loss_curve else "",
    loss_curve_pdf=os.path.relpath(loss_curve_pdf, outdir) if loss_curve_pdf else "",
    accuracy_curve=os.path.relpath(accuracy_curve, outdir) if accuracy_curve else "",
    accuracy_curve_pdf=os.path.relpath(accuracy_curve_pdf, outdir) if accuracy_curve_pdf else "",
    training_metrics_csv=os.path.relpath(training_metrics_csv, outdir) if training_metrics_csv else "",
    sample_predictions_html=sample_predictions_html,
    confusion_matrix=os.path.relpath(confusion_matrix_png, outdir) if confusion_matrix_png else "",
    confusion_matrix_pdf=os.path.relpath(confusion_matrix_pdf, outdir) if confusion_matrix_pdf else "",
    confusion_matrix_csv=os.path.relpath(confusion_matrix_csv, outdir) if confusion_matrix_csv else "",
    clf_accuracy=clf_accuracy,
    clf_precision=clf_precision,
    clf_recall=clf_recall,
    clf_f1=clf_f1,
    clf_auc=clf_auc,
    classifier_stats_csv=os.path.relpath(classifier_stats_csv, outdir) if classifier_stats_csv else "",
    perclass_table_rows=perclass_rows,
    perclass_csv=os.path.relpath(perclass_csv, outdir) if perclass_csv else "",
    tap_table_rows=tap_rows,
    tap_csv=os.path.relpath(tap_csv, outdir) if tap_csv else "",
    tap_overlays_html=tap_overlays_html,
    extra_content=extra_content,
    pipeline_mode=pipeline_mode,
)

# ────────────────────────────────────────────────────────────────
#   Write HTML output to report file
# ────────────────────────────────────────────────────────────────

report_path = os.path.join(outdir, "report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_out)

print(f"Report saved to {report_path}")
