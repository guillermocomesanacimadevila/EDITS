#!/usr/bin/env python3
import os
import sys
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def safe_read_yaml(yaml_path):
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f) or {}

def safe_read_json(json_path):
    if not json_path or not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f) or {}

def find_first(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""

def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def crawl_outputs(outdir):
    outputs = []
    for root, _, files in os.walk(outdir):
        rel_root = os.path.relpath(root, outdir)
        for f in sorted(files):
            if f.lower() == "report.html":
                continue
            rel_path = os.path.join(rel_root, f) if rel_root != "." else f
            outputs.append(rel_path)
    return sorted(outputs)

def group_files_by_type(files):
    groups = {
        "Config & Requirements": [],
        "Logs": [],
        "Results / Statistics": [],
        "Visualizations": [],
        "PDFs": [],
        "Model Weights": [],
        "Other": [],
    }
    for f in files:
        fl = f.lower()
        ext = os.path.splitext(f)[1].lower()
        if ext in {".yml", ".yaml"} or any(x in fl for x in ["requirement", "conda", "env", "config"]):
            groups["Config & Requirements"].append(f)
        elif ext in {".log", ".txt"} and ("log" in fl or "stdout" in fl or "stderr" in fl):
            groups["Logs"].append(f)
        elif ext in {".csv", ".tsv", ".json"} or any(x in fl for x in ["result", "metric", "stats"]):
            groups["Results / Statistics"].append(f)
        elif ext in {".png", ".jpg", ".jpeg", ".svg", ".tif", ".tiff"}:
            groups["Visualizations"].append(f)
        elif ext == ".pdf":
            groups["PDFs"].append(f)
        elif ext in {".pth", ".h5", ".ckpt", ".npz"} or any(x in fl for x in ["weight", "model"]):
            groups["Model Weights"].append(f)
        else:
            groups["Other"].append(f)
    return groups

def preview_text_file(full_path, max_lines=300):
    try:
        with open(full_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        preview = "".join(lines[:max_lines])
        if len(lines) > max_lines:
            preview += "\n... (truncated, see file for full content)"
        return f"<pre>{html_escape(preview)}</pre>"
    except Exception as e:
        return f"<div class='muted'>Error reading file: {html_escape(e)}</div>"

def detect_metrics_json(outdir):
    return find_first(
        os.path.join(outdir, "metrics.json"),
        os.path.join(outdir, "classifier_metrics.json"),
        os.path.join(outdir, "run_metrics.json"),
    )

def human_bytes(n):
    try:
        n = float(n)
    except Exception:
        return str(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PB"

def gather_run_metrics(outdir):
    candidates = [
        detect_metrics_json(outdir),
        os.path.join(outdir, "timing.json"),
        os.path.join(outdir, "runtime.json"),
        os.path.join(outdir, "system.json"),
        os.path.join(outdir, "logs", "pipeline_metrics.csv"),
        os.path.join(outdir, "logs", "01_finetune_metrics.csv"),
    ]
    d = {}
    for c in candidates:
        if not c:
            continue
        if c.lower().endswith(".json"):
            d.update(safe_read_json(c))
        elif c.lower().endswith(".csv"):
            try:
                df = pd.read_csv(c)
                for _, row in df.iterrows():
                    k = row.get("step_name") or row.get("Metric") or None
                    v = row.get("elapsed_sec") if "elapsed_sec" in row else row.get("Value", None)
                    if k is not None and v is not None:
                        d[str(k)] = v
            except Exception:
                pass
    if "results_size_bytes" in d and "Results size" not in d:
        d["Results size"] = human_bytes(d["results_size_bytes"])
    printable = {}
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            try:
                printable[k] = json.dumps(v)
            except Exception:
                printable[k] = str(v)
        else:
            printable[k] = v
    return printable

def read_csv_safe(path, max_rows=100000):
    try:
        if path.lower().endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", nrows=max_rows)
        else:
            df = pd.read_csv(path, nrows=max_rows)
        return df
    except Exception:
        return None

def ensure_png_preview(outdir, rel_path):
    try:
        from imageio.v2 import imread, imwrite
    except Exception:
        return ""
    src = os.path.join(outdir, rel_path)
    if not os.path.exists(src):
        return ""
    rel_png = os.path.splitext(rel_path)[0] + ".png"
    prev_path = os.path.join(outdir, "_previews", rel_png)
    os.makedirs(os.path.dirname(prev_path), exist_ok=True)
    if os.path.exists(prev_path) and os.path.getmtime(prev_path) >= os.path.getmtime(src):
        return os.path.relpath(prev_path, outdir)
    try:
        img = imread(src)
        arr = np.asarray(img)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (2, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        arr = arr.astype(np.float32)
        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
        if mx > mn:
            arr = (255 * (arr - mn) / (mx - mn)).clip(0, 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        imwrite(prev_path, arr)
        return os.path.relpath(prev_path, outdir)
    except Exception:
        return ""

def _read_csv(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None

def _parse_confusion_matrix(cm_csv_path):
    df = _read_csv(cm_csv_path)
    if df is None or df.empty:
        return {}, None
    if df.columns[0].lower().startswith("unnamed") or "actual" in df.columns[0].lower():
        df = df.drop(columns=[df.columns[0]])
    cols = [c for c in df.columns if "predicted" in c.lower()]
    if len(cols) != 2:
        cols = df.columns[:2]
    df = df[cols].copy()
    if df.shape != (2, 2):
        return {}, df
    try:
        tn = int(round(float(df.iloc[0, 0])))
        fp = int(round(float(df.iloc[0, 1])))
        fn = int(round(float(df.iloc[1, 0])))
        tp = int(round(float(df.iloc[1, 1])))
        return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}, df
    except Exception:
        return {}, df

def _cm_metrics(cm):
    if not cm:
        return {}
    TN, FP, FN, TP = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
    total = TN + FP + FN + TP
    acc = (TP + TN) / total if total else 0.0
    prec1 = TP / (TP + FP) if (TP + FP) else 0.0
    rec1 = TP / (TP + FN) if (TP + FN) else 0.0
    f1_1 = (2 * prec1 * rec1 / (prec1 + rec1)) if (prec1 + rec1) else 0.0
    prec0 = TN / (TN + FN) if (TN + FN) else 0.0
    rec0 = TN / (TN + FP) if (TN + FP) else 0.0
    f1_0 = (2 * prec0 * rec0 / (prec0 + rec0)) if (prec0 + rec0) else 0.0
    supp0, supp1 = (TN + FP), (TP + FN)
    macro_p = (prec0 + prec1) / 2
    macro_r = (rec0 + rec1) / 2
    macro_f = (f1_0 + f1_1) / 2
    if (supp0 + supp1) > 0:
        w_p = (prec0 * supp0 + prec1 * supp1) / (supp0 + supp1)
        w_r = (rec0 * supp0 + rec1 * supp1) / (supp0 + supp1)
        w_f = (f1_0 * supp0 + f1_1 * supp1) / (supp0 + supp1)
    else:
        w_p = w_r = w_f = 0.0
    return {
        "accuracy": acc,
        "class0_precision": prec0,
        "class0_recall": rec0,
        "class0_f1": f1_0,
        "class0_support": supp0,
        "class1_precision": prec1,
        "class1_recall": rec1,
        "class1_f1": f1_1,
        "class1_support": supp1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "weighted_precision": w_p,
        "weighted_recall": w_r,
        "weighted_f1": w_f,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "total": total,
    }

def _fmt_float(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

parser = argparse.ArgumentParser(description="EDITS: Generate HTML report for a run")
parser.add_argument("--config", required=True)
parser.add_argument("--outdir", required=True)
parser.add_argument("--batch_outdirs", nargs="*", default=[])
args = parser.parse_args()

config = safe_read_yaml(args.config)
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

metrics_path = detect_metrics_json(outdir)
metrics = safe_read_json(metrics_path) if metrics_path else {}

model_id = config.get("name", "-")
crop_size = config.get("size", "-")
epochs = config.get("epochs", "-")
pixel_res = config.get("pixel_resolution", "-")
backbone = config.get("backbone", "-")

run_root = outdir

candidate_sup_dirs = [
    os.path.join(run_root, "figures", "supervised"),
]
if model_id and model_id != "-":
    candidate_sup_dirs += [
        os.path.join("results", "supervised_classification", "figures", model_id),
        os.path.join(os.path.dirname(run_root), "supervised_classification", "figures", model_id),
    ]
sup_dir = ""
for p in candidate_sup_dirs:
    if p and os.path.isdir(p):
        sup_dir = p
        break

candidate_metrics_dirs = [
    os.path.join(run_root, "metrics"),
]
if sup_dir:
    candidate_metrics_dirs.append(sup_dir)
metrics_dir = ""
for p in candidate_metrics_dirs:
    if p and os.path.isdir(p):
        metrics_dir = p
        break

def choose_path(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return ""

cm_last_csv = choose_path(
    os.path.join(metrics_dir, "confusion_matrix_last_run.csv") if metrics_dir else "",
    os.path.join(sup_dir, "confusion_matrix_last_run.csv") if sup_dir else "",
)
cm_agg_csv = choose_path(
    os.path.join(metrics_dir, "confusion_matrix_aggregate.csv") if metrics_dir else "",
    os.path.join(sup_dir, "confusion_matrix_aggregate.csv") if sup_dir else "",
)
per_run_csv = choose_path(
    os.path.join(metrics_dir, "per_run_metrics.csv") if metrics_dir else "",
    os.path.join(sup_dir, "per_run_metrics.csv") if sup_dir else "",
)

cb_train_before = choose_path(
    os.path.join(metrics_dir, "class_balance_train_before_balancing.csv") if metrics_dir else "",
    os.path.join(metrics_dir, "class_balance", "class_balance_train_before_balancing.csv") if metrics_dir else "",
    os.path.join(sup_dir, "class_balance_train_before_balancing.csv") if sup_dir else "",
)
cb_train_after = choose_path(
    os.path.join(metrics_dir, "class_balance_train_after_balancing.csv") if metrics_dir else "",
    os.path.join(metrics_dir, "class_balance", "class_balance_train_after_balancing.csv") if metrics_dir else "",
    os.path.join(sup_dir, "class_balance_train_after_balancing.csv") if sup_dir else "",
)
cb_val = choose_path(
    os.path.join(metrics_dir, "class_balance_validation.csv") if metrics_dir else "",
    os.path.join(metrics_dir, "class_balance", "class_balance_validation.csv") if metrics_dir else "",
    os.path.join(sup_dir, "class_balance_validation.csv") if sup_dir else "",
)
cb_test = choose_path(
    os.path.join(metrics_dir, "class_balance_test.csv") if metrics_dir else "",
    os.path.join(metrics_dir, "class_balance", "class_balance_test.csv") if metrics_dir else "",
    os.path.join(sup_dir, "class_balance_test.csv") if sup_dir else "",
)

cm_last, cm_last_df = _parse_confusion_matrix(cm_last_csv) if cm_last_csv else ({}, None)
cm_agg, cm_agg_df = _parse_confusion_matrix(cm_agg_csv) if cm_agg_csv else ({}, None)
cm_last_metrics = _cm_metrics(cm_last) if cm_last else {}
cm_agg_metrics = _cm_metrics(cm_agg) if cm_agg else {}

mask_file = "-"
if config.get("input_mask"):
    mask0 = config["input_mask"]
    if isinstance(mask0, (list, tuple)) and mask0:
        mask_file = os.path.basename(mask0[0])
    elif isinstance(mask0, str):
        mask_file = os.path.basename(mask0)

config_file = os.path.basename(args.config)

all_files = crawl_outputs(outdir)
file_groups = group_files_by_type(all_files)

def fmt_metric(m, key, nd=3):
    val = m.get(key, None)
    if val is None:
        return "-"
    try:
        if isinstance(val, (int, float, np.floating)):
            return f"{float(val):.{nd}f}"
        return str(val)
    except Exception:
        return str(val)

clf_accuracy = fmt_metric(metrics, "accuracy")
clf_precision = fmt_metric(metrics, "precision")
clf_recall = fmt_metric(metrics, "recall")
clf_f1 = fmt_metric(metrics, "f1")
clf_auc = fmt_metric(metrics, "auc")

if clf_accuracy == "-" and cm_last_metrics:
    clf_accuracy = _fmt_float(cm_last_metrics.get("accuracy"), 3)
    clf_precision = _fmt_float(cm_last_metrics.get("macro_precision"), 3)
    clf_recall = _fmt_float(cm_last_metrics.get("macro_recall"), 3)
    clf_f1 = _fmt_float(cm_last_metrics.get("macro_f1"), 3)

summary_text = (
    f"Run completed for model <b>{html_escape(model_id)}</b>. "
    f"Backbone: <b>{html_escape(backbone)}</b>, Crop: <b>{html_escape(crop_size)}</b>, "
    f"Epochs: <b>{html_escape(epochs)}</b>, Pixel Res: <b>{html_escape(pixel_res)}</b>."
)
if clf_accuracy != "-":
    summary_text += f" Final accuracy: <b>{clf_accuracy}</b>."

all_outputs_zip = find_first(os.path.join(outdir, "all_outputs.zip"))
zip_button_html = (
    f'<a href="{html_escape(os.path.relpath(all_outputs_zip, outdir))}" download class="pill">⬇️ Download All Outputs (.zip)</a>'
    if all_outputs_zip
    else '<span class="pill" style="opacity:.5;pointer-events:none">⬇️ Download All Outputs (.zip)</span>'
)

tables_html = []
table_init_js = []
table_idx = 0

def add_table_from_df(title, df, table_id):
    thead = "<thead><tr>" + "".join(f"<th>{html_escape(c)}</th>" for c in df.columns) + "</tr></thead>"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            cells.append(f"<td>{html_escape('' if pd.isna(v) else v)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    tbody = "<tbody>" + "".join(rows) + "</tbody>"
    html = f"""
      <div class="section" style="margin-top:12px">
        <div class="section-header">
          <h2>📊 {html_escape(title)}</h2>
          <span class="pill">interactive • sortable • export</span>
        </div>
        <div class="table-wrap">
          <table id="{table_id}" class="display nowrap compact" style="width:100%">
            {thead}{tbody}
          </table>
        </div>
      </div>
    """
    return html

for rel in file_groups.get("Results / Statistics", []):
    full = os.path.join(outdir, rel)
    ext = os.path.splitext(rel)[1].lower()
    if ext in [".csv", ".tsv"]:
        df = read_csv_safe(full, max_rows=100000)
        if df is not None and df.shape[0] > 0:
            tid = f"tbl_{table_idx}"
            tables_html.append(add_table_from_df(rel, df, tid))
            table_init_js.append(
                f"""$('#{tid}').DataTable({{
                dom:'Bfrtip',
                buttons:[
                  {{extend:'csv',text:'<i class="fa-solid fa-download"></i> CSV',className:'btn-csv'}},
                  {{extend:'colvis',text:'<i class="fa-solid fa-table-cells"></i> Columns'}}
                ],
                deferRender:true, autoWidth:false, scrollX:true, pageLength:25,
                lengthMenu:[[10,25,50,100,-1],[10,25,50,100,'All']]
            }});"""
            )
            table_idx += 1
        else:
            tables_html.append(
                f"""
            <div class="section" style="margin-top:12px">
              <div class="section-header"><h2>📄 {html_escape(rel)}</h2></div>
              <p><a class="pill" href="{html_escape(rel)}" download>{html_escape(os.path.basename(rel))}</a></p>
              <div class="muted">Could not preview table or file is empty.</div>
            </div>
            """
            )
    elif ext == ".json":
        data = safe_read_json(full)
        if isinstance(data, dict) and data:
            kv = "".join(
                f"<tr><td>{html_escape(k)}</td><td class='num'>{html_escape(v)}</td></tr>"
                for k, v in data.items()
            )
            tid = f"tbl_{table_idx}"
            tables_html.append(
                f"""
            <div class="section" style="margin-top:12px">
              <div class="section-header"><h2>🧾 {html_escape(rel)}</h2></div>
              <div class="table-wrap">
                <table id="{tid}" class="display nowrap compact" style="width:100%">
                  <thead><tr><th>Key</th><th>Value</th></tr></thead>
                  <tbody>{kv}</tbody>
                </table>
              </div>
            </div>
            """
            )
            table_init_js.append(
                f"""$('#{tid}').DataTable({{
                autoWidth:false, scrollX:true, paging:false, searching:false, info:false
            }});"""
            )
            table_idx += 1
        else:
            tables_html.append(
                f"""
            <div class="section" style="margin-top:12px">
              <div class="section-header"><h2>🧾 {html_escape(rel)}</h2></div>
              <pre>{html_escape(json.dumps(data, indent=2))}</pre>
            </div>
            """
            )

TABLES_BLOCK = (
    "\n".join(tables_html)
    if tables_html
    else """
<div class="section">
  <div class="section-header"><h2>📊 Tables</h2></div>
  <p class="muted">No CSV/TSV/JSON tables found in Results / Statistics.</p>
</div>
"""
)

CLASSIF_SECTIONS = []

def _render_df_as_table(df, tbl_id, title, init_buttons=True):
    if df is None or df.empty:
        return
    thead = "<thead><tr>" + "".join(f"<th>{html_escape(c)}</th>" for c in df.columns) + "</tr></thead>"
    rows = []
    for _, row in df.iterrows():
        tds = "".join(f"<td>{html_escape(row[c])}</td>" for c in df.columns)
        rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>" + "".join(rows) + "</tbody>"
    CLASSIF_SECTIONS.append(
        f"""
      <div class="section">
        <div class="section-header"><h2>🧮 {html_escape(title)}</h2></div>
        <div class="table-wrap">
          <table id="{tbl_id}" class="display nowrap compact" style="width:100%">{thead}{tbody}</table>
        </div>
      </div>
    """
    )
    if init_buttons:
        table_init_js.append(
            f"""$('#{tbl_id}').DataTable({{
            dom:'Bfrtip',
            buttons:[{{extend:'csv',text:'CSV'}},{{extend:'colvis',text:'Columns'}}],
            deferRender:true, autoWidth:false, scrollX:true, pageLength:25
        }});"""
        )
    else:
        table_init_js.append(
            f"""$('#{tbl_id}').DataTable({{
            autoWidth:false, scrollX:true, paging:false, searching:false, info:false
        }});"""
        )

if cm_last_metrics:
    kpi_rows = [
        ("Accuracy", _fmt_float(cm_last_metrics["accuracy"], 4)),
        ("Macro Precision", _fmt_float(cm_last_metrics["macro_precision"], 4)),
        ("Macro Recall", _fmt_float(cm_last_metrics["macro_recall"], 4)),
        ("Macro F1", _fmt_float(cm_last_metrics["macro_f1"], 4)),
        ("Weighted Precision", _fmt_float(cm_last_metrics["weighted_precision"], 4)),
        ("Weighted Recall", _fmt_float(cm_last_metrics["weighted_recall"], 4)),
        ("Weighted F1", _fmt_float(cm_last_metrics["weighted_f1"], 4)),
        (
            "TN / FP / FN / TP",
            f'{cm_last_metrics["TN"]} / {cm_last_metrics["FP"]} / {cm_last_metrics["FN"]} / {cm_last_metrics["TP"]}',
        ),
        (
            "Support (class 0 / 1)",
            f'{cm_last_metrics["class0_support"]} / {cm_last_metrics["class1_support"]}',
        ),
    ]
    df_kpi = pd.DataFrame(kpi_rows, columns=["Metric", "Value"])
    _render_df_as_table(df_kpi, "clf_kpis", "Classification summary (last run)", init_buttons=False)

if per_run_csv:
    df_pr = _read_csv(per_run_csv)
    if df_pr is not None and not df_pr.empty:
        _render_df_as_table(
            df_pr,
            "clf_per_run",
            f"Per-run training metrics — {os.path.relpath(per_run_csv, outdir) if os.path.isabs(per_run_csv) or per_run_csv.startswith('..') else per_run_csv}",
        )

if cm_last_df is not None and not cm_last_df.empty:
    _render_df_as_table(cm_last_df, "clf_cm_last", "Confusion Matrix — last run", init_buttons=False)
if cm_agg_df is not None and not cm_agg_df.empty:
    _render_df_as_table(cm_agg_df, "clf_cm_agg", "Confusion Matrix — aggregate", init_buttons=False)

for title, path, tid in [
    ("Class balance — train (before balancing)", cb_train_before, "cb_train_before"),
    ("Class balance — train (after balancing)", cb_train_after, "cb_train_after"),
    ("Class balance — validation", cb_val, "cb_val"),
    ("Class balance — test", cb_test, "cb_test"),
]:
    if path:
        df_cb = _read_csv(path)
        if df_cb is not None and not df_cb.empty:
            _render_df_as_table(df_cb, tid, title, init_buttons=False)

CLASSIFICATION_BLOCK = (
    "\n".join(CLASSIF_SECTIONS)
    if CLASSIF_SECTIONS
    else """
<div class="section">
  <div class="section-header"><h2>✅ Classification</h2></div>
  <p class="muted">No supervised-classification artefacts were found for this run.</p>
</div>
"""
)

def section_for(relpath):
    p = relpath.lower()
    if "model_artifacts" in p or "finetune" in p or "pretraining" in p:
        return "Fine-tune"
    if "/phase2_data_prep" in p or "data_prep" in p or "dataprep" in p:
        return "Data Prep"
    if "figures/supervised" in p or "supervised_classification" in p:
        return "Supervised Classification"
    if "grad_cam" in p or "gradcam" in p or "phase5_grad_cam" in p:
        return "Grad-CAM"
    if "mistaken" in p or "mistake" in p or "phase4_mistake" in p:
        return "Mistake Analysis"
    return "Other"

section_cards = {
    "Fine-tune": [],
    "Data Prep": [],
    "Supervised Classification": [],
    "Grad-CAM": [],
    "Mistake Analysis": [],
    "Other": [],
}

for rel in file_groups.get("Visualizations", []):
    base = os.path.basename(rel)
    ext = os.path.splitext(base)[1].lower()
    sec = section_for(rel)
    display_rel = rel
    if ext in {".tif", ".tiff"}:
        preview_rel = ensure_png_preview(outdir, rel)
        if preview_rel:
            display_rel = preview_rel
    if display_rel.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
        section_cards[sec].append(
            f"""
          <div class="fig"><h4>{html_escape(base)}</h4>
            <img src="{html_escape(display_rel)}" alt="{html_escape(base)}"/>
            <p><a href="{html_escape(rel)}" download class="pill">Download original</a></p>
          </div>
        """
        )
    else:
        section_cards[sec].append(
            f"""
          <div class="fig"><h4>{html_escape(base)}</h4>
            <div class="muted">Preview unavailable.</div>
            <p><a href="{html_escape(rel)}" download class="pill">Download</a></p>
          </div>
        """
        )

for rel in file_groups.get("PDFs", []):
    base = os.path.basename(rel)
    sec = section_for(rel)
    section_cards[sec].append(
        f"""
      <div class="fig"><h4>{html_escape(base)}</h4>
        <div class="pdf-wrap" style="border:1px solid var(--border);border-radius:10px;overflow:hidden">
          <iframe src="{html_escape(rel)}" style="width:100%;height:640px;border:0;"></iframe>
        </div>
        <p><a href="{html_escape(rel)}" download class="pill">Download PDF</a></p>
      </div>
    """
    )

def render_section(title, cards):
    if not cards:
        return ""
    return f"""
      <div class="section" style="margin-top:14px">
        <div class="section-header"><h2>{html_escape(title)}</h2></div>
        <div class="fig-grid">
          {''.join(cards)}
        </div>
      </div>
    """

FIGURES_BLOCK = (
    render_section("Fine-tune", section_cards["Fine-tune"])
    + render_section("Data Prep", section_cards["Data Prep"])
    + render_section("Supervised Classification", section_cards["Supervised Classification"])
    + render_section("Grad-CAM", section_cards["Grad-CAM"])
    + render_section("Mistake Analysis", section_cards["Mistake Analysis"])
    + render_section("Other", section_cards["Other"])
)
if not any(section_cards.values()):
    FIGURES_BLOCK = "<p class='muted'>No figures found.</p>"

run_metrics = gather_run_metrics(outdir)
metric_rows = (
    "".join(
        f"<tr><td>{html_escape(k)}</td><td class='num'>{html_escape(v)}</td></tr>"
        for k, v in run_metrics.items()
    )
    if run_metrics
    else ""
)

RUN_METRICS_BLOCK = f"""
<div class="grid-kpi" style="margin-top:0">
  <div class="kpi"><h3>Report date</h3><div class="val">{datetime.now().strftime('%Y-%m-%d')}</div></div>
  <div class="kpi"><h3>Accuracy</h3><div class="val">{html_escape(clf_accuracy)}</div></div>
  <div class="kpi"><h3>Precision</h3><div class="val">{html_escape(clf_precision)}</div></div>
  <div class="kpi"><h3>Recall</h3><div class="val">{html_escape(clf_recall)}</div></div>
  <div class="kpi"><h3>F1</h3><div class="val">{html_escape(clf_f1)}</div></div>
  <div class="kpi"><h3>AUC</h3><div class="val">{html_escape(clf_auc)}</div></div>
</div>
<div class="table-wrap">
  <table id="metricsTable" class="display nowrap compact" style="width:100%">
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      {metric_rows if metric_rows else "<tr><td colspan='2' class='muted'>No run metrics detected.</td></tr>"}
    </tbody>
  </table>
</div>
"""

logs_blocks = []
for rel in file_groups.get("Logs", []):
    full = os.path.join(outdir, rel)
    logs_blocks.append(
        f"""
    <div class="section" style="margin-top:12px">
      <div class="section-header"><h2>🗒️ {html_escape(rel)}</h2><span class="pill">preview (truncated)</span></div>
      {preview_text_file(full)}
      <p><a href="{html_escape(rel)}" download class="pill">Download full log</a></p>
    </div>
    """
    )
LOGS_BLOCK = "\n".join(logs_blocks) if logs_blocks else "<p class='muted'>No logs found.</p>"

download_links = []
for group_name in [
    "Results / Statistics",
    "Visualizations",
    "PDFs",
    "Model Weights",
    "Config & Requirements",
    "Other",
]:
    for rel in file_groups.get(group_name, []):
        download_links.append(
            f'<a href="{html_escape(rel)}" download class="pill">{html_escape(os.path.basename(rel))}</a>'
        )
DOWNLOADS_BLOCK = (
    "<p style='display:flex;flex-wrap:wrap;gap:8px;margin:.4rem 0 0'>"
    + (zip_button_html + " " if zip_button_html else "")
    + (
        "".join(download_links)
        if download_links
        else "<span class='muted'>No downloadable files found.</span>"
    )
    + "</p>"
)

RUN_CONFIG_KPIS = f"""
<div class="grid-kpi">
  <div class="kpi"><h3>Model ID</h3><div class="val">{html_escape(model_id)}</div></div>
  <div class="kpi"><h3>Crop Size</h3><div class="val">{html_escape(crop_size)}</div></div>
  <div class="kpi"><h3>Epochs</h3><div class="val">{html_escape(epochs)}</div></div>
  <div class="kpi"><h3>Pixel Resolution</h3><div class="val">{html_escape(pixel_res)}</div></div>
  <div class="kpi"><h3>Backbone</h3><div class="val">{html_escape(backbone)}</div></div>
  <div class="kpi"><h3>Mask File</h3><div class="val" style="font-size:1.15rem">{html_escape(mask_file)}</div></div>
  <div class="kpi"><h3>Config File</h3><div class="val" style="font-size:1.05rem">{html_escape(config_file)}</div></div>
</div>
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>EDITS | Pipeline Report</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.5/css/jquery.dataTables.min.css"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.1/css/buttons.dataTables.min.css"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css"/>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs/themes/prism.min.css"/>
  <style>
    :root{
      --bg:#0b1020; --bg-soft:#0f1530; --card-bg:rgba(255,255,255,.06);
      --accent:#2f63f0; --accent-2:#57d3b0; --danger:#ff6b6b;
      --text:#e9eefc; --muted:#a8b0ca; --table-head:#f1f4ff; --border:rgba(255,255,255,.12);
      --shadow:0 12px 30px rgba(0,0,0,.35); --radius:18px; --font:'Inter',system-ui,Arial,sans-serif; --mono:'JetBrains Mono',ui-monospace,Consolas,monospace;
      --space:14px; --lh:1.6; --rowpad:.68rem;
    }
    [data-theme="light"]{
      --bg:#f6f7fb; --bg-soft:#eef1fb; --card-bg:#fffffff2;
      --accent:#2f63f0; --accent-2:#00b894; --danger:#e74c3c;
      --text:#13172a; --muted:#667099; --table-head:#111524; --border:#e7ebf5; --shadow:0 10px 28px rgba(41,62,133,.12)
    }
    *{box-sizing:border-box}
    html,body{
      margin:0;
      background:
        radial-gradient(1000px 600px at 10% 0%, var(--bg-soft), transparent),
        radial-gradient(800px 600px at 120% -10%, rgba(122,162,255,.15), transparent),
        var(--bg);
      color:var(--text);
      font-family:var(--font);
      font-size:16px;
      line-height:var(--lh);
    }
    .container{max-width:1260px;margin:22px auto 80px;padding:0 20px}
    .utilbar{position:fixed; right:18px; top:14px; display:flex; gap:8px; z-index:1000;}
    .utilbar .pill-btn{
      border:1px solid var(--border); background:var(--card-bg); color:var(--accent);
      border-radius:12px; padding:.48rem .7rem; cursor:pointer; box-shadow:var(--shadow);
      display:inline-flex; align-items:center; gap:8px; font-weight:600; font-size:.9rem;
    }
    .utilbar .pill-btn:hover{ filter:brightness(.98); }
    :focus-visible{
      outline:3px solid color-mix(in srgb, var(--accent) 70%, white);
      outline-offset:2px; border-radius:10px;
    }
    @media (prefers-reduced-motion: reduce){
      *{ animation:none !important; transition:none !important; }
      .fig:hover{ transform:none !important; }
      .fig img:hover{ transform:none !important; }
    }
    .hero{
      position:relative;border-radius:28px;padding:26px 22px;
      background:linear-gradient(135deg, rgba(122,162,255,.18), rgba(0,184,148,.14)) , var(--card-bg);
      border:1px solid var(--border); box-shadow:var(--shadow); overflow:hidden
    }
    .hero h1{font-size:clamp(1.6rem, 2.2vw + 1rem, 2.4rem);letter-spacing:-.02em;margin:0;color:var(--table-head);font-weight:800}
    .subhead{color:var(--muted);margin:.35rem 0 .8rem}
    .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    .chip{display:inline-flex;gap:8px;align-items:center;border:1px solid var(--border);padding:.38rem .6rem;border-radius:999px;font-size:.82rem;color:var(--muted);backdrop-filter:saturate(1.1) blur(6px);background:var(--card-bg)}
    .chip i{opacity:.85}
    .section{background:var(--card-bg);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:calc(var(--space) + 6px);margin:18px 0 28px}
    .section-header{display:flex;gap:12px;flex-wrap:wrap;justify-content:space-between;align-items:center;margin:-4px 0 10px}
    .section h2{font-weight:800;letter-spacing:-.02em;margin:0;font-size:clamp(1.05rem, .7vw + .8rem, 1.25rem)}
    .pill{display:inline-flex;gap:8px;align-items:center;border:1px solid var(--border);padding:.32rem .6rem;border-radius:999px;font-size:.8rem;color:var(--muted);background:transparent}
    .kpi{background:linear-gradient(180deg, rgba(255,255,255,.06), transparent), var(--card-bg);border:1px solid var(--border);border-radius:16px;box-shadow:var(--shadow);padding:18px}
    .grid-kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:14px 0 10px}
    .kpi h3{margin:0 0 4px;color:var(--muted);font-size:.92rem;font-weight:700}
    .kpi .val{font-size:1.7rem;font-weight:900;letter-spacing:-.02em;color:var(--table-head)}
    .muted{color:var(--muted)}
    .tabbed{position:relative}
    .tabs{
      position:sticky; top:8px; z-index:999; display:flex; gap:8px;
      border:1px solid var(--border); background:var(--card-bg);
      backdrop-filter:blur(8px) saturate(1.2); padding:6px; border-radius:14px; margin:6px 0 14px; box-shadow:var(--shadow);
      overflow:auto; scroll-snap-type:x mandatory; scrollbar-width:none;
    }
    .tabs::-webkit-scrollbar{ display:none; }
    .tab{
      background:transparent;border:none;color:var(--accent);font-weight:800;padding:.55rem .9rem;border-radius:10px;cursor:pointer; white-space:nowrap; scroll-snap-align:start;
    }
    .tab.active{background:linear-gradient(180deg, rgba(122,162,255,.12), transparent);border:1px solid var(--border);color:var(--table-head)}
    .tab:hover{opacity:.95}
    .tab-content{display:none;animation:fade .18s ease}
    .tab-content.active{display:block}
    @keyframes fade{from{opacity:.6;transform:translateY(2px)}to{opacity:1;transform:none}}
    .fig-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
    .fig{background:linear-gradient(180deg, rgba(255,255,255,.05), transparent), var(--card-bg);border:1px solid var(--border);border-radius:14px;padding:12px;transition:transform .15s ease}
    .fig:hover{transform:translateY(-2px)}
    .fig h4{margin:6px 6px 10px;font-weight:800;font-size:.94rem;color:var(--table-head)}
    .fig img{width:100%;height:auto;border-radius:10px;border:1px solid var(--border);display:block;cursor:zoom-in;transition:transform .2s ease; aspect-ratio:16/10; object-fit:contain}
    .fig img:hover{transform:scale(1.02)}
    .pdf-wrap{border:1px solid var(--border);border-radius:12px;overflow:hidden;box-shadow:var(--shadow);background:#fff}
    .table-wrap{width:100%;overflow:auto;border-radius:12px;border:1px solid var(--border)}
    table.dataTable thead th{color:var(--table-head)!important}
    table.dataTable tbody td, table.dataTable thead th{ padding: var(--rowpad) .75rem; }
    [data-dense] table.dataTable tbody td, [data-dense] table.dataTable thead th{ padding:.35rem .5rem !important; }
    .btn-csv{background:var(--accent)!important;border:none!important;color:#fff!important;border-radius:999px!important;padding:.45rem .8rem!important}
    .btn-csv:hover{filter:brightness(.95)}
    pre{background:#0d1024; color:#cfe2ff; border:1px solid var(--border); border-radius:10px; padding:12px; overflow:auto; line-height:1.45; font-family:var(--mono); font-size:.86rem}
    [data-theme="light"] pre{background:#0f1433; color:#e6ecff}
    .footer{margin-top:28px;text-align:center;color:var(--muted);font-size:.95rem}
    ::selection{background:var(--accent);color:#fff}
    @media (max-width:760px){.hero h1{font-size:1.7rem}}
    #modal-zoom{position:fixed;inset:0;background:rgba(0,0,0,.86);display:none;align-items:center;justify-content:center;z-index:9999;cursor:zoom-out;padding:2em}
    #modal-zoom img{max-width:97vw;max-height:96vh;border-radius:16px;box-shadow:0 0 40px rgba(0,0,0,.5)}
    @media print{
      .utilbar,.tabs{display:none}
      body{background:#fff}
      .section{box-shadow:none}
      .pdf-wrap iframe{height:420px!important}
      table.dataTable{ font-size:11px; }
      .kpi .val{ font-size:1.3rem; }
      .fig img{ max-height:320px; object-fit:contain; }
    }
  </style>
</head>
<body data-auto="1">
  <div class="utilbar" aria-label="Utilities">
    <button class="pill-btn" id="theme-toggle" title="Toggle theme"><i class="fa-solid fa-circle-half-stroke"></i><span>Theme</span></button>
    <button class="pill-btn" id="print-btn" title="Print / Save PDF"><i class="fa-solid fa-print"></i><span>Print</span></button>
    <button class="pill-btn" id="copy-link" title="Copy report link"><i class="fa-regular fa-copy"></i><span>Copy link</span></button>
    <button class="pill-btn" id="to-top" title="Back to top"><i class="fa-solid fa-arrow-up"></i><span>Top</span></button>
  </div>
  <div class="container">
    <div class="hero">
      <h1>🔬 EDITS Report</h1>
      <div class="subhead"><strong>Date:</strong> <span id="today"></span></div>
      <div class="chips">
        <span class="chip"><i class="fa-solid fa-microchip"></i> Backbone</span>
        <span class="chip">{backbone}</span>
        <span class="chip"><i class="fa-solid fa-vector-square"></i> Crop</span>
        <span class="chip">{crop}</span>
        <span class="chip"><i class="fa-solid fa-layer-group"></i> Epochs</span>
        <span class="chip">{epochs}</span>
        <span class="chip"><i class="fa-solid fa-ruler-combined"></i> Pixel Res</span>
        <span class="chip">{pixres}</span>
      </div>
    </div>
    <div class="section" style="margin-top:18px">
      <div class="section-header"><h2>📌 Summary</h2></div>
      <div>{summary_text}</div>
    </div>
    <div class="section">
      <div class="section-header"><h2>⚙️ Run Configuration</h2><span class="pill"><i class="fa-solid fa-file-lines"></i> {config_file}</span></div>
      {run_config_kpis}
    </div>
    <div class="tabbed section">
      <div class="tabs" id="tabs" role="tablist" aria-label="Report sections">
        <button class="tab active" data-tab="t_tables" role="tab" aria-controls="t_tables" aria-selected="true"><i class="fa-solid fa-table"></i> Tables</button>
        <button class="tab" data-tab="t_clf" role="tab" aria-controls="t_clf" aria-selected="false"><i class="fa-solid fa-clipboard-check"></i> Classification</button>
        <button class="tab" data-tab="t_figs" role="tab" aria-controls="t_figs" aria-selected="false"><i class="fa-solid fa-images"></i> Figures</button>
        <button class="tab" data-tab="t_metrics" role="tab" aria-controls="t_metrics" aria-selected="false"><i class="fa-solid fa-gauge-high"></i> Run metrics</button>
        <button class="tab" data-tab="t_logs" role="tab" aria-controls="t_logs" aria-selected="false"><i class="fa-solid fa-scroll"></i> Logs</button>
        <button class="tab" data-tab="t_dl" role="tab" aria-controls="t_dl" aria-selected="false"><i class="fa-solid fa-download"></i> Downloads</button>
      </div>
      <div id="t_tables" class="tab-content active" role="tabpanel" aria-labelledby="t_tables">
        <div class="section-header" style="margin-bottom:6px">
          <h2 style="margin-bottom:0">Tables</h2>
          <label class="pill" for="dense"><input type="checkbox" id="dense" style="margin-right:8px"> Dense</label>
        </div>
        {tables_block}
      </div>
      <div id="t_clf" class="tab-content" role="tabpanel" aria-labelledby="t_clf">
        <div class="section-header"><h2>✅ Classification</h2><span class="pill">confusion matrices • metrics • class balance</span></div>
        {classification_block}
      </div>
      <div id="t_figs" class="tab-content" role="tabpanel" aria-labelledby="t_figs">
        <div class="section-header"><h2>🖼️ Figures</h2><span class="pill">embedded previews</span></div>
        {figures_block}
      </div>
      <div id="t_metrics" class="tab-content" role="tabpanel" aria-labelledby="t_metrics">
        <div class="section-header"><h2>⏱️ Run metrics</h2><span class="pill">runtime • RAM • CPUs • results size</span></div>
        {run_metrics_block}
      </div>
      <div id="t_logs" class="tab-content" role="tabpanel" aria-labelledby="t_logs">
        <div class="section-header"><h2>🗒️ Logs</h2></div>
        {logs_block}
      </div>
      <div id="t_dl" class="tab-content" role="tabpanel" aria-labelledby="t_dl">
        <div class="section-header"><h2>⬇️ Downloads</h2><span class="pill">files detected in output</span></div>
        {downloads_block}
      </div>
    </div>
    <div class="footer">EDITS — Open Source — &copy; {year}</div>
  </div>
  <div id="modal-zoom" aria-hidden="true"><img src="" alt="zoom"/></div>
  <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/buttons/2.4.1/js/dataTables.buttons.min.js"></script>
  <script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.html5.min.js"></script>
  <script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.colVis.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs/prism.min.js"></script>
  <script>
    document.getElementById("today").textContent = new Date().toLocaleDateString();
    document.body.innerHTML = document.body.innerHTML
      .replace("{backbone}", {backbone_json})
      .replace("{crop}", {crop_json})
      .replace("{epochs}", {epochs_json})
      .replace("{pixres}", {pixres_json});
    $('#dense').on('change', function(){
      if(this.checked){ document.documentElement.setAttribute('data-dense', ''); }
      else{ document.documentElement.removeAttribute('data-dense'); }
      setTimeout(()=>{$($.fn.dataTable.tables(true)).DataTable().columns.adjust();},0);
    });
    function adjustTables(){ setTimeout(()=>{$($.fn.dataTable.tables(true)).DataTable().columns.adjust();},0); }
    function toggleTheme(){
      const r=document.documentElement;
      const t=r.getAttribute("data-theme")==="light"?"dark":"light";
      r.setAttribute("data-theme",t);
      localStorage.setItem("edits-theme",t);
      adjustTables();
    }
    (function(){
      const t=localStorage.getItem("edits-theme");
      if(t) document.documentElement.setAttribute("data-theme",t);
    })();
    document.getElementById("theme-toggle").onclick = toggleTheme;
    document.getElementById('print-btn').onclick = ()=>window.print();
    document.getElementById('copy-link').onclick = async ()=>{
      try{
        await navigator.clipboard.writeText(window.location.href);
        const btn = document.getElementById('copy-link');
        const old = btn.innerHTML; btn.innerHTML = '<i class="fa-solid fa-check"></i><span>Copied</span>';
        setTimeout(()=>{ btn.innerHTML = old; }, 1200);
      }catch(e){ console.log(e); }
    };
    document.getElementById('to-top').onclick = ()=>window.scrollTo({top:0, behavior:'smooth'});
    let lastScroll = {};
    function activateTab(id){
      const active = document.querySelector('.tab-content.active');
      if(active){ lastScroll[active.id] = document.scrollingElement.scrollTop; }
      $(".tab").removeClass("active"); $(`.tab[data-tab="${id}"]`).addClass("active").attr('aria-selected','true');
      $(".tab-content").removeClass("active"); $("#"+id).addClass("active");
      localStorage.setItem("edits-tab", id);
      setTimeout(()=>{ window.scrollTo(0, lastScroll[id]||0); }, 0);
      adjustTables();
    }
    $(document).on("click",".tab",function(){ activateTab($(this).data("tab")); });
    (function(){
      const saved = localStorage.getItem("edits-tab");
      if(saved && document.getElementById(saved)) activateTab(saved);
    })();
    $(document).ready(function(){
      {table_init_js}
      if (document.getElementById('metricsTable')){
        $('#metricsTable').DataTable({
          autoWidth:false, scrollX:true, paging:false, searching:false, info:false,
          columnDefs:[{targets:[1], className:'num'}]
        });
      }
      adjustTables();
      window.addEventListener('resize', adjustTables);
    });
    (function(){
      const modal=document.getElementById('modal-zoom');
      const mimg=modal.querySelector('img');
      document.querySelectorAll('.fig img').forEach(img=>{
        img.addEventListener('click',()=>{ mimg.src=img.src; modal.style.display='flex'; modal.setAttribute('aria-hidden','false'); });
      });
      modal.addEventListener('click',()=>{ modal.style.display='none'; mimg.src=''; modal.setAttribute('aria-hidden','true'); });
      document.addEventListener('keydown',e=>{ if(e.key==='Escape' && modal.style.display==='flex'){ modal.click(); }});
    })();
  </script>
</body>
</html>
"""

replacements = {
    "{summary_text}": summary_text,
    "{config_file}": html_escape(config_file),
    "{run_config_kpis}": RUN_CONFIG_KPIS,
    "{tables_block}": TABLES_BLOCK,
    "{classification_block}": CLASSIFICATION_BLOCK,
    "{figures_block}": FIGURES_BLOCK,
    "{run_metrics_block}": RUN_METRICS_BLOCK,
    "{logs_block}": LOGS_BLOCK,
    "{downloads_block}": DOWNLOADS_BLOCK,
    "{year}": str(datetime.now().year),
    "{table_init_js}": "\n      ".join(table_init_js),
    "{backbone}": html_escape(backbone),
    "{crop}": html_escape(str(crop_size)),
    "{epochs}": html_escape(str(epochs)),
    "{pixres}": html_escape(str(pixel_res)),
    "{backbone_json}": json.dumps(str(backbone)),
    "{crop_json}": json.dumps(str(crop_size)),
    "{epochs_json}": json.dumps(str(epochs)),
    "{pixres_json}": json.dumps(str(pixel_res)),
}

html_out = HTML_TEMPLATE
for k, v in replacements.items():
    html_out = html_out.replace(k, v)

report_path = os.path.join(outdir, "report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_out)

print(f"Report saved to {report_path}")
