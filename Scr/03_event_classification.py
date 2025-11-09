#!/usr/bin/env python3
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

custom_dir = '/EDITS/TAP/tarrow/'
sys.path.insert(0, custom_dir)
sys.path.append('/EDITS/TAP/tarrow/tarrow')

import tarrow
import torch
from torch.utils.data import Sampler, DataLoader, RandomSampler
from pathlib import Path
import logging
import torch.nn as nn
import numpy as np
import csv
import yaml
import torch.nn.functional as F
import faulthandler
faulthandler.enable()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def gini_index_from_labels(labels):
    p = float(np.mean(labels))
    return 2.0 * p * (1.0 - p)


def log_normalized_shannon_entropy_from_labels(labels):
    ps = np.bincount(labels.astype(int))
    s = ps.sum()
    ps = ps / (s if s > 0 else 1.0)
    ps = ps[ps > 0]
    if len(ps) == 0:
        return 0.0
    entropy = -np.sum(ps * np.log(ps))
    max_entropy = np.log(2)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def imbalance_ratio_from_labels(labels):
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return float(n_pos) / float(max(n_neg, 1))


def summarize_labels(name, labels):
    labels = np.asarray(labels).astype(int)
    n_total = labels.size
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    ir = imbalance_ratio_from_labels(labels)
    gi = gini_index_from_labels(labels)
    lse = log_normalized_shannon_entropy_from_labels(labels)
    print(f"\n[{name}]")
    print(f"  Total: {n_total}")
    print(f"  Pos (class 1): {n_pos}")
    print(f"  Neg (class 0): {n_neg}")
    print(f"  Imbalance ratio (pos/neg): {ir:.4f}  ({ir*100:.2f}%)")
    print(f"  Gini index: {gi:.4f}")
    print(f"  Log-normalized Shannon entropy: {lse:.4f}")
    return {
        "Total": n_total,
        "Pos (class 1)": n_pos,
        "Neg (class 0)": n_neg,
        "Imbalance ratio (pos/neg)": ir,
        "Imbalance ratio %": ir * 100.0,
        "Gini index": gi,
        "Log-norm Shannon entropy": lse,
    }


def save_stats_csv(stats_dict, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"class_balance_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in stats_dict.items():
            w.writerow([k, v])
    print(f"  ↳ saved to {path}")


def load_tap_model_robust(model_dir, device=None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"TAP model folder does not exist: {model_dir}")
    kwargs_path = model_dir / "model_kwargs.yaml"
    if not kwargs_path.exists():
        raise FileNotFoundError(f"model_kwargs.yaml not found in {model_dir}")
    with open(kwargs_path, "r") as f:
        kw = yaml.safe_load(f) or {}
    candidates = [
        model_dir / "model_latest.pt",
        model_dir / "model.pt",
        model_dir / "model_full.pt",
        model_dir / "model_state.pt",
    ]
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.exists():
        candidates += sorted(ckpt_dir.glob("epoch_*.pt"))
    candidates += sorted(model_dir.glob("*.pth"))
    state_dict = None
    found_path = None
    for ckpt in candidates:
        if ckpt.exists():
            blob = torch.load(ckpt, map_location="cpu")
            state_dict = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
            found_path = ckpt
            break
    if state_dict is None:
        look_list = ", ".join(p.name for p in candidates)
        raise FileNotFoundError(f"No weights found in {model_dir}. Looked for: {look_list}")
    safe_outdir = model_dir / "__loaded_eval_artifacts"
    backbone = kw.get("backbone", "unet")
    projhead = kw.get("projection_head", kw.get("projhead", "minimal_batchnorm"))
    classhead = kw.get("classification_head", kw.get("classhead", "minimal"))
    n_frames = kw.get("n_frames", 2)
    n_input_channels = kw.get("n_input_channels", 1)
    n_features = kw.get("n_features", kw.get("features", 32))
    symmetric = kw.get("symmetric", True)
    model = tarrow.models.TimeArrowNet(
        backbone=backbone,
        projection_head=projhead,
        classification_head=classhead,
        n_frames=n_frames,
        n_input_channels=n_input_channels,
        n_features=n_features,
        symmetric=symmetric,
        device=torch.device("cpu"),
        outdir=str(safe_outdir),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[TAP] load_state_dict(strict=False)  missing={len(missing)}  unexpected={len(unexpected)}")
    model.to(torch.device(device))
    model.eval()
    print(f"[TAP] Loaded weights from: {found_path}")
    return model


def _flatten_features(x):
    if x.ndim > 2:
        return x.view(x.size(0), -1)
    return x


class LinearHead(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = _flatten_features(x)
        return self.fc(x)


class MinimalHead(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2):
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = _flatten_features(x)
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


class ResNetHead(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=2):
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = _flatten_features(x)
        x = self.fc1(x)
        x = F.relu(x)
        r = x
        x = self.fc2(x)
        x = F.relu(x + r)
        return self.fc_out(x)


def reinitialize_weights(model):
    import torch.nn.init as init
    for _, layer in model.named_modules():
        if hasattr(layer, "weight") and layer.weight is not None:
            if len(layer.weight.shape) >= 2:
                init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            else:
                init.normal_(layer.weight, mean=0.0, std=1.0)
        if hasattr(layer, "bias") and layer.bias is not None:
            init.zeros_(layer.bias)


def save_confusion_matrix_plot(cm, labels, out_pdf_path):
    sns.set(style="whitegrid", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
        }
    )
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 11},
        square=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.tick_params(axis="x", labelrotation=0, labelsize=10, pad=4)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10, pad=4)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    plt.savefig(out_pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_csv(cm, labels, out_csv_path):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(out_csv_path)


def save_runs_metrics_csv(per_run_rows, out_csv_path):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df = pd.DataFrame(per_run_rows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary_mean = df[numeric_cols].mean().to_dict()
    summary_std = df[numeric_cols].std(ddof=0).to_dict()
    summary_mean["run"] = "mean"
    summary_std["run"] = "std"
    df = pd.concat([df, pd.DataFrame([summary_mean, summary_std])], ignore_index=True)
    df.to_csv(out_csv_path, index=False)


def train_cls_head(
    train_loader,
    test_loader,
    patch_size,
    num_epochs,
    random_seed,
    device,
    model_load_dir,
    cls_head_arch,
    TAP_init,
    load_saved_cls_head=False,
    cls_head_load_path=None,
):
    import torch.optim as optim
    from sklearn.metrics import confusion_matrix, classification_report

    model = load_tap_model_robust(model_load_dir, device=device)
    model.to(device)

    if TAP_init in ("km_uniform", "km_init"):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        reinitialize_weights(model)
        print(f"- - - Initialising TAP model using {TAP_init} - - - ")
    elif TAP_init == "loaded":
        print("- - - Initialising TAP model using loaded weights - - - ")

    for p in model.parameters():
        p.requires_grad = False

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    if cls_head_arch == "linear":
        cls_head = LinearHead(num_classes=2).to(device)
    elif cls_head_arch == "minimal":
        cls_head = MinimalHead(num_classes=2).to(device)
    elif cls_head_arch == "resnet":
        cls_head = ResNetHead(num_classes=2).to(device)
    else:
        print(f"[warn] Unknown cls_head_arch='{cls_head_arch}', defaulting to 'linear'.")
        cls_head = LinearHead(num_classes=2).to(device)

    if load_saved_cls_head and cls_head_load_path:
        print(" - - Loading pretrained cls head - - ")
        cls_head_state_dict = torch.load(cls_head_load_path, map_location=device)
        cls_head.load_state_dict(cls_head_state_dict)

    optimizer = optim.Adam(cls_head.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        cls_head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for datapoint in train_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            with torch.no_grad():
                rep = model.embedding(x)
            outputs = cls_head(rep)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        epoch_loss = running_loss / max(total, 1)
        epoch_accuracy = correct / max(total, 1)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    running_loss = 0.0
    correct = 0
    total = 0
    count_event_interest = 0
    y_pred, y_true = [], []
    cls_head.eval()
    with torch.no_grad():
        for datapoint in test_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            rep = model.embedding(x)
            outputs = cls_head(rep)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            count_event_interest += (y == 1).sum().item()
            y_pred.extend([t.item() for t in predicted])
            y_true.extend([t.item() for t in y])

    epoch_loss = running_loss / max(total, 1)
    epoch_accuracy = correct / max(total, 1)
    print(f"Test Loss: {epoch_loss:.4f}, Test accuracy: {epoch_accuracy:.4f}")
    print(f"There are {count_event_interest} out of {total} crops containing events of interest in the test set")

    from sklearn.metrics import confusion_matrix, classification_report

    cm_test = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm_test, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    print("Confusion Matrix test data:")
    print(cm_df)
    print(classification_report(y_true, y_pred, target_names=["class 0", "class 1"]))

    count_event_interest_train = 0
    total = 0
    y_pred, y_true = [], []
    with torch.no_grad():
        for datapoint in train_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            count_event_interest_train += (y == 1).sum().item()
            total += y.size(0)
            rep = model.embedding(x)
            outputs = cls_head(rep)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend([t.item() for t in predicted])
            y_true.extend([t.item() for t in y])

    print(f"There are {count_event_interest_train} out of {total} crops containing events of interest in the training set")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    print("Confusion Matrix train data:")
    print(cm_df)
    print(classification_report(y_true, y_pred, target_names=["class 0", "class 1"]))

    return cls_head, model, cm_test


def count_data_points(dataloader):
    count = 0
    num_positive_event = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, event_labels = batch[0], batch[1]
        else:
            raise ValueError("Expected batch to be a (inputs, labels, ...) tuple/list.")
        count += inputs.size(0)
        num_positive_event += (event_labels == 1).sum().item()
    ratio = num_positive_event / count if count > 0 else 0
    print(f"Total: {count}, Positives: {num_positive_event} ({ratio*100:.2f}%)")
    return count, num_positive_event


class BalancedSampler(Sampler):
    def __init__(self, data_source, num_crops_per_image, balanced_sample_size, data_gen_seed, sequential=False):
        self.data_source = data_source
        self.sequential = sequential
        self.num_crops_per_image = num_crops_per_image
        self.balanced_sample_size = balanced_sample_size
        self.data_gen_seed = data_gen_seed
        num_image_pairs = len(self.data_source)
        self.positive_indices = []
        self.negative_indices = []
        for i in range(num_image_pairs):
            if data_source[i][1] > 0:
                self.positive_indices.append(i)
            if data_source[i][1] == 0:
                self.negative_indices.append(i)
        self.num_samples_per_class = min(
            self.balanced_sample_size // 2, len(self.positive_indices), len(self.negative_indices)
        )

    def get_combined_samples(self, data_gen_seed):
        import random
        torch.manual_seed(data_gen_seed)
        if self.sequential:
            positive_samples = self.positive_indices[: self.num_samples_per_class]
            negative_samples = self.negative_indices[: self.num_samples_per_class]
        else:
            positive_samples = torch.multinomial(
                torch.ones(len(self.positive_indices)), self.num_samples_per_class, replacement=True
            ).tolist()
            positive_samples = [self.positive_indices[i] for i in positive_samples]
            negative_samples = torch.multinomial(
                torch.ones(len(self.negative_indices)), self.num_samples_per_class, replacement=True
            ).tolist()
            negative_samples = [self.negative_indices[i] for i in negative_samples]
        combined_samples = positive_samples + negative_samples
        if not self.sequential:
            random.seed(data_gen_seed + 123)
            combined_samples = random.sample(combined_samples, len(combined_samples))
        return combined_samples

    def __iter__(self):
        combined_samples = self.get_combined_samples(data_gen_seed=self.data_gen_seed)
        return iter(combined_samples)

    def __len__(self):
        return 2 * self.num_samples_per_class


def probing_mistake_predictions(model, cls_head, test_data_loader, device):
    false_positives = []
    false_negatives = []
    logits_false_pos = []
    logits_false_neg = []
    cls_head.eval()
    with torch.no_grad():
        for datapoint in test_data_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            rep = model.embedding(x)
            outputs = cls_head(rep)
            _, predicted = torch.max(outputs, 1)
            if (predicted == 1) and (y == 0):
                false_positives.append(datapoint)
                logits_false_pos.append(torch.squeeze(outputs.detach().cpu()))
            elif (predicted == 0) and (y == 1):
                false_negatives.append(datapoint)
                logits_false_neg.append(torch.squeeze(outputs.detach().cpu()))
    return false_positives, false_negatives, logits_false_pos, logits_false_neg


def probing_mistaken_preds(model, cls_head_trained, test_loader_probing, device):
    (
        false_positives,
        false_negatives,
        logits_false_pos,
        logits_false_neg,
    ) = probing_mistake_predictions(model, cls_head_trained, test_loader_probing, device)
    false_positives_coordinates = [tuple(e[1:]) for e in false_positives]
    false_negatives_coordinates = [tuple(e[1:]) for e in false_negatives]
    print(
        f"number of false_positives predictions: {len(false_positives_coordinates)}\n"
        f"number of false_negatives predictions: {len(false_negatives_coordinates)}"
    )
    return (
        false_positives_coordinates,
        false_negatives_coordinates,
        false_positives,
        false_negatives,
        logits_false_pos,
        logits_false_neg,
    )


def estimate_total_events(input_data):
    total = 0
    for i in range(len(input_data)):
        c = input_data[i][1]
        if isinstance(c, torch.Tensor):
            c = c.detach().item()
        total += c
    return total


def save_as_json(input_data, file_save_path):
    import json
    data_to_save = []
    for item in input_data:
        converted_item = []
        for element in item:
            if hasattr(element, "tolist"):
                converted_item.append(element.tolist())
            else:
                converted_item.append(element)
        data_to_save.append(converted_item)
    with open(file_save_path, "w") as f:
        json.dump(data_to_save, f)
    print(f"data saved to {file_save_path}")


def data_split(input_image_crops, train_data_ratio, validation_data_ratio, data_seed):
    import random
    random.seed(data_seed)
    random.shuffle(input_image_crops)
    total_length = len(input_image_crops)
    train_end = int(train_data_ratio * total_length)
    valid_end = train_end + int(validation_data_ratio * total_length)
    train_data = input_image_crops[:train_end]
    valid_data = input_image_crops[train_end:valid_end]
    test_data = input_image_crops[valid_end:]
    print(f"Total data points: {total_length}")
    print(f"Training data points: {len(train_data)}")
    print(f"Validation data points: {len(valid_data)}")
    print(f"Test data points: {len(test_data)}")
    return train_data, valid_data, test_data


def multi_runs_training(
    num_runs,
    model_seed_init,
    train_loader,
    test_loader,
    size,
    training_epochs,
    device,
    model_load_dir,
    cls_head_arch,
    TAP_init,
    load_saved_cls_head=False,
    cls_head_load_path=None,
):
    precision_class_0_all = []
    precision_class_1_all = []
    recall_class_0_all = []
    recall_class_1_all = []
    cms_test_all = []
    per_run_rows = []
    cls_head_trained = None
    model = None

    for i in range(num_runs):
        run_id = i + 1
        model_seed = model_seed_init + i * 20
        cls_head_trained, model, cm_test = train_cls_head(
            cls_head_arch=cls_head_arch,
            train_loader=train_loader,
            test_loader=test_loader,
            patch_size=size,
            num_epochs=training_epochs,
            random_seed=model_seed,
            device=device,
            model_load_dir=model_load_dir,
            load_saved_cls_head=load_saved_cls_head,
            cls_head_load_path=cls_head_load_path,
            TAP_init=TAP_init,
        )
        cms_test_all.append(cm_test)
        tn, fp, fn, tp = cm_test[0, 0], cm_test[0, 1], cm_test[1, 0], cm_test[1, 1]
        precision_class_0 = tn / max((tn + fn), 1)
        precision_class_1 = tp / max((fp + tp), 1)
        recall_class_0 = tn / max((tn + fp), 1)
        recall_class_1 = tp / max((fn + tp), 1)
        precision_class_0_all.append(precision_class_0)
        precision_class_1_all.append(precision_class_1)
        recall_class_0_all.append(recall_class_0)
        recall_class_1_all.append(recall_class_1)
        per_run_rows.append(
            {
                "run": run_id,
                "seed": model_seed,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision_class_0": precision_class_0,
                "precision_class_1": precision_class_1,
                "recall_class_0": recall_class_0,
                "recall_class_1": recall_class_1,
            }
        )

    return (
        np.array(precision_class_0_all),
        np.array(precision_class_1_all),
        np.array(recall_class_0_all),
        np.array(recall_class_1_all),
        cls_head_trained,
        model,
        cms_test_all,
        per_run_rows,
    )


def save_datasets(train_data_crops_flat, valid_data_crops_flat, test_data_crops_flat, dataset_save_dir):
    os.makedirs(dataset_save_dir, exist_ok=True)
    torch.save(train_data_crops_flat, os.path.join(dataset_save_dir, "train_data_crops_flat.pth"))
    torch.save(valid_data_crops_flat, os.path.join(dataset_save_dir, "valid_data_crops_flat.pth"))
    torch.save(test_data_crops_flat, os.path.join(dataset_save_dir, "test_data_crops_flat.pth"))
    print(f"Train, validation and test data all saved to {dataset_save_dir}")


class CellEventClassModel(nn.Module):
    def __init__(self, TAPmodel, ClsHead):
        super(CellEventClassModel, self).__init__()
        self._TAPmodel = TAPmodel
        self._ClsHead = ClsHead

    def forward(self, _input):
        z = self._TAPmodel.embedding(_input)
        y = self._ClsHead(z)
        return y


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_frame")
    parser.add_argument("--input_mask")
    parser.add_argument("--cam_size", type=int, default=None)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--n_images")
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--binarize")
    parser.add_argument("--timestamp")
    parser.add_argument("--backbone", default="unet")
    parser.add_argument("--name")
    parser.add_argument("--size", type=int, default=96, required=True)
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--batchsize", type=int, default=108)
    parser.add_argument("--cam_subsampling", type=int, default=1)
    parser.add_argument("--training_epochs", type=int, required=True, default=1)
    parser.add_argument("--binary_problem", type=bool, default=True)
    parser.add_argument("--balanced_sample_size", required=True, type=int)
    parser.add_argument("--crops_per_image", required=True, type=int)
    parser.add_argument("--model_seed", required=True, type=int)
    parser.add_argument("--data_seed", required=True, type=int)
    parser.add_argument("--dataset_save_dir")
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--model_save_dir")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--load_saved_cls_head", type=bool, default=False)
    parser.add_argument("--cls_head_load_path", default=None)
    parser.add_argument("--TAP_model_load_path")
    parser.add_argument("--cls_head_arch")
    parser.add_argument("--TAP_init", default="loaded")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.dataset_save_dir:
        args.dataset_save_dir = "results/supervised_classification/datasets"
    if not args.model_save_dir:
        args.model_save_dir = "results/supervised_classification/models"

    dev = str(args.device).lower() if args.device is not None else "auto"
    if dev in ("", "auto", None):
        if torch.cuda.is_available():
            resolved = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        if dev == "cuda":
            if torch.cuda.is_available():
                resolved = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved = "mps"
            else:
                resolved = "cpu"
        else:
            resolved = args.device
    device = torch.device(resolved)
    print("Running on", resolved)

    data_load_path = os.path.join(args.dataset_save_dir, "preprocessed_image_crops.pth")
    if not os.path.isfile(data_load_path):
        raise FileNotFoundError(f"preprocessed_image_crops.pth not found at {data_load_path}")
    image_crops_flat_loaded = torch.load(data_load_path)
    print(f"image_crops_flat_loaded: {len(image_crops_flat_loaded)}")

    train_data_ratio = 0.6
    validation_data_ratio = 0.2
    train_data_crops_flat, valid_data_crops_flat, test_data_crops_flat = data_split(
        image_crops_flat_loaded, train_data_ratio, validation_data_ratio, args.data_seed
    )
    estimated_total_event_count = estimate_total_events(image_crops_flat_loaded)
    print(f"\nEstimated total event count (entire dataset): {estimated_total_event_count}")

    def labels_from_flat(ds):
        out = []
        for x in ds:
            lbl = x[1]
            if isinstance(lbl, torch.Tensor):
                lbl = int(lbl.item())
            else:
                lbl = int(lbl)
            out.append(lbl)
        return np.array(out, dtype=int)

    model_save_dir_path = Path(args.model_save_dir).resolve()
    run_root = model_save_dir_path.parents[1] if len(model_save_dir_path.parents) >= 2 else None
    if run_root is not None and (run_root / "config").exists():
        figures_dir = run_root / "figures" / "supervised"
        metrics_dir = run_root / "metrics"
    else:
        fallback = Path("results") / "supervised_classification" / "figures" / args.model_id
        figures_dir = fallback
        metrics_dir = fallback

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    stats_outdir = metrics_dir / "class_balance"
    os.makedirs(stats_outdir, exist_ok=True)

    labels_train_before = labels_from_flat(train_data_crops_flat)
    labels_valid = labels_from_flat(valid_data_crops_flat)
    labels_test = labels_from_flat(test_data_crops_flat)

    train_before_stats = summarize_labels("Train (BEFORE balancing)", labels_train_before)
    valid_stats = summarize_labels("Validation", labels_valid)
    test_stats = summarize_labels("Test", labels_test)

    save_stats_csv(train_before_stats, str(stats_outdir), "train_before_balancing")
    save_stats_csv(valid_stats, str(stats_outdir), "validation")
    save_stats_csv(test_stats, str(stats_outdir), "test")

    train_loader = DataLoader(
        train_data_crops_flat,
        sampler=BalancedSampler(
            train_data_crops_flat,
            args.crops_per_image,
            args.balanced_sample_size,
            data_gen_seed=args.data_seed,
            sequential=False,
        ),
        batch_size=args.batchsize,
        num_workers=0,
        drop_last=False,
        persistent_workers=False,
    )

    sampler = train_loader.sampler
    indices_balanced = list(sampler.get_combined_samples(args.data_seed))
    labels_after = np.array(
        [
            int(
                train_data_crops_flat[i][1].item()
                if isinstance(train_data_crops_flat[i][1], torch.Tensor)
                else train_data_crops_flat[i][1]
            )
            for i in indices_balanced
        ]
    )
    train_after_stats = summarize_labels("Train (AFTER balancing)", labels_after)
    save_stats_csv(train_after_stats, str(stats_outdir), "train_after_balancing")

    torch.manual_seed(args.data_seed)
    test_loader = DataLoader(
        test_data_crops_flat,
        sampler=RandomSampler(test_data_crops_flat, replacement=False),
        batch_size=args.batchsize,
        num_workers=0,
        drop_last=False,
        persistent_workers=False,
    )

    print(f"The number of events in the combined dataset is estimated to be {estimated_total_event_count}")
    print("number of data points, positive events in balanced training crops, ", count_data_points(train_loader))
    print("number of data points, positive events in test crops: ", count_data_points(test_loader))
    print(f" - - - loading pretrained TAP model from : {args.TAP_model_load_path} - - - ")

    test_loader_probing = DataLoader(
        test_data_crops_flat,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        persistent_workers=False,
    )
    print("number of data points in test_loader_probing, : ", count_data_points(test_loader_probing))

    start_time = time.time()
    (
        precision_class_0_all,
        precision_class_1_all,
        recall_class_0_all,
        recall_class_1_all,
        cls_head_trained,
        model,
        cms_test_all,
        per_run_rows,
    ) = multi_runs_training(
        args.num_runs,
        args.model_seed,
        train_loader,
        test_loader,
        args.size,
        args.training_epochs,
        device,
        args.TAP_model_load_path,
        cls_head_arch=args.cls_head_arch,
        TAP_init=args.TAP_init,
        load_saved_cls_head=args.load_saved_cls_head,
        cls_head_load_path=args.cls_head_load_path,
    )

    precision_class_0_all_mean = round(np.mean(precision_class_0_all), 4)
    precision_class_1_all_mean = round(np.mean(precision_class_1_all), 4)
    recall_class_0_all_mean = round(np.mean(recall_class_0_all), 4)
    recall_class_1_all_mean = round(np.mean(recall_class_1_all), 4)
    precision_class_0_all_std = round(np.std(precision_class_0_all, ddof=0), 4)
    precision_class_1_all_std = round(np.std(precision_class_1_all, ddof=0), 4)
    recall_class_0_all_std = round(np.std(recall_class_0_all, ddof=0), 4)
    recall_class_1_all_std = round(np.std(recall_class_1_all, ddof=0), 4)

    print("mean and standard deviations of each class: ")
    print(f"class 0, precision : {(precision_class_0_all_mean, precision_class_0_all_std)}")
    print(f"class 0, recall    : {(recall_class_0_all_mean, recall_class_0_all_std)}")
    print(f"class 1, precision : {(precision_class_1_all_mean, precision_class_1_all_std)}")
    print(f"class 1, recall    : {(recall_class_1_all_mean, recall_class_1_all_std)}")

    end_time_model_training = time.time()
    print(f"Time used for model fine-tuning : {end_time_model_training - start_time:.2f} seconds")

    metrics_csv_path = metrics_dir / "per_run_metrics.csv"
    save_runs_metrics_csv(per_run_rows, str(metrics_csv_path))
    print(f"Saved per-run metrics CSV: {metrics_csv_path}")

    labels = ["No Event", "Event"]

    agg_cm = np.zeros((2, 2), dtype=int)
    for cm in cms_test_all:
        agg_cm += cm.astype(int)

    agg_cm_csv = metrics_dir / "confusion_matrix_aggregate.csv"
    agg_cm_pdf = figures_dir / "confusion_matrix_aggregate.pdf"
    save_confusion_matrix_csv(agg_cm, labels, str(agg_cm_csv))
    save_confusion_matrix_plot(agg_cm, labels, str(agg_cm_pdf))
    print(f"Saved aggregate CM CSV: {agg_cm_csv}")
    print(f"Saved aggregate CM PDF: {agg_cm_pdf}")

    last_cm = cms_test_all[-1]
    last_cm_csv = metrics_dir / "confusion_matrix_last_run.csv"
    last_cm_pdf = figures_dir / "confusion_matrix_last_run.pdf"
    save_confusion_matrix_csv(last_cm, labels, str(last_cm_csv))
    save_confusion_matrix_plot(last_cm, labels, str(last_cm_pdf))
    print(f"Saved last-run CM CSV: {last_cm_csv}")
    print(f"Saved last-run CM PDF: {last_cm_pdf}")

    TAPmodel = load_tap_model_robust(args.TAP_model_load_path, device=device)
    TAPmodel.to(device)
    for p in TAPmodel.parameters():
        p.requires_grad = False
    cls_head_trained.to(device)
    combined_model = CellEventClassModel(TAPmodel=TAPmodel, ClsHead=cls_head_trained)
    os.makedirs(args.model_save_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_save_dir, f"{args.model_id}.pth")
    torch.save(combined_model.state_dict(), model_save_path)
    print(f"Combined model saved to {model_save_path}")

    save_datasets(
        train_data_crops_flat,
        valid_data_crops_flat,
        test_data_crops_flat,
        os.path.join(args.dataset_save_dir, args.model_id),
    )


if __name__ == "__main__":
    main()
