#!/usr/bin/env python3
import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import faulthandler

faulthandler.enable()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

try:
    import torch as _torch
    _torch.set_num_threads(1)
    _torch.set_num_interop_threads(1)
except Exception:
    pass

custom_dir = "/EDITS/TAP/tarrow/"
sys.path.insert(0, custom_dir)
sys.path.append("/EDITS/TAP/tarrow/tarrow")

import tarrow
import torch
from torch.utils.data import DataLoader
from typing import Sequence, Any, Tuple
from pathlib import Path
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch must have at least (inputs, labels).")
        return batch[0], batch[1]
    if isinstance(batch, dict):
        x = batch.get("x") or batch.get("inputs") or batch.get("input")
        y = batch.get("y") or batch.get("labels") or batch.get("label")
        if x is None or y is None:
            raise ValueError("Dict batch missing 'inputs'/'labels'.")
        return x, y
    raise ValueError(f"Unsupported batch type: {type(batch)}")


def load_tap_model_robust(model_dir: str, device="cpu"):
    import glob
    import yaml

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"TAP model dir not found: {model_dir}")

    candidates = [
        model_dir / "model_latest.pt",
        model_dir / "model.pt",
    ]
    ckpts = sorted(glob.glob(str(model_dir / "checkpoints" / "epoch_*.pt")))
    candidates.extend([Path(p) for p in ckpts])
    candidates.append(model_dir / "model_state.pt")
    candidates.append(model_dir / "model_full.pt")

    weights_path = None
    for p in candidates:
        if p.is_file():
            weights_path = p
            break
    if weights_path is None:
        names = ", ".join([c.name for c in candidates])
        raise FileNotFoundError(f"No weights found in {model_dir}. Looked for: {names}")

    kwargs_yaml = model_dir / "model_kwargs.yaml"
    if kwargs_yaml.is_file():
        try:
            with open(kwargs_yaml, "r") as f:
                kw = yaml.safe_load(f) or {}
        except Exception:
            kw = {}
        try:
            model = tarrow.models.TimeArrowNet(**kw)
        except Exception:
            model = tarrow.models.TimeArrowNet(
                backbone=kw.get("backbone", "unet"),
                projection_head=kw.get("projection_head", kw.get("projhead", "minimal_batchnorm")),
                classification_head=kw.get("classification_head", kw.get("classhead", "minimal")),
                n_frames=kw.get("n_frames", 2),
                n_input_channels=kw.get("n_input_channels", 1),
                n_features=kw.get("n_features", kw.get("features", 32)),
                symmetric=kw.get("symmetric", True),
                device=torch.device("cpu"),
                outdir=str(model_dir / "__loaded_eval_artifacts"),
            )
    else:
        model = tarrow.models.TimeArrowNet(
            backbone="unet",
            projection_head="minimal_batchnorm",
            classification_head="minimal",
            n_frames=2,
            n_input_channels=1,
            n_features=32,
            symmetric=True,
            device=torch.device("cpu"),
            outdir=str(model_dir / "__loaded_eval_artifacts"),
        )

    obj = torch.load(weights_path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state" in obj:
        state = obj["model_state"]
    elif isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
    elif isinstance(obj, dict):
        state = obj
    else:
        state = obj

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[TAP] load_state_dict(strict=False) missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    print(f"[TAP] Loaded weights from: {weights_path}")
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
        residual = x
        x = self.fc2(x)
        x = F.relu(x + residual)
        return self.fc_out(x)


def _get_paths_recursive(paths: Sequence[str], level: int):
    input_rec = paths
    for _ in range(level):
        new_inps = []
        for p in input_rec:
            p = Path(p)
            if p.is_dir():
                children = [x for x in p.iterdir() if x.is_dir() or x.suffix == ".tif"]
                new_inps.extend(children)
            if p.suffix == ".tif":
                new_inps.append(p)
        input_rec = new_inps
    return input_rec


def _load_image_folder(fnames, split_start: float, split_end: float):
    import tifffile
    import imageio
    from tqdm import tqdm

    idx_start = int(len(fnames) * split_start)
    idx_end = int(len(fnames) * split_end)
    sel = fnames[idx_start:idx_end]

    imgs = []
    for f in tqdm(sel, leave=False, desc="loading images"):
        f = Path(f)
        if f.suffix.lower() in (".tif", ".tiff"):
            x = tifffile.imread(str(f))
        elif f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            x = imageio.imread(f)
            if x.ndim == 3:
                x = np.moveaxis(x[..., :3], -1, 0)
        else:
            continue
        x = np.squeeze(x)
        imgs.append(x)
    return np.stack(imgs)


def _load(path, split_start, split_end, n_images=None):
    import tifffile

    assert split_start >= 0
    assert split_end <= 1

    inp = Path(path).expanduser()

    if inp.is_dir():
        suffixes = ("png", "jpg", "tif", "tiff")
        fnames = []
        for s in suffixes:
            cand = sorted(inp.glob(f"*.{s}"))
            if cand:
                fnames = cand
                break
        if not fnames:
            raise ValueError(f"Could not find any images in {inp}")
        fnames = fnames[:n_images]
        imgs = _load_image_folder(fnames, split_start, split_end)
    elif inp.suffix.lower() in (".tif", ".tiff"):
        logger.info(f"Loading {inp}")
        imgs = tifffile.imread(str(inp))
        logger.info("Done")
        print(f"imags shape : {imgs.shape}")
        assert imgs.ndim == 3
        imgs = imgs[int(len(imgs) * split_start) : int(len(imgs) * split_end)]
        imgs = imgs[:n_images]
    else:
        raise ValueError(f"Cannot form a dataset from {inp}.")
    return imgs


def plot_images_gray_scale(image1, image2, mask1, mask2, save_path):
    image1_np = image1.squeeze().numpy()
    image2_np = image2.squeeze().numpy()
    mask1_np = mask1.squeeze().numpy()
    mask2_np = mask2.squeeze().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    axes[0].imshow(image1_np, cmap="gray")
    axes[0].axis("off")
    axes[1].imshow(image2_np, cmap="gray")
    axes[1].axis("off")
    axes[2].imshow(mask1_np, cmap="gray")
    axes[2].axis("off")
    axes[3].imshow(mask2_np, cmap="gray")
    axes[3].axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def masks_prep(n_frames, masks_path, delta_frames=1):
    import numpy as np

    inputs_mask = [_get_paths_recursive(masks_path, 0)]
    masks_sequences = []
    for masks in inputs_mask:
        if isinstance(masks, (str, Path)):
            masks = _load(path=masks, split_start=0, split_end=1)
        elif isinstance(masks, (tuple, list, np.ndarray)) and isinstance(masks[0], np.ndarray):
            masks = np.asarray(masks)
        else:
            raise ValueError(f"Cannot form a dataset from {masks}.")
        masks = torch.as_tensor(masks)
        for delta in [delta_frames]:
            n, k = n_frames, delta
            tslices = tuple(
                slice(i, i + k * (n - 1) + 1, k) for i in range(len(masks) - (n - 1) * k)
            )
            seq = [torch.as_tensor(masks[ss]) for ss in tslices]
            masks_sequences.extend(seq)
    return masks_sequences


def masks_lookup(coord_x, coord_y, patch_size, time_index, imgs_masks_sequences):
    from torchvision import transforms

    x, y = imgs_masks_sequences[time_index]
    m1 = transforms.functional.crop(x, coord_x, coord_y, patch_size, patch_size)
    m2 = transforms.functional.crop(y, coord_x, coord_y, patch_size, patch_size)
    return m1, m2


class CellEventClassModel(nn.Module):
    def __init__(self, TAPmodel, ClsHead):
        super().__init__()
        self._TAPmodel = TAPmodel
        self._ClsHead = ClsHead

    def forward(self, x):
        z = self._TAPmodel.embedding(x)
        y = self._ClsHead(z)
        return y


def probing_mistaken_preds(model, test_loader, device, is_true_positive, is_true_negative):
    false_positives = []
    false_negatives = []
    logits_false_pos = []
    logits_false_neg = []
    true_positives = []
    true_negatives = []
    logits_true_positives = []
    logits_true_negatives = []
    fp_coords = []
    fn_coords = []
    tp_coords = []
    tn_coords = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)

            crop_coords = batch[3]
            if isinstance(crop_coords, torch.Tensor):
                crop_coords = crop_coords[0]
            t_idx = int(crop_coords[2].item())
            coord = (int(crop_coords[0].item()), int(crop_coords[1].item()), t_idx)

            if (predicted == 1) & (y == 0):
                false_positives.append(batch)
                logits_false_pos.append(torch.squeeze(outputs.detach().cpu()))
                fp_coords.append(coord)
            elif (predicted == 0) & (y == 1):
                false_negatives.append(batch)
                logits_false_neg.append(torch.squeeze(outputs.detach().cpu()))
                fn_coords.append(coord)
            if is_true_positive and (predicted == 1) & (y == 1):
                true_positives.append(batch)
                logits_true_positives.append(torch.squeeze(outputs.detach().cpu()))
                tp_coords.append(coord)
            if is_true_negative and (predicted == 0) & (y == 0):
                true_negatives.append(batch)
                logits_true_negatives.append(torch.squeeze(outputs.detach().cpu()))
                tn_coords.append(coord)

    print(
        f"number of false_positives predictions: {len(fp_coords)}\n"
        f"number of false_negatives predictions: {len(fn_coords)}\n"
        f"number of true positive predictions: {len(tp_coords)}"
    )

    return (
        fp_coords,
        fn_coords,
        false_positives,
        false_negatives,
        logits_false_pos,
        logits_false_neg,
        tp_coords,
        true_positives,
        logits_true_positives,
        tn_coords,
        true_negatives,
        logits_true_negatives,
    )


def count_data_points(dataloader):
    count = 0
    num_pos = 0
    for batch in dataloader:
        inputs, labels = unpack_batch(batch)
        count += inputs.size(0)
        num_pos += (labels == 1).sum().item()
    return count, num_pos


def save_output_as_txt(data, output_f_path):
    with open(output_f_path, "w") as f:
        for item in data:
            if isinstance(item, (list, tuple)):
                vals = []
                for e in item:
                    if isinstance(e, torch.Tensor):
                        if e.numel() == 1:
                            vals.append(str(e.item()))
                        else:
                            vals.extend(str(x) for x in e.view(-1).tolist())
                    else:
                        vals.append(str(e))
                line = ",".join(vals)
            elif isinstance(item, torch.Tensor):
                if item.numel() == 1:
                    line = str(item.item())
                else:
                    line = ",".join(str(x) for x in item.view(-1).tolist())
            else:
                line = str(item)
            f.write(line + "\n")
    print(f"Successfully saved to {output_f_path}")


def plot_mistaken_examples(num_egs_to_show, total_number, example_set, coords, imgs_masks_seq, image_outdir, patch_size):
    for i in range(min(num_egs_to_show, total_number)):
        x_crop, _, _, crop_coords = example_set[i][0]
        image1 = x_crop[0]
        image2 = x_crop[1]
        coord_x = int(crop_coords[0].item())
        coord_y = int(crop_coords[1].item())
        time_index = int(crop_coords[2].item())
        time_arrow_label = int(crop_coords[3].item())
        mask_1, mask_2 = masks_lookup(coord_x, coord_y, patch_size, time_index, imgs_masks_seq)
        save_path = os.path.join(image_outdir, f"example_{i}.pdf")
        if time_arrow_label == 0:
            plot_images_gray_scale(image1, image2, mask_1, mask_2, save_path)
        else:
            plot_images_gray_scale(image2, image1, mask_2, mask_1, save_path)
    print("Example images saved")


def _collect_time_index_stats(test_dataset, fp_coords, fn_coords):
    gt_pos = {}
    gt_neg = {}
    total = {}
    for item in test_dataset:
        event_label = int(item[1])
        crop_coordinates = item[3]
        t_idx = int(crop_coordinates[2])
        total[t_idx] = total.get(t_idx, 0) + 1
        if event_label == 1:
            gt_pos[t_idx] = gt_pos.get(t_idx, 0) + 1
        else:
            gt_neg[t_idx] = gt_neg.get(t_idx, 0) + 1

    fp = {}
    for c in fp_coords:
        t_idx = int(c[2])
        fp[t_idx] = fp.get(t_idx, 0) + 1

    fn = {}
    for c in fn_coords:
        t_idx = int(c[2])
        fn[t_idx] = fn.get(t_idx, 0) + 1

    if total:
        t_min, t_max = min(total.keys()), max(total.keys())
    else:
        t_min, t_max = 0, 0
    T = np.arange(t_min, t_max + 1, dtype=int)

    def arr(d):
        return np.array([d.get(t, 0) for t in T], dtype=float)

    total_arr = arr(total)
    gt_pos_arr = arr(gt_pos)
    gt_neg_arr = arr(gt_neg)
    fp_arr = arr(fp)
    fn_arr = arr(fn)

    with np.errstate(divide="ignore", invalid="ignore"):
        fp_pct = np.where(gt_pos_arr > 0, 100.0 * fp_arr / gt_pos_arr, 0.0)
        fn_pct = np.where(gt_neg_arr > 0, 100.0 * fn_arr / gt_neg_arr, 0.0)
        gtp_pct = np.where(total_arr > 0, 100.0 * gt_pos_arr / total_arr, 0.0)
        gtn_pct = np.where(total_arr > 0, 100.0 * gt_neg_arr / total_arr, 0.0)

    return T, fp_pct, fn_pct, gtp_pct, gtn_pct


def _plot_fp_fn_curves(T, fp_pct, fn_pct, gtp_pct, gtn_pct, outpath_pdf):
    from matplotlib import rcParams

    rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.titleweight": "bold",
            "axes.labelweight": "regular",
            "axes.edgecolor": "#d1d5db",
            "axes.linewidth": 1.0,
        }
    )
    plt.style.use("default")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=150)

    axes[0].plot(T, fp_pct, label="False Positives (%)", linewidth=2.0)
    axes[0].plot(T, gtp_pct, label="Ground Truth Positives (%)", linewidth=2.0)
    axes[0].set_title("False Positives")
    axes[0].set_xlabel("Time Index of Frame")
    axes[0].set_ylabel("Percentage (%)")
    if len(T):
        ymin0 = min(fp_pct.min(), gtp_pct.min())
        ymax0 = max(fp_pct.max(), gtp_pct.max())
    else:
        ymin0, ymax0 = 0.0, 1.0
    axes[0].set_ylim(ymin0 - 1.0, ymax0 + 1.0)
    axes[0].grid(axis="y", alpha=0.15)
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(T, fn_pct, label="False Negatives (%)", linewidth=2.0)
    axes[1].plot(T, gtn_pct, label="Ground Truth Negatives (%)", linewidth=2.0)
    axes[1].set_title("False Negatives")
    axes[1].set_xlabel("Time Index of Frame")
    if len(T):
        ymin1 = min(fn_pct.min(), gtn_pct.min())
        ymax1 = max(fn_pct.max(), gtn_pct.max())
    else:
        ymin1, ymax1 = 0.0, 1.0
    axes[1].set_ylim(ymin1 - 1.0, ymax1 + 1.0)
    axes[1].grid(axis="y", alpha=0.15)
    axes[1].legend(frameon=False, loc="upper right")

    os.makedirs(os.path.dirname(outpath_pdf), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath_pdf, bbox_inches="tight", dpi=600)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mistake_pred_dir", default="results/mistaken_predictions")
    parser.add_argument("--masks_path", required=True)
    parser.add_argument("--num_egs_to_show", type=int, default=10)
    parser.add_argument("--TAP_model_load_path", type=str)
    parser.add_argument("--patch_size", type=int, default=48)
    parser.add_argument("--test_data_load_path", type=str)
    parser.add_argument("--combined_model_load_dir", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--is_true_positive", action="store_true")
    parser.add_argument("--is_true_negative", action="store_true")
    parser.add_argument("--cls_head_arch", type=str)
    parser.add_argument("--save_data", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    if args.device is None or str(args.device).lower() in ("", "auto"):
        if torch.cuda.is_available():
            dev = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        d = str(args.device).lower()
        if d == "cuda" and torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = args.device
    device = torch.device(dev)
    print(f"Using device: {device}")

    if not args.TAP_model_load_path or not os.path.isdir(args.TAP_model_load_path):
        print(f"TAP model directory not found: {args.TAP_model_load_path}")
        return

    try:
        TAPmodel = load_tap_model_robust(args.TAP_model_load_path, device=device)
    except FileNotFoundError as e:
        print(f"Could not load TAP model: {e}")
        return

    if args.cls_head_arch == "linear":
        cls_head = LinearHead(num_classes=2).to(device)
    elif args.cls_head_arch == "minimal":
        cls_head = MinimalHead(num_classes=2).to(device)
    elif args.cls_head_arch == "resnet":
        cls_head = ResNetHead(num_classes=2).to(device)
    else:
        print(f"Unknown cls_head_arch: {args.cls_head_arch}")
        return

    model = CellEventClassModel(TAPmodel=TAPmodel, ClsHead=cls_head)

    if not args.combined_model_load_dir or not args.model_id:
        print("Missing combined_model_load_dir or model_id")
        return

    combined_ckpt = os.path.join(args.combined_model_load_dir, f"{args.model_id}.pth")
    if not os.path.isfile(combined_ckpt):
        print(f"Combined model checkpoint not found: {combined_ckpt}")
        return

    state = torch.load(combined_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict(strict=False) missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    if not args.test_data_load_path or not os.path.isfile(args.test_data_load_path):
        print(f"Test data crops file not found: {args.test_data_load_path}")
        return

    test_data_crops_flat = torch.load(args.test_data_load_path, map_location="cpu")

    test_loader = DataLoader(
        test_data_crops_flat,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        persistent_workers=False,
    )
    print(f"number of data points in test_loader, : {count_data_points(test_loader)}")

    (
        false_positive_coordinates,
        false_negative_coordinates,
        false_positive_egs,
        false_negative_egs,
        logits_false_pos,
        logits_false_neg,
        true_positives_coordinates,
        true_positives_egs,
        logits_true_positives,
        true_negatives_coordinates,
        true_negatives_egs,
        logits_true_negatives,
    ) = probing_mistaken_preds(
        model,
        test_loader,
        device,
        is_true_positive=args.is_true_positive,
        is_true_negative=args.is_true_negative,
    )

    base_dir = args.outdir if args.outdir else args.mistake_pred_dir
    mistake_pred_model_id_dir = os.path.join(base_dir, args.model_id or "model")
    os.makedirs(mistake_pred_model_id_dir, exist_ok=True)

    save_output_as_txt(
        false_positive_coordinates,
        os.path.join(mistake_pred_model_id_dir, "false_positives_coordinates.txt"),
    )
    save_output_as_txt(
        false_negative_coordinates,
        os.path.join(mistake_pred_model_id_dir, "false_negatives_coordinates.txt"),
    )
    save_output_as_txt(
        logits_false_neg,
        os.path.join(mistake_pred_model_id_dir, "false_negatives_logits.txt"),
    )
    save_output_as_txt(
        logits_false_pos,
        os.path.join(mistake_pred_model_id_dir, "false_positives_logits.txt"),
    )

    if args.save_data:
        torch.save(
            false_positive_egs,
            os.path.join(mistake_pred_model_id_dir, "false_positives_egs.pth"),
        )
        torch.save(
            false_negative_egs,
            os.path.join(mistake_pred_model_id_dir, "false_negatives_egs.pth"),
        )

    print(f"false positive and false negatives examples saved to {mistake_pred_model_id_dir}")

    imgs_masks_sequences = masks_prep(n_frames=2, masks_path=args.masks_path, delta_frames=1)

    fp_img_dir = os.path.join(mistake_pred_model_id_dir, "false_positive_image_examples")
    fn_img_dir = os.path.join(mistake_pred_model_id_dir, "false_negative_image_examples")
    os.makedirs(fp_img_dir, exist_ok=True)
    os.makedirs(fn_img_dir, exist_ok=True)

    print("plotting false positive examples")
    plot_mistaken_examples(
        args.num_egs_to_show,
        len(false_positive_egs),
        false_positive_egs,
        false_positive_coordinates,
        imgs_masks_sequences,
        fp_img_dir,
        args.patch_size,
    )

    print("plotting false negative examples")
    plot_mistaken_examples(
        args.num_egs_to_show,
        len(false_negative_egs),
        false_negative_egs,
        false_negative_coordinates,
        imgs_masks_sequences,
        fn_img_dir,
        args.patch_size,
    )

    if args.is_true_positive:
        save_output_as_txt(
            true_positives_coordinates,
            os.path.join(mistake_pred_model_id_dir, "true_positives_coordinates.txt"),
        )
        save_output_as_txt(
            logits_true_positives,
            os.path.join(mistake_pred_model_id_dir, "true_positives_logits.txt"),
        )
        if args.save_data:
            torch.save(
                true_positives_egs,
                os.path.join(mistake_pred_model_id_dir, "true_positives_egs.pth"),
            )
        tp_img_dir = os.path.join(mistake_pred_model_id_dir, "true_positive_image_examples")
        os.makedirs(tp_img_dir, exist_ok=True)
        print("plotting true positive examples")
        plot_mistaken_examples(
            args.num_egs_to_show,
            len(true_positives_egs),
            true_positives_egs,
            true_positives_coordinates,
            imgs_masks_sequences,
            tp_img_dir,
            args.patch_size,
        )

    if args.is_true_negative:
        save_output_as_txt(
            true_negatives_coordinates,
            os.path.join(mistake_pred_model_id_dir, "true_negatives_coordinates.txt"),
        )
        save_output_as_txt(
            logits_true_negatives,
            os.path.join(mistake_pred_model_id_dir, "true_negatives_logits.txt"),
        )
        if args.save_data:
            torch.save(
                true_negatives_egs,
                os.path.join(mistake_pred_model_id_dir, "true_negatives_egs.pth"),
            )
        tn_img_dir = os.path.join(mistake_pred_model_id_dir, "true_negative_image_examples")
        os.makedirs(tn_img_dir, exist_ok=True)
        print("plotting true negative examples")
        plot_mistaken_examples(
            args.num_egs_to_show,
            len(true_negatives_egs),
            true_negatives_egs,
            true_negatives_coordinates,
            imgs_masks_sequences,
            tn_img_dir,
            args.patch_size,
        )

    curves_dir = mistake_pred_model_id_dir
    os.makedirs(curves_dir, exist_ok=True)
    out_pdf = os.path.join(curves_dir, "fp_fn_time_curves.pdf")
    out_csv = os.path.join(curves_dir, "fp_fn_time_curves.csv")

    T, fp_pct, fn_pct, gtp_pct, gtn_pct = _collect_time_index_stats(
        test_data_crops_flat,
        false_positive_coordinates,
        false_negative_coordinates,
    )
    _plot_fp_fn_curves(T, fp_pct, fn_pct, gtp_pct, gtn_pct, out_pdf)
    print(f"Saved FP/FN temporal curves to: {out_pdf}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_index", "fp_pct", "fn_pct", "gt_pos_pct", "gt_neg_pct"])
        for i, t in enumerate(T):
            w.writerow(
                [
                    int(t),
                    float(fp_pct[i]),
                    float(fn_pct[i]),
                    float(gtp_pct[i]),
                    float(gtn_pct[i]),
                ]
            )
    print(f"Saved FP/FN temporal curves CSV: {out_csv}")


if __name__ == "__main__":
    main()
