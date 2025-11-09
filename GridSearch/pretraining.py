#!/usr/bin/env python3
import sys, os
custom_dir = '/EDITS/TAP/tarrow/'
sys.path.insert(0, custom_dir)
sys.path.append('/EDITS/TAP/tarrow/tarrow')
print("sys.path:", sys.path)

from typing import Sequence
import logging
import platform
from pathlib import Path
from datetime import datetime
import yaml
import git
import configargparse
import time

import torch
from torch.utils.data import ConcatDataset, Subset, Dataset

import tarrow
from tarrow.models import TimeArrowNet
from tarrow.data import TarrowDataset, get_augmenter
from tarrow.visualizations import create_visuals

# ==== plotting deps for panels ====
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless servers
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# ==================================

logging.basicConfig(
    format="%(filename)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_argparser():
    p = configargparse.ArgParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
    )

    p.add(
        "-c",
        "--config",
        is_config_file=True,
        help="Config file path (other given arguments will superseed this).",
    )
    p.add("--name", type=str, default=None, help="Name of the training run.")
    p.add(
        "--input_train",
        type=str,
        nargs="+",
        required=True,
        help="Input training data. Can be 2D+time images, or directories with time sequences of 2d images.",
    )
    p.add(
        "--input_val",
        type=str,
        nargs="*",
        default=None,
        help="Same as `--input_train`. If not given, `--input_train` is used for validation.",
    )
    p.add("--read_recursion_level", type=int, default=0)
    p.add(
        "--split_train",
        type=float,
        nargs=2,
        action="append",
        required=True,
        help="Relative split of training data as (start, end).",
    )
    p.add(
        "--split_val",
        type=float,
        nargs="+",
        action="append",
        required=True,
        help="Same as `--split_val`.",
    )
    p.add("-e", "--epochs", type=int, default=200)
    p.add("--seed", type=int, default=42)
    p.add("--backbone", type=str, default="unet")
    p.add("--projhead", default="minimal_batchnorm")
    p.add("--classhead", default="minimal")
    p.add(
        "--perm_equiv",
        type=tarrow.utils.str2bool,
        default=True,
        help="Whether to use permutation equivariant prediction head.",
    )
    p.add(
        "--features",
        type=int,
        default=32,
        help="Dimesionality of the dense representations.",
    )
    p.add(
        "--n_images",
        type=int,
        default=None,
        help="Limit the number of images to use. Useful for debugging.",
    )
    p.add(
        "-o",
        "--outdir",
        type=str,
        default="results/pretraining",
        help="Save models and tensorboard here.",
    )
    p.add("--size", type=int, default=96, help="Patch size for training.")
    p.add(
        "--cam_size",
        type=int,
        default=None,
        help="Patch size for CAM visualization. If not given, full images are used.",
    )
    p.add("--batchsize", type=int, default=128)
    p.add("--train_samples_per_epoch", type=int, default=100000)
    p.add("--val_samples_per_epoch", type=int, default=10000)
    p.add(
        "--channels",
        type=int,
        default=0,
        help="Number of channels in the input images. Set to 0 for images do not have a explicit channel dimension.",
    )
    p.add(
        "--reject_background",
        type=tarrow.utils.str2bool,
        default=False,
        help="Set to `True` to heuristically reject background patches during training.",
    )
    p.add(
        "--cam_subsampling",
        type=int,
        default=3,
        help="Number of time frames with periodic CAM visualization.",
    )
    p.add(
        "--write_final_cams",
        type=tarrow.utils.str2bool,
        default=False,
        help="Write out CAMs of validation datasets after training is finished.",
    )
    p.add(
        "--augment",
        type=int,
        default=5,
        help="Level of data augmentation from 0 (no augmentation) to 5 (strong augmentation).",
    )
    p.add(
        "--subsample",
        type=int,
        default=1,
        help="Subsample the input images by this factor.",
    )
    p.add(
        "--delta",
        type=int,
        nargs="+",
        default=[1],
        help="Temporal delta(s) between input frames.",
    )
    p.add(
        "--frames",
        type=int,
        default=2,
        help="Number of input frames for each training sample.",
    )
    p.add("--lr", type=float, default=1e-4)
    p.add("--lr_scheduler", default="cyclic")
    p.add("--lr_patience", type=int, default=50)
    p.add("--ndim", type=int, default=2)
    p.add(
        "--binarize",
        action="store_true",
        help="Binarize the input images. Should only be used for images stored in integer format.",
    )
    p.add(
        "--decor_loss",
        type=float,
        default=0.01,
        help="Relative weighting of the decorrelation loss.",
    )
    p.add("--save_checkpoint_every", type=int, default=25)
    p.add("--num_workers", type=int, default=8, help="Number of CPU workers.")
    p.add(
        "--gpu",
        "-g",
        type=str,
        default="auto",
        help="GPUs to use. 'auto' picks CUDA if available, else MPS, else CPU. "
             "Can also be 'cpu', a single integer, a comma-separated list, or an interval 'a-b'.",
    )
    p.add("--tensorboard", type=tarrow.utils.str2bool, default=True)
    p.add(
        "--visual_dataset_frequency",
        type=int,
        default=10,
        help="Save attribution maps to tensorboard every n-th epoch.",
    )
    p.add(
        "--timestamp",
        action="store_true",
        help="Prepend output directory name with timestamp.",
    )

    # --- sweep / CSV plumbing (tiny & self-contained) ---
    p.add("--metrics_csv", type=str, default=None,
          help="If set, append a single row with final train/val metrics and full config to this CSV.")
    p.add("--sweep_tag", type=str, default=None,
          help="Optional label to group runs in sweeps (e.g. 'fullgrid').")

    return p


def _get_paths_recursive(paths: Sequence[str], level: int):
    input_rec = paths
    for _ in range(level):
        new_inps = []
        for pth in input_rec:
            pth = Path(pth)
            if pth.is_dir():
                children = [x for x in pth.iterdir() if x.is_dir() or x.suffix.lower() in (".tif", ".tiff")]
                new_inps.extend(children)
            if pth.suffix.lower() in (".tif", ".tiff"):
                new_inps.append(pth)
        input_rec = new_inps
    return input_rec


def _build_dataset(
    imgs,
    split,
    size,
    args,
    n_frames,
    delta_frames,
    augmenter=None,
    permute=True,
    random_crop=True,
    reject_background=False,
):
    return TarrowDataset(
        imgs=imgs,
        split_start=split[0],
        split_end=split[1],
        n_images=args.n_images,
        n_frames=n_frames,
        delta_frames=delta_frames,
        subsample=args.subsample,
        size=size,
        mode="flip",
        permute=permute,
        augmenter=augmenter,
        device="cpu",
        channels=args.channels,
        binarize=args.binarize,
        random_crop=random_crop,
        reject_background=reject_background,
    )


def _subset(data: Dataset, split=(0, 1.0)):
    low, high = int(len(data) * split[0]), int(len(data) * split[1])
    return Subset(data, range(low, high))


def _create_loader(dataset, args, num_samples, num_workers, idx=None, sequential=False):
    return torch.utils.data.DataLoader(
        dataset,
        sampler=(
            torch.utils.data.SequentialSampler(
                torch.utils.data.Subset(
                    dataset,
                    torch.multinomial(
                        torch.ones(len(dataset)), num_samples, replacement=True
                    ),
                )
            )
            if sequential
            else torch.utils.data.RandomSampler(
                dataset, replacement=True, num_samples=num_samples
            )
        ),
        batch_size=args.batchsize,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )


def _build_outdir_path(args) -> Path:
    name = ""
    if args.timestamp:
        timestamp = f'{datetime.now().strftime("%m-%d-%H-%M-%S")}'
        name = f"{timestamp}_"
    suffix = f"backbone_{args.backbone}"
    name = f"{name}{args.name}_{suffix}"
    if (Path(args.outdir) / name).exists():
        logger.info(f"Run name `{name}` already exists, prepending timestamp.")
        timestamp = f'{datetime.now().strftime("%m-%d-%H-%M-%S")}'
        name = f"{timestamp}_{name}"
    else:
        logger.info(f"Run name `{name}`")

    return Path(args.outdir) / name


def _convert_to_split_pairs(lst):
    """converts lst to a tuple of split pairs (ensuring backwards compatibility)

    [[0, .1, .2, .5]] -> [[0, .1], [.2, .5]]
    """
    if all(isinstance(x, (tuple, list)) and len(x) == 2 for x in lst):
        return tuple(lst)
    else:
        lst = tuple(
            elem for x in lst for elem in (x if isinstance(x, (list, tuple)) else (x,))
        )
        if len(lst) % 2 == 0:
            return tuple(lst[i: i + 2] for i in range(0, len(lst), 2))
        else:
            raise ValueError(f"length of split {lst} should be even!")


def _write_cams(data_visuals, model, device):
    for i, data in enumerate(data_visuals):
        _ = create_visuals(
            dataset=data,
            model=model,
            device=device,
            max_height=720,
            outdir=model.outdir / "visuals" / f"dataset_{i}",
        )


# ==== helpers to read TB scalars and make the 3-panel plot ====
def _load_tb_scalars(events_dir, tags, smooth_window=1):
    ea = EventAccumulator(events_dir, size_guidance={'scalars': 100000})
    ea.Reload()
    out = {}
    available = set(ea.Tags().get('scalars', []))
    for tag in tags:
        if tag in available:
            sv = ea.Scalars(tag)
            steps = np.array([s.step for s in sv], dtype=int)
            vals = np.array([s.value for s in sv], dtype=float)
            if smooth_window and smooth_window > 1 and len(vals) >= smooth_window:
                k = smooth_window
                vals = np.convolve(vals, np.ones(k) / k, mode='valid')
                steps = steps[k - 1:]
            out[tag] = (steps, vals)
    return out, available


def _plot_training_panels_from_tb(run_outdir, filename="styled_panel_no_shade.pdf"):
    TAG_MAP = {
        "train_loss":  ["loss/train", "train/loss", "Train/Loss"],
        "val_loss":    ["loss/val",   "val/loss",   "Val/Loss", "validation/loss"],
        "train_decor": ["decor/train", "train/decorr", "Train/DecorrelationLoss"],
        "val_decor":   ["decor/val",   "val/decorr",   "Val/DecorrelationLoss"],
        "train_acc":   ["acc/train", "train/acc", "Train/Accuracy"],
        "val_acc":     ["acc/val",   "val/acc",   "Val/Accuracy", "validation/accuracy"],
    }

    candidate_dirs = [
        str(run_outdir),
        os.path.join(run_outdir, "tensorboard"),
        os.path.join(run_outdir, "tb"),
        os.path.join(run_outdir, "events"),
    ]
    event_dir = None
    for d in candidate_dirs:
        if os.path.isdir(d) and any(f.startswith("events.out.tfevents") for f in os.listdir(d)):
            event_dir = d
            break
    if event_dir is None:
        print(f"[plots] No TensorBoard event files found under {run_outdir}. Skipping panel plot.")
        return

    needed = [a for aliases in TAG_MAP.values() for a in aliases]
    scalars, available = _load_tb_scalars(event_dir, needed)

    def pick(aliases):
        for a in aliases:
            if a in scalars:
                return a
        return None

    chosen = {k: pick(v) for k, v in TAG_MAP.items()}
    missing = [k for k, v in chosen.items() if v is None]
    if len(missing) == len(TAG_MAP):
        print(f"[plots] No expected tags found. Available: {sorted(list(available))}")
        return
    if missing:
        print(f"[plots] Warning: missing tags: {missing}. Plotting what’s available.")

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

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    lw = 2.1
    c_train = "#2386E6"
    c_val   = "#FC573B"
    c_decorr = "#43A047"
    c_train_acc = "#8B3DE9"
    c_val_acc = "#F7B801"

    def plot_pair(ax, tag_tr, tag_val, label_tr, label_val, title, ylabel, color_tr, color_val, vline_on_val_min=False):
        if tag_tr:
            st, vt = scalars[tag_tr]
            ax.plot(st, vt, color=color_tr, linewidth=lw, label=label_tr)
        if tag_val:
            sv, vv = scalars[tag_val]
            ax.plot(sv, vv, color=color_val, linewidth=lw, label=label_val)
            if vline_on_val_min and len(vv) > 0:
                idx = int(np.argmin(vv))
                ax.axvline(sv[idx], color="red", linestyle=":", linewidth=2, alpha=0.82)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.21, linewidth=0.7)
        ax.legend()

    plot_pair(
        axs[0],
        chosen["train_loss"], chosen["val_loss"],
        "Train Loss", "Val Loss",
        "Training and Validation Loss", r"Loss ($\mathcal{L}$)",
        c_train, c_val,
        vline_on_val_min=True
    )
    plot_pair(
        axs[1],
        chosen["train_decor"], chosen["val_decor"],
        "Train Decorr", "Val Decorr",
        "Decorrelation Loss", r"Loss ($\mathcal{L}_{\mathrm{Decorr}}$)",
        c_train, c_decorr
    )
    plot_pair(
        axs[2],
        chosen["train_acc"], chosen["val_acc"],
        "Train Acc", "Val Acc",
        "Accuracy", "Accuracy",
        c_train_acc, c_val_acc
    )

    plt.tight_layout()
    figdir = os.path.join(run_outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fpath = os.path.join(figdir, filename)
    plt.savefig(fpath, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"[plots] Saved: {fpath}")
# ================================================================

# --- exact TB tag extractor for this repo layout (tb/<phase>) ---
def _extract_last_phase_metrics(run_outdir: str, phase: str):
    """
    Read last scalars from <run_outdir>/tb/<phase>, where tags are plain:
    'loss', 'loss_decorr', 'accuracy'. Returns dict with those (or None).
    """
    phase_dir = os.path.join(run_outdir, "tb", phase)
    if not (os.path.isdir(phase_dir) and any(f.startswith("events.out.tfevents") for f in os.listdir(phase_dir))):
        return {}
    scalars, _ = _load_tb_scalars(phase_dir, ["loss", "loss_decorr", "accuracy"], smooth_window=1)

    def last(tag):
        if tag not in scalars:
            return None
        _, vals = scalars[tag]
        return float(vals[-1]) if len(vals) else None

    return {
        "loss": last("loss"),
        "loss_decorr": last("loss_decorr"),
        "accuracy": last("accuracy"),
    }


def main(args):
    if platform.system() == "Darwin":
        args.num_workers = 0
        logger.warning("Setting num_workers to 0 to avoid MacOS multiprocessing issues.")

    if args.input_val is None:
        args.input_val = args.input_train

    args.split_train = _convert_to_split_pairs(args.split_train)
    args.split_val = _convert_to_split_pairs(args.split_val)

    outdir = _build_outdir_path(args)

    try:
        repo = git.Repo(Path(__file__).resolve().parents[1])
        args.tarrow_experiments_commit = str(repo.commit())
    except git.InvalidGitRepositoryError:
        pass

    tarrow.utils.seed(args.seed)

    # ----- AUTODETECT DEVICE (CUDA → MPS → CPU), with safe fallback -----
    gpu_arg = str(args.gpu or "auto").strip().lower()
    if gpu_arg in ("", "auto"):
        if torch.cuda.is_available():
            gpu_arg = "0"  # first visible GPU
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            gpu_arg = "mps"
        else:
            gpu_arg = "cpu"
    elif gpu_arg in {"cpu", "none", "-1"}:
        gpu_arg = "cpu"

    # If explicitly CPU, bypass tarrow.set_device to avoid "GPUs 0 not available"
    if gpu_arg == "cpu":
        device = torch.device("cpu")
        n_gpus = 0
    else:
        device, n_gpus = tarrow.utils.set_device(gpu_arg)

    if n_gpus > 1:
        raise NotImplementedError("Multi-GPU training not implemented yet.")
    print(f"Using device: {device}")

    augmenter = get_augmenter(args.augment)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    inputs = {}
    for inp, phase in zip((args.input_train, args.input_val), ("train", "val")):
        inputs[phase] = _get_paths_recursive(inp, args.read_recursion_level)
        logger.debug(f"{phase} datasets: {inputs[phase]}")

    logger.info("Build visualisation datasets.")
    data_visuals = tuple(
        _build_dataset(
            inp,
            split=(0, 1.0),
            size=None if args.cam_size is None else (args.cam_size,) * args.ndim,
            args=args,
            n_frames=args.frames,
            delta_frames=args.delta[-1:],
            permute=False,
            random_crop=False,
        )
        for inp in set([*inputs["train"], *inputs["val"]])
    )

    print(f'inputs train : {inputs["train"]}')
    logger.info("Build training datasets.")
    data_train = ConcatDataset(
        _build_dataset(
            inp,
            split=split,
            size=(args.size,) * args.ndim,
            args=args,
            n_frames=args.frames,
            delta_frames=args.delta,
            augmenter=augmenter,
            reject_background=args.reject_background,
        )
        for split in args.split_train
        for inp in inputs["train"]
    )
    logger.info("Build validation datasets.")
    data_val = ConcatDataset(
        (
            _build_dataset(
                inp,
                split,
                size=(args.size,) * args.ndim,
                args=args,
                n_frames=args.frames,
                delta_frames=args.delta,
            )
            for split in args.split_val
            for inp in inputs["val"]
        )
    )

    loader_train = _create_loader(
        data_train,
        num_samples=args.train_samples_per_epoch,
        num_workers=args.num_workers,
        args=args,
    )

    loader_val = _create_loader(
        data_val,
        num_samples=args.val_samples_per_epoch,
        num_workers=0,
        args=args,
    )

    logger.info(f"Training set: {len(data_train)} images")
    logger.info(f"Validation set: {len(data_val)} images")

    # --- Keep original structure; only guard unsupported symmetric heads ---
    effective_classhead = args.classhead
    if bool(args.perm_equiv) and effective_classhead.lower() == "resnet":
        logger.warning("Symmetric classification head 'resnet' not available for permutation-equivariant pretraining. "
                       "Falling back to 'minimal' for Step-01 pretraining.")
        effective_classhead = "minimal"

    model_kwargs = dict(
        backbone=args.backbone,
        projection_head=args.projhead,
        classification_head=effective_classhead,
        n_frames=args.frames,
        n_input_channels=args.channels if args.channels > 0 else 1,
        n_features=args.features,
        device=device,
        symmetric=args.perm_equiv,
        outdir=outdir,
    )
    model = TimeArrowNet(**model_kwargs)
    model.to(device)

    # --- initial checkpoint (safe: state_dict container, no pickling of hooks) ---
    try:
        init_state = outdir / "model_state.pt"
        init_full  = outdir / "model_full.pt"
        torch.save(model.state_dict(), init_state)
        torch.save({"state_dict": model.state_dict(), "format": "state_dict_only"}, init_full)
        print(f"[save] Wrote initial state_dict to: {init_state}")
        print(f"[save] Wrote stub 'model_full.pt' (state_dict container) to: {init_full}")
    except Exception as e:
        print(f"[warn] Could not write initial model checkpoint(s): {e}")

    # --- dump model kwargs immediately (so folder is discoverable even if training aborts) ---
    try:
        kwargs_for_yaml = dict(
            backbone=args.backbone,
            projection_head=args.projhead,
            classification_head=effective_classhead,
            n_frames=args.frames,
            n_input_channels=args.channels if args.channels > 0 else 1,
            n_features=args.features,
            symmetric=args.perm_equiv,
        )
        with open(outdir / "model_kwargs.yaml", "w") as f:
            yaml.safe_dump(kwargs_for_yaml, f)
        print(f"[save] Wrote model kwargs to: {outdir / 'model_kwargs.yaml'}")
    except Exception as e:
        print(f"[save] Could not write model_kwargs.yaml: {e}")

    logger.info(
        f"Number of params: {sum(p.numel() for p in model.parameters())/1.e6:.2f} M"
    )

    with open(outdir / "train_args.yaml", "w") as f:
        yaml.safe_dump(vars(args), f)

    assert args.ndim == 2

    # --- avoid ZeroDivisionError in CyclicLR when train_samples_per_epoch < batchsize ---
    safe_steps_per_epoch = max(1, args.train_samples_per_epoch // args.batchsize)

    model.fit(
        loader_train=loader_train,
        loader_val=loader_val,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_patience=args.lr_patience,
        epochs=args.epochs,
        steps_per_epoch=safe_steps_per_epoch,
        visual_datasets=tuple(
            Subset(d, list(range(0, len(d), 1 + (len(d) // args.cam_subsampling))))
            for d in data_visuals
        ),
        visual_dataset_frequency=args.visual_dataset_frequency,
        tensorboard=bool(args.tensorboard),
        save_checkpoint_every=args.save_checkpoint_every,
        lambda_decorrelation=args.decor_loss,
    )

    # ===== Always save a final snapshot the way 03/04 expect =====
    try:
        ckpt_dir = outdir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        final_weights = outdir / "model_latest.pt"
        torch.save({"state_dict": model.state_dict()}, final_weights)

        alt_weights = outdir / "model.pt"
        torch.save(model.state_dict(), alt_weights)

        with open(outdir / "model_kwargs.yaml", "w") as f:
            yaml.safe_dump(kwargs_for_yaml, f)

        torch.save({"state_dict": model.state_dict()}, ckpt_dir / f"epoch_{int(args.epochs):04d}.pt")

        print(f"✅ Saved final artifacts to: {outdir}")
        print(f"   - Weights: {final_weights.name}  (and {alt_weights.name})")
        print(f"   - Kwargs : model_kwargs.yaml")
    except Exception as e:
        print(f"⚠️  Could not write final artifacts to {outdir}: {e}")

    # render 3-panel PDF from TB logs
    try:
        _plot_training_panels_from_tb(str(outdir))
    except Exception as e:
        print(f"[plots] Skipped (error while plotting): {e}")

    if args.write_final_cams:
        _write_cams(data_visuals, model, device)

    # --- Append sweep metrics row (train+val losses & accuracies) ---
    if getattr(args, "metrics_csv", None):
        import csv

        train_last = _extract_last_phase_metrics(str(outdir), "train")
        val_last   = _extract_last_phase_metrics(str(outdir), "val")

        # incremental config id (1..N) based on existing file length
        csv_path = Path(args.metrics_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        next_id = 1
        if csv_path.exists():
            try:
                with open(csv_path, "r", newline="") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if rows:
                        # subtract header
                        next_id = max(1, len(rows) - 1 + 1)
            except Exception:
                pass

        row = {
            "config_id": next_id,
            "configuration": f"config_{next_id}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "outdir": str(outdir),
            "sweep_tag": getattr(args, "sweep_tag", None),

            # metrics
            "train_loss": train_last.get("loss"),
            "train_loss_decorr": train_last.get("loss_decorr"),
            "train_accuracy": train_last.get("accuracy"),
            "val_loss": val_last.get("loss"),
            "val_loss_decorr": val_last.get("loss_decorr"),
            "val_accuracy": val_last.get("accuracy"),

            # config snapshot (epochs fixed but logged)
            "epochs": args.epochs,
            "seed": args.seed,
            "backbone": args.backbone,
            "projhead": args.projhead,
            "classhead": args.classhead,
            "perm_equiv": args.perm_equiv,
            "features": args.features,
            "n_images": args.n_images,
            "size": args.size,
            "cam_size": args.cam_size,
            "batchsize": args.batchsize,
            "train_samples_per_epoch": args.train_samples_per_epoch,
            "val_samples_per_epoch": args.val_samples_per_epoch,
            "channels": args.channels,
            "reject_background": args.reject_background,
            "cam_subsampling": args.cam_subsampling,
            "write_final_cams": args.write_final_cams,
            "augment": args.augment,
            "subsample": args.subsample,
            "delta": " ".join(map(str, args.delta)) if isinstance(args.delta, (list, tuple)) else args.delta,
            "frames": args.frames,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "lr_patience": args.lr_patience,
            "ndim": args.ndim,
            "binarize": args.binarize,
            "decor_loss": args.decor_loss,
            "save_checkpoint_every": args.save_checkpoint_every,
            "num_workers": args.num_workers,
            "gpu": args.gpu,
            "tensorboard": args.tensorboard,
            "visual_dataset_frequency": args.visual_dataset_frequency,
            "tarrow_commit": getattr(args, "tarrow_experiments_commit", None),
            "input_train": " ".join(args.input_train),
            "input_val": " ".join(args.input_val) if args.input_val else "",
            "split_train": str(args.split_train),
            "split_val": str(args.split_val),
            "name": args.name,
            "host": platform.node(),
        }

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"[sweep] Appended metrics to {csv_path} (config_id={next_id})")


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    main(args)
