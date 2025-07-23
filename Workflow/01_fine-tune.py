# 01_fine-tune.py

import os
import sys
from pathlib import Path
import logging
import platform
from datetime import datetime
import yaml
import git
import configargparse
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset, Dataset
import csv
import numpy as np
import matplotlib.pyplot as plt

# -- Setup paths and tarrow import --
script_dir = Path(__file__).resolve().parent
tarrow_path = (script_dir.parent / "TAP" / "tarrow").resolve()
if str(tarrow_path) not in sys.path:
    sys.path.insert(0, str(tarrow_path))

import tarrow
from tarrow.models import TimeArrowNet
from tarrow.data import TarrowDataset, get_augmenter
from tarrow.visualizations import create_visuals

# --- Logging setup ---
logging.basicConfig(
    format="%(filename)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Argument parser ---
def get_argparser():
    parser = configargparse.ArgParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
    )
    parser.add("-c", "--config", is_config_file=True, help="Path to YAML config file.")
    parser.add("--name", type=str, default=None)
    parser.add("--input_train", type=str, nargs="+", required=False)
    parser.add("--input_val", type=str, nargs="*", default=None)
    parser.add("--read_recursion_level", type=int, default=0)
    parser.add("--split_train", type=float, nargs=2, action="append", required=False)
    parser.add("--split_val", type=float, nargs="+", action="append", required=False)
    parser.add("-e", "--epochs", type=int, default=200)
    parser.add("--seed", type=int, default=42)
    parser.add("--backbone", type=str, default="unet")
    parser.add("--projhead", default="minimal_batchnorm")
    parser.add("--classhead", default="minimal")
    parser.add("--perm_equiv", type=tarrow.utils.str2bool, default=True)
    parser.add("--features", type=int, default=32)
    parser.add("--n_images", type=int, default=None)
    parser.add("-o", "--outdir", type=str, default="runs")
    parser.add("--size", type=int, default=96)
    parser.add("--cam_size", type=int, default=None)
    parser.add("--batchsize", type=int, default=128)
    parser.add("--train_samples_per_epoch", type=int, default=100000)
    parser.add("--val_samples_per_epoch", type=int, default=10000)
    parser.add("--channels", type=int, default=0)
    parser.add("--reject_background", type=tarrow.utils.str2bool, default=False)
    parser.add("--cam_subsampling", type=int, default=3)
    parser.add("--write_final_cams", type=tarrow.utils.str2bool, default=False)
    parser.add("--augment", type=int, default=5)
    parser.add("--subsample", type=int, default=1)
    parser.add("--delta", type=int, nargs="+", default=[1])
    parser.add("--frames", type=int, default=2)
    parser.add("--lr", type=float, default=1e-4)
    parser.add("--lr_scheduler", default="cyclic")
    parser.add("--lr_patience", type=int, default=50)
    parser.add("--ndim", type=int, default=2)
    parser.add("--binarize", action="store_true")
    parser.add("--decor_loss", type=float, default=0.01)
    parser.add("--save_checkpoint_every", type=int, default=25)
    parser.add("--num_workers", type=int, default=8)
    parser.add("--gpu", "-g", type=str, default="0")
    parser.add("--tensorboard", type=tarrow.utils.str2bool, default=True)
    parser.add("--visual_dataset_frequency", type=int, default=10)
    parser.add("--timestamp", action="store_true")
    parser.add("--input_mask", type=str, nargs="*", default=None)
    parser.add("--pixel_resolution", type=float, default=None)
    parser.add("--min_pixels", type=int, default=None)
    parser.add("--config_yaml", type=str, default=None)
    parser.add("--metrics_csv_list", type=str, nargs="*", default=None,
               help="List of metrics.csv files (for summary plotting only, skips training)")
    return parser

# --- Config saving functions ---
def save_full_config(args, outdir):
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        repo = git.Repo(search_parent_directories=True)
        metadata["git_commit"] = repo.head.object.hexsha
    except Exception as e:
        logger.warning(f"Could not get git commit: {e}")
        metadata["git_commit"] = "unknown"

    config_path = outdir / "train_args_full.yaml"
    try:
        with open(config_path, "w") as f:
            yaml.safe_dump({**vars(args), **metadata}, f, sort_keys=False)
        logger.info(f"Saved full config to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save full config: {e}")

def save_partial_config(args, outdir):
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    partial_config_path = outdir / "train_args.yaml"
    try:
        with open(partial_config_path, "w") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False)
        logger.info(f"Saved training args to {partial_config_path}")
    except Exception as e:
        logger.error(f"Failed to save training args: {e}")

# --- Data loading and dataset utils ---
def _get_paths_recursive(paths, level):
    input_rec = paths
    for _ in range(level):
        new_inps = []
        for i in input_rec:
            p = Path(i)
            if p.is_dir():
                children = [str(x) for x in p.iterdir() if x.is_dir() or x.suffix == ".tif"]
                new_inps.extend(children)
            elif p.suffix == ".tif":
                new_inps.append(str(p))
        input_rec = new_inps
    return input_rec

def _build_dataset(
    imgs, split, size, args, n_frames, delta_frames,
    augmenter=None, permute=True, random_crop=True, reject_background=False,
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
    sampler = (
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
    )
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batchsize,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

def _build_outdir_path(args) -> Path:
    name = ""
    if args.timestamp:
        timestamp = f'{datetime.now().strftime("%m-%d-%H-%M-%S")}'
        name = f"{timestamp}_"
    suffix = f"backbone_{args.backbone}"
    safe_name = args.name if args.name else "run"
    name = f"{name}{safe_name}_{suffix}"
    outdir = Path(args.outdir).resolve()
    outdir_path = outdir / name
    if outdir_path.exists():
        logger.info(f"Run name `{name}` already exists, prepending timestamp.")
        timestamp = f'{datetime.now().strftime("%m-%d-%H-%M-%S")}'
        name = f"{timestamp}_{name}"
        outdir_path = outdir / name
    else:
        logger.info(f"Run name `{name}`")
    return outdir_path

def _convert_to_split_pairs(lst):
    if all(isinstance(x, (tuple, list)) and len(x) == 2 for x in lst):
        return tuple(lst)
    else:
        lst = tuple(
            elem for x in lst for elem in (x if isinstance(x, (list, tuple)) else (x,))
        )
        if len(lst) % 2 == 0:
            return tuple(lst[i : i + 2] for i in range(0, len(lst), 2))
        else:
            raise ValueError(f"length of split {lst} should be even!")

def _write_cams(data_visuals, model, device):
    for i, data in enumerate(data_visuals):
        create_visuals(
            dataset=data,
            model=model,
            device=device,
            max_height=720,
            outdir=model.outdir / "visuals" / f"dataset_{i}",
        )

def save_metrics_csv(metrics, outdir):
    if not metrics:
        logger.warning("No metrics to save for CSV.")
        return
    csv_path = Path(outdir) / "metrics.csv"
    fieldnames = sorted(metrics[0].keys())
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    logger.info(f"Saved per-epoch metrics to {csv_path}")

# -- Multi-run mean±std plotting (with shadow) --
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
        import pandas as pd
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

    def plot_loss_curve(self, filename="loss_curve_shadow"):
        if not self.is_per_epoch:
            print("Skipping loss curve: no per-epoch data available.")
            return

        # === Styling ===
        c_train = "#2386E6"
        c_val   = "#FC573B"
        lw = 2
        alpha_shade = 0.16
        fs_title = 17
        fs_labels = 15

        fig, ax = plt.subplots(figsize=(9, 5))
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
        ax.set_title("Training and Validation Loss (mean ± std across runs)", fontsize=fs_title)
        ax.legend(fontsize=13)
        ax.grid(alpha=0.18)
        plt.tight_layout()
        fpath_png = os.path.join(self.outdir, f"{filename}.png")
        fpath_pdf = os.path.join(self.outdir, f"{filename}.pdf")
        plt.savefig(fpath_png, dpi=320, bbox_inches='tight')
        plt.savefig(fpath_pdf, bbox_inches='tight')
        plt.close()
        print(f"Loss curves with shaded std saved to {fpath_png} and {fpath_pdf}")

    def plot_accuracy_curve(self, filename="accuracy_curve_shadow"):
        if not self.is_per_epoch:
            print("Skipping accuracy curve: no per-epoch data available.")
            return
        if self.train_acc is None or self.val_acc is None:
            print("Accuracy columns not found in metrics, skipping accuracy plot.")
            return

        # === Styling ===
        c_train = "#2386E6"
        c_val   = "#FC573B"
        lw = 2
        alpha_shade = 0.16
        fs_title = 17
        fs_labels = 15

        fig, ax = plt.subplots(figsize=(9, 5))
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
        ax.set_title("Training and Validation Accuracy (mean ± std across runs)", fontsize=fs_title)
        ax.legend(fontsize=13)
        ax.grid(alpha=0.18)
        plt.tight_layout()
        fpath_png = os.path.join(self.outdir, f"{filename}.png")
        fpath_pdf = os.path.join(self.outdir, f"{filename}.pdf")
        plt.savefig(fpath_png, dpi=320, bbox_inches='tight')
        plt.savefig(fpath_pdf, bbox_inches='tight')
        plt.close()
        print(f"Accuracy curves with shaded std saved to {fpath_png} and {fpath_pdf}")

    def plot_all_multipanel(self, filename="curves_multipanel"):
        if not self.is_per_epoch:
            print("Skipping multipanel curves: no per-epoch data available.")
            return

        # === Styling ===
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
        fpath_pdf = os.path.join(self.outdir, f"{filename}.pdf")
        plt.savefig(fpath_png, dpi=320, bbox_inches='tight')
        plt.savefig(fpath_pdf, bbox_inches='tight')
        plt.close()
        print(f"Multipanel loss/accuracy curves saved to {fpath_png} and {fpath_pdf}")

    def plot_all(self):
        self.plot_loss_curve()
        self.plot_accuracy_curve()

# --- Main training logic ---
def main(args):
    if args.metrics_csv_list:
        # Plot summary and exit (skip training)
        plotter = TrainingCurvesPlotter(args.metrics_csv_list, args.outdir)
        plotter.plot_all()
        print("Summary plots generated. Exiting (no training run).")
        return

    if not args.input_train:
        raise ValueError("Missing required field: input_train (use CLI or YAML).")
    if not args.split_train:
        raise ValueError("Missing required field: split_train (use CLI or YAML).")
    if not args.split_val:
        raise ValueError("Missing required field: split_val (use CLI or YAML).")

    if platform.system() == "Darwin":
        args.num_workers = 0
        logger.warning("Setting num_workers to 0 to avoid MacOS multiprocessing issues.")

    if args.input_val is None:
        args.input_val = args.input_train

    args.split_train = _convert_to_split_pairs(args.split_train)
    args.split_val = _convert_to_split_pairs(args.split_val)

    outdir = _build_outdir_path(args)
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(exist_ok=True)

    for p in args.input_train:
        if not Path(p).exists():
            raise FileNotFoundError(f"Training path not found: {p}")
    if args.input_val:
        for p in args.input_val:
            if not Path(p).exists():
                raise FileNotFoundError(f"Validation path not found: {p}")

    try:
        repo = git.Repo(Path(__file__).resolve().parents[1])
        args.tarrow_experiments_commit = str(repo.commit())
    except git.InvalidGitRepositoryError:
        pass

    tarrow.utils.seed(args.seed)
    try:
        use_gpu = (
            hasattr(args, "gpu")
            and args.gpu is not None
            and str(args.gpu).lower() not in ["none", "cpu"]
            and torch.cuda.is_available()
        )
        if use_gpu:
            device, n_gpus = tarrow.utils.set_device(args.gpu)
            if n_gpus > 1:
                logger.info(f"Using {n_gpus} GPUs with DataParallel.")
                device = torch.device(f"cuda:{args.gpu.split(',')[0]}")
            else:
                device = torch.device(f"cuda:{args.gpu}")
        else:
            logger.info("No GPU requested or available, using CPU.")
            device = torch.device("cpu")
            n_gpus = 0
    except Exception as e:
        logger.warning(f"Could not set GPU device ({e}), falling back to CPU.")
        device = torch.device("cpu")
        n_gpus = 0
    logger.info(f"Using device: {device}")

    augmenter = get_augmenter(args.augment)

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

    loader_train = _create_loader(
        data_train, args=args, num_samples=args.train_samples_per_epoch, num_workers=args.num_workers
    )
    loader_val = _create_loader(
        data_val, args=args, num_samples=args.val_samples_per_epoch, num_workers=0
    )

    logger.info(f"Training set: {len(data_train)} images")
    logger.info(f"Validation set: {len(data_val)} images")

    model_kwargs = dict(
        backbone=args.backbone,
        projection_head=args.projhead,
        classification_head=args.classhead,
        n_frames=args.frames,
        n_input_channels=args.channels if args.channels > 0 else 1,
        n_features=args.features,
        device=device,
        symmetric=args.perm_equiv,
        outdir=outdir,
    )

    model = TimeArrowNet(**model_kwargs)

    if n_gpus > 1:
        model = nn.DataParallel(model)
        logger.info("Model wrapped with DataParallel for multi-GPU training.")

    model.to(device)
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1.e6:.2f} Million")

    save_partial_config(args, outdir)
    save_full_config(args, outdir)

    assert args.ndim == 2

    metrics = None
    try:
        metrics = model.fit(
            loader_train=loader_train,
            loader_val=loader_val,
            lr=args.lr,
            lr_scheduler=args.lr_scheduler,
            lr_patience=args.lr_patience,
            epochs=args.epochs,
            steps_per_epoch=args.train_samples_per_epoch // args.batchsize,
            visual_datasets=tuple(
                Subset(d, list(range(0, len(d), 1 + (len(d) // args.cam_subsampling))))
                for d in data_visuals
            ),
            visual_dataset_frequency=args.visual_dataset_frequency,
            tensorboard=bool(args.tensorboard),
            save_checkpoint_every=args.save_checkpoint_every,
            lambda_decorrelation=args.decor_loss,
        )
        # --------------- PATCH: Add epoch column if missing ---------------
        if metrics and 'epoch' not in metrics[0]:
            for idx, row in enumerate(metrics):
                row['epoch'] = idx + 1
        # ------------------------------------------------------------------
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)

    save_metrics_csv(metrics, outdir)

    model_kwargs_serializable = model_kwargs.copy()
    model_kwargs_serializable['device'] = str(model_kwargs_serializable['device'])
    model_folder = outdir / f"{outdir.name}_backbone_{args.backbone}"
    model_folder.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_folder / "model.pth")
    else:
        torch.save(model.state_dict(), model_folder / "model.pth")

    with open(model_folder / "model_kwargs.yaml", "w") as f:
        yaml.safe_dump(model_kwargs_serializable, f)

    logger.info(f"Saved model and configuration to {model_folder}")

    try:
        csv_path = Path(outdir) / "metrics.csv"
        if csv_path.exists():
            plotter = TrainingCurvesPlotter([str(csv_path)], outdir)
            plotter.plot_all()
            plotter.plot_all_multipanel() 
    except Exception as e:
        logger.warning(f"Failed to plot training curves: {e}")

    if args.write_final_cams:
        _write_cams(data_visuals, model, device)

if __name__ == "__main__":
    parser = get_argparser()
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Unknown config fields detected (ignored): {unknown}")
    main(args)
