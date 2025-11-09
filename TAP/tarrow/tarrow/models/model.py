# model.py
from collections import defaultdict
import logging
from pathlib import Path
from typing import Sequence, Union
import json
from time import time as now

import dill
import numpy as np
from scipy.ndimage import zoom
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision.utils import make_grid

import yaml
from .backbones import get_backbone
from .proj_heads import ProjectionHead
from .class_heads import ClassificationHead
from .losses import DecorrelationLoss
from ..utils import normalize, tile_iterator
from ..visualizations import create_visuals
from ..visualizations import cam_insets

from pdb import set_trace
import shutil

from pathlib import Path
import yaml
import torch
from torch import nn
import dill
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
from time import time as now
from tqdm.auto import tqdm
import json
from .backbones import get_backbone
from .proj_heads import ProjectionHead
from .class_heads import ClassificationHead
from .losses import DecorrelationLoss
from ..utils import normalize, tile_iterator
from ..visualizations import create_visuals
from ..visualizations import cam_insets
import logging

logger = logging.getLogger(__name__)

class NoOutputFolderException(Exception):
    def __init__(self) -> None:
        super().__init__("Model doesnt have an associated output folder!")

def _tensor_random_choice(
    x: torch.Tensor, n_samples: Union[int, float]
) -> torch.Tensor:
    assert x.ndim == 1

    if isinstance(n_samples, float):
        assert 0 <= n_samples <= 1.0
        n_samples = int(len(x) * n_samples)

    n_samples = min(max(1, n_samples), len(x))
    idx = np.random.randint(0, len(x), n_samples)
    return x[idx]

def _git_commit():
    import git

    try:
        return str(git.Repo(Path(__file__).resolve().parents[2]).commit())
    except git.exc.InvalidGitRepositoryError:
        return None


class TimeArrowNet(nn.Module):
    def __init__(
        self,
        backbone="unet",
        projection_head="minimal_batchnorm",
        classification_head="minimal",
        n_frames=2,
        n_features=16,
        n_input_channels=1,
        device="cpu",
        symmetric=False,
        outdir: str = None,
        commit=None,
    ):
        super().__init__()
        self.outdir = Path(outdir) if outdir is not None else None

        model_kwargs = dict(
            backbone=backbone,
            projection_head=projection_head,
            classification_head=classification_head,
            n_frames=n_frames,
            n_features=n_features,
            n_input_channels=n_input_channels,
            symmetric=symmetric,
            outdir=str(self.outdir) if self.outdir is not None else None,
            commit=commit,
        )

        self.n_features = n_features
        self.backbone, self.bb_n_feat = get_backbone(backbone, n_input=n_input_channels)

        self.projection_head = ProjectionHead(
            in_features=self.bb_n_feat,
            out_features=n_features,
            mode=projection_head,
        )

        self.classification_head = ClassificationHead(
            in_features=n_features,
            n_frames=n_frames,
            out_features=n_features,
            n_classes=n_frames,
            mode=classification_head,
            symmetric=symmetric,
        )

        self.n_frames = n_frames
        self.device = device

        self.proj_activations = None
        self.proj_gradients = None
        self.projection_head.register_forward_hook(self.get_activation)

        model_kwargs["commit"] = _git_commit() if "commit" not in model_kwargs else model_kwargs["commit"]

        if self.outdir is not None:
            self.outdir.mkdir(parents=True, exist_ok=True)
            for sub in (".", "tb", "visuals"):
                (self.outdir / sub).mkdir(exist_ok=True, parents=True)
            yaml_model_kwargs = model_kwargs.copy()
            for k, v in yaml_model_kwargs.items():
                if isinstance(v, Path):
                    yaml_model_kwargs[k] = str(v)
            with open(self.outdir / "model_kwargs.yaml", "wt") as f:
                yaml.dump(yaml_model_kwargs, f)

    @property
    def outdir(self):
        return self._outdir

    @outdir.setter
    def outdir(self, path):
        import shutil
        if path is None:
            self._outdir = None
            return
        path = Path(path)
        if path.exists():
            if any(path.iterdir()):
                print(f"⚠️ [TimeArrowNet] Removing existing model folder {path}")
                shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self._outdir = path
        for sub in (".", "tb", "visuals"):
            (self._outdir / sub).mkdir(exist_ok=True, parents=True)

    def get_activation(self, model, input, output):
        self.proj_activations = output.detach()

    def get_gradients(self, grad):
        self.proj_gradients = grad.detach()

    def forward(self, x, mode="classification"):
        s_in = x.shape
        x = x.flatten(end_dim=1)
        x = self.backbone(x)
        s_out = x.shape
        features = x.reshape(s_in[:2] + (self.bb_n_feat,) + s_out[2:])
        projections = self.projection_head(features)
        if projections.requires_grad:
            projections.register_hook(self.get_gradients)
        if mode == "classification":
            final = self.classification_head(projections)
            return final
        elif mode == "projection":
            return projections
        elif mode == "both":
            final = self.classification_head(projections)
            return final, projections
        else:
            raise ValueError(f"unknown mode {mode}")

    def gradcam(self, x, class_id=0, norm=False, all_frames=False, tile_size=(512, 512)):
        if is_training := self.training:
            self.eval()
        assert x.ndim == 4, f"{x.ndim=}"

        def _get_alpha_and_activation(_x: torch.Tensor):
            self.zero_grad()
            u = self(_x.unsqueeze(0))[0]
            u = u[class_id]
            u.backward()
            A = self.proj_activations[0].detach()
            alpha = self.proj_gradients[0].detach()
            return alpha, A

        if tile_size is None or torch.all(
            torch.as_tensor(tile_size) >= torch.as_tensor(x.shape[-2:])
        ):
            x = torch.as_tensor(x, device=self.device)
            alpha, A = _get_alpha_and_activation(x)
        else:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            shape = (x.shape[0], self.n_features) + x.shape[2:]
            alpha = torch.zeros(shape, device=self.device)
            A = torch.zeros(shape, device=self.device)
            blocksize = x.shape[:2] + tuple(
                min(t, s) for t, s in zip(tile_size, x.shape[2:])
            )
            tq = tile_iterator(
                x,
                blocksize=blocksize,
                padsize=(0, 0, min(64, tile_size[0] // 4), min(64, tile_size[1] // 4)),
                mode="reflect",
            )
            for tile, s_src, s_dest in tq:
                tile = torch.as_tensor(tile, device=self.device)
                _alpha, _A = _get_alpha_and_activation(tile)
                if _alpha.shape[-2:] != tile.shape[-2:]:
                    raise NotImplementedError(
                        "Tiled CAMs only for nets with input size == output size"
                    )
                s_src = (slice(None),) * 2 + s_src[2:]
                s_dest = (slice(None),) * 2 + s_dest[2:]
                alpha[s_src] = _alpha[s_dest]
                A[s_src] = _A[s_dest]

        alpha = torch.sum(alpha, (-1, -2))
        cam = torch.einsum("tc,tcyx->tyx", alpha, A)
        cam = torch.abs(cam)
        if all_frames:
            cam = cam.sum(0)
        else:
            cam = cam[0]
        if norm:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cam.cpu().numpy()
        factor = np.array(x.shape[-2:]) / np.array(cam.shape)
        if not np.all(factor - 1 == 0):
            cam = zoom(cam, factor, order=1)
        if is_training:
            self.train()
        return cam

    def embedding(self, x, layer=0):
        _ = self(x, mode="projection")
        n = len(self.projection_head.features)
        if n <= layer:
            raise ValueError(
                f"{n} available feature layers. Embedding for layer {layer} not available."
            )
        features = self.projection_head.features[layer]
        features = features.reshape(x.shape[:2] + features.shape[1:])
        return features

    def save(self, prefix="model", which="both", exist_ok: bool = False, outdir=None):
        if outdir is None:
            outdir = self.outdir
        outdir.mkdir(exist_ok=True, parents=True)
        if which == "both" or which == "full":
            fpath = outdir / f"{prefix}_full.pt"
            if not exist_ok and fpath.exists():
                raise FileExistsError(fpath)
            torch.save(self, fpath, pickle_module=dill)
        if which == "both" or which == "state":
            fpath = outdir / f"{prefix}_state.pt"
            if not exist_ok and fpath.exists():
                raise FileExistsError(fpath)
            torch.save(self.state_dict(), fpath)

    @classmethod
    def from_folder(
        cls,
        model_folder,
        from_state_dict=False,
        map_location="cpu",
        ignore_commit=True,
        state_dict_path="model_state.pt",
    ):
        model_folder = Path(model_folder)
        logging.info(f"Loading model from {model_folder}")
        kwargs = yaml.safe_load(open(model_folder / "model_kwargs.yaml", "rt"))
        # ====== FIX: Robust to empty YAML ======
        if kwargs is None:
            kwargs = {}

        if not ignore_commit:
            _commit = _git_commit()
            if "commit" in kwargs and kwargs["commit"] != _commit:
                raise RuntimeError(
                    f"Git commit of saved model ({kwargs['commit']}) does not match current commit of tarrow repo ({_commit}). Set `ignore_commit` parameter to `True` to proceed."
                )
        if "commit" in kwargs:
            del kwargs["commit"]

        if from_state_dict:
            kwargs["device"] = map_location
            state_dict = torch.load(
                model_folder / state_dict_path, map_location=map_location
            )
            logging.info(f"Loading state dict {state_dict_path}")
            kwargs["outdir"] = None
            model = TimeArrowNet(**kwargs)
            model.load_state_dict(state_dict)
            model.to(map_location)
        else:
            model = torch.load(
                model_folder / "model_full.pt", map_location=map_location
            )
            model.device = torch.device(map_location)
            model.outdir = None
        return model

    def save_example_images(self, loader, tb_writer, phase):
        logger.info("Write example images to tensorboard.")
        example_imgs, _ = next(iter(loader))
        logger.debug(f"{example_imgs.shape=}")

        def write_to_tb(imgs, name):
            imgs = imgs[:, :, 0, ...]
            for i in range(imgs.shape[1]):
                tb_writer.add_image(
                    name,
                    make_grid(
                        imgs[:64, i : i + 1, ...],
                        scale_each=True,
                        value_range=(0, 1),
                        padding=0,
                    ),
                    global_step=i,
                )

        write_to_tb(example_imgs, f"example_images/{phase}")

    def fit(
        self,
        loader_train,
        loader_val,
        lr,
        lr_patience,
        epochs,
        steps_per_epoch,
        lr_scheduler="plateau",
        visual_datasets=(),
        visual_dataset_frequency=10,
        tensorboard=True,
        save_checkpoint_every=100,
        weight_decay=1e-6,
        lambda_decorrelation=0.01,
    ):
        assert self.outdir is not None

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        if lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.2, patience=lr_patience, verbose=True
            )
        elif lr_scheduler == "cyclic":
            # --- tiny-dataset guard to avoid ZeroDivisionError in PyTorch ---
            steps_per_epoch_safe = max(1, int(steps_per_epoch) if steps_per_epoch is not None else 0)
            total_steps = steps_per_epoch_safe * max(1, int(epochs))
            can_use_cyclic = steps_per_epoch_safe >= 2 and total_steps >= 2

            if can_use_cyclic:
                step_size_up = max(1, steps_per_epoch_safe // 2)
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=lr * 0.1,
                    max_lr=lr,
                    step_size_up=step_size_up,
                    cycle_momentum=False,
                    verbose=False,
                )
            else:
                # fall back to a no-op scheduler to keep code paths consistent
                scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=1.0, total_iters=1
                )
        else:
            raise ValueError()

        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        criterion_decorr = DecorrelationLoss()

        def _save_visuals(
            dataset: Sequence[Dataset], tb_writer, epoch: int, save_features=False
        ):
            n_insets = 8
            inset_size = 48

            for i, data in enumerate(dataset):
                vis = create_visuals(
                    dataset=data,
                    model=self,
                    device=self.device,
                    max_height=480,
                    outdir=None,
                    return_feats=save_features,
                )
                if tb_writer is not None:
                    for j, (raw_with_time, cam) in tqdm(
                        enumerate(zip(vis.raw_with_time, vis.cam)),
                        desc="Write insets",
                        total=len(vis.raw_with_time),
                        leave=True,
                    ):
                        fig, _, _ = cam_insets(
                            xs=raw_with_time,
                            cam=cam,
                            n_insets=n_insets,
                            w_inset=inset_size,
                            main_frame=0,
                        )
                        tb_writer["cams"].add_figure(
                            f"dataset_{i}/{j}", fig, global_step=epoch
                        )

                    if save_features:
                        for j, feats in tqdm(
                            enumerate(vis.feats),
                            desc="Write features",
                            total=len(vis.feats),
                            leave=True,
                        ):
                            for k, feat in enumerate(feats):
                                tb_writer["features"].add_image(
                                    f"features_{i}/{j}",
                                    normalize(feat[None], clip=True),
                                    global_step=k,
                                )

        def _model_step(loader, phase="train", title="Training"):
            start = now()
            if phase == "train":
                self.train()
            else:
                self.eval()

            losses, losses_decorr, accs = 0.0, 0.0, 0.0
            count, sum_preds = 0, 0

            with torch.set_grad_enabled(phase == "train"):
                pbar = tqdm(loader, leave=False)

                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)

                    if phase == "train":
                        optimizer.zero_grad()

                    out, pro = self(x, mode="both")

                    if out.ndim > 2:
                        y = torch.broadcast_to(
                            y.unsqueeze(1).unsqueeze(1), (y.shape[0],) + out.shape[-2:]
                        )
                        loss = criterion(out, y)
                        loss = torch.mean(loss, tuple(range(1, loss.ndim)))
                        y = y[:, 0, 0]
                        u_avg = torch.mean(out, tuple(range(2, out.ndim)))

                    else:
                        u_avg = out
                        loss = criterion(out, y)

                    pred = torch.argmax(u_avg.detach(), 1)

                    loss = torch.mean(loss)

                    pro_batched = pro.flatten(0, 1)
                    loss_decorr = criterion_decorr(pro_batched)

                    loss_all = loss + lambda_decorrelation * loss_decorr
                    if phase == "train":
                        loss_all.backward()
                        optimizer.step()
                        if lr_scheduler == "cyclic":
                            scheduler.step()

                    sum_preds += pred.sum().item()

                    count += pred.shape[0]
                    acc = torch.mean((pred == y).float())
                    losses += loss.item() * pred.shape[0]
                    losses_decorr += loss_decorr.item() * pred.shape[0]
                    accs += acc.item() * pred.shape[0]
                    pbar.set_description(
                        f"{losses/count:.6f} | {losses_decorr/count:.6f} ({phase})"
                    )

            metrics = dict(
                loss=losses / count,
                loss_decorr=losses_decorr / count,
                accuracy=accs / count,
                pred1_ratio=sum_preds / count,
                lr=scheduler.optimizer.param_groups[0]["lr"],
            )
            logger.info(
                f"{title} ({int(now() - start):3}s) Loss: {metrics['loss']:.5f} Decorr: {metrics['loss_decorr']:.5f} ACC: {metrics['accuracy']:.5f}"
            )
            return metrics

        # --- NEW: Accumulate per-epoch metrics for CSV ---
        per_epoch_metrics = []

        if tensorboard:
            tb_writer = dict(
                (key, SummaryWriter(str(self.outdir / "tb" / key)))
                for key in ("train", "val", "cams", "features")
            )
        else:
            tb_writer = None

        if tb_writer is not None and visual_dataset_frequency > 0:
            self.save_example_images(loader_train, tb_writer["train"], "train")
            self.save_example_images(loader_val, tb_writer["val"], "val")

        # --- ensure there's a loadable model before training kicks off ---
        try:
            init_full = self.outdir / "model_full.pt"
            if not init_full.exists():
                self.save(which="both", exist_ok=True)
                logger.info(f"[init] Wrote initial checkpoint: {init_full}")
        except Exception as e:
            logger.warning(f"[init] Could not write initial checkpoint: {e}")

        metrics = defaultdict(lambda: defaultdict(list))

        for i in range(1, epochs + 1):
            metrics_train = _model_step(
                loader_train,
                "train",
                f"--- Training   ({i}/{epochs})",
            )
            metrics_val = _model_step(
                loader_val, "val", f"+++ Validation ({i}/{epochs})"
            )

            if lr_scheduler == "plateau":
                scheduler.step(metrics_val["loss"])

            for loader_, phase_, metr in zip(
                (loader_train, loader_val),
                ("train", "val"),
                (metrics_train, metrics_val),
            ):
                for k, v in metr.items():
                    metrics[phase_][k].append(v)
                metrics[phase_]["steps"].append((i + 1) * len(loader_))

            # --- CSV collection ---
            per_epoch_metrics.append({
                "epoch": i,
                "train_loss": metrics_train["loss"],
                "train_loss_decorr": metrics_train["loss_decorr"],
                "train_acc": metrics_train["accuracy"],
                "train_pred1_ratio": metrics_train["pred1_ratio"],
                "val_loss": metrics_val["loss"],
                "val_loss_decorr": metrics_val["loss_decorr"],
                "val_acc": metrics_val["accuracy"],
                "val_pred1_ratio": metrics_val["pred1_ratio"],
                "lr": metrics_train["lr"],
            })

            if self.outdir is not None:
                if visual_dataset_frequency > 0 and i % visual_dataset_frequency == 0:
                    _save_visuals(visual_datasets, tb_writer, epoch=i)

            with open(self.outdir / "losses.json", "wt") as f:
                json.dump(metrics, f)

            if tb_writer is not None:
                for phase, met in metrics.items():
                    for k, v in met.items():
                        tb_writer[phase].add_scalar(k, v[-1], i)

                for tbw in tb_writer.values():
                    tbw.flush()

            checkpoint_dir = self.outdir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True, parents=True)

            if i % save_checkpoint_every == 0:
                logger.info(f"Saving checkpoint: epoch = {i}")
                self.save(
                    prefix=f"epoch_{i:05d}",
                    which="state",
                    outdir=checkpoint_dir,
                )

            target = [m["val_loss"] for m in per_epoch_metrics]
            if np.argmin(target) + 1 == i:
                logger.info(f"Saving best model: epoch = {i} val_loss = {target[-1]}")
                self.save(which="state", exist_ok=True)

        # --- Write per-epoch CSV at end of training ---
        import csv
        csv_path = self.outdir / "metrics.csv"
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "epoch",
                "train_loss", "train_loss_decorr", "train_acc", "train_pred1_ratio",
                "val_loss", "val_loss_decorr", "val_acc", "val_pred1_ratio",
                "lr",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_epoch_metrics:
                writer.writerow(row)
        logger.info(f"Saved per-epoch metrics to {csv_path}")

        if tb_writer is not None:
            _save_visuals(visual_datasets, tb_writer, epoch=i, save_features=True)
            for tbw in tb_writer.values():
                tbw.close()

        # The essential fix: return per-epoch metrics!
        return per_epoch_metrics

    def validate(self,
                 loader_val,
                 visual_datasets,
                 visual_dataset_frequency,
                 tensorboard
                 ):
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        criterion_decorr = DecorrelationLoss()

        def _save_visuals(
                dataset: Sequence[Dataset], tb_writer, epoch: int, save_features=False
        ):
            n_insets = 8
            inset_size = 24

            for i, data in enumerate(dataset):
                vis = create_visuals(
                    dataset=data,
                    model=self,
                    device=self.device,
                    max_height=480,
                    outdir=None,
                    return_feats=save_features,
                )
                if tb_writer is not None:
                    for j, (raw_with_time, cam) in tqdm(
                            enumerate(zip(vis.raw_with_time, vis.cam)),
                            desc="Write insets",
                            total=len(vis.raw_with_time),
                            leave=True,
                    ):
                        fig, _, _ = cam_insets(
                            xs=raw_with_time,
                            cam=cam,
                            n_insets=n_insets,
                            w_inset=inset_size,
                            main_frame=0,
                        )
                        tb_writer["cams"].add_figure(
                            f"dataset_{i}/{j}", fig, global_step=epoch
                        )

                    if save_features:
                        for j, feats in tqdm(
                                enumerate(vis.feats),
                                desc="Write features",
                                total=len(vis.feats),
                                leave=True,
                        ):
                            for k, feat in enumerate(feats):
                                tb_writer["features"].add_image(
                                    f"features_{i}/{j}",
                                    normalize(feat[None], clip=True),
                                    global_step=k,
                                )
        def _model_step(loader, title="Validating"):
            start = now()
            self.eval()

            losses, losses_decorr, accs = 0.0, 0.0, 0.0
            count, sum_preds = 0, 0

            with torch.set_grad_enabled(False):
                pbar = tqdm(loader, leave=False)

                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)
                    out, pro = self(x, mode="both")

                    if out.ndim > 2:
                        y = torch.broadcast_to(
                            y.unsqueeze(1).unsqueeze(1), (y.shape[0],) + out.shape[-2:]
                        )
                        loss = criterion(out, y)
                        loss = torch.mean(loss, tuple(range(1, loss.ndim)))
                        y = y[:, 0, 0]
                        u_avg = torch.mean(out, tuple(range(2, out.ndim)))

                    else:
                        u_avg = out
                        loss = criterion(out, y)

                    pred = torch.argmax(u_avg.detach(), 1)

                    loss = torch.mean(loss)

                    pro_batched = pro.flatten(0, 1)
                    loss_decorr = criterion_decorr(pro_batched)

                    sum_preds += pred.sum().item()

                    count += pred.shape[0]
                    acc = torch.mean((pred == y).float())
                    losses += loss.item() * pred.shape[0]
                    losses_decorr += loss_decorr.item() * pred.shape[0]
                    accs += acc.item() * pred.shape[0]
                    pbar.set_description(
                        f"{losses/count:.6f} | {losses_decorr/count:.6f} (Validation)"
                    )

            metrics = dict(
                loss=losses / count,
                loss_decorr=losses_decorr / count,
                accuracy=accs / count,
                pred1_ratio=sum_preds / count
            )
            logger.info(
                f"{title} ({int(now() - start):3}s) Loss: {metrics['loss']:.5f} Decorr: {metrics['loss_decorr']:.5f} ACC: {metrics['accuracy']:.5f}"
            )
            print(metrics)
            return metrics

        if tensorboard:
            tb_writer = dict(
                (key, SummaryWriter(str(self.outdir / "tb" / key)))
                for key in ("val", "cams", "features")
            )
        else:
            tb_writer = None

        if tb_writer is not None and visual_dataset_frequency > 0:
            self.save_example_images(loader_val, tb_writer["val"], "val")

        metrics = defaultdict(lambda: defaultdict(list))

        metrics_val = _model_step(
            loader_val, "val"
        )

        for loader, phase, metr in zip(
                [loader_val],
                ["val"],
                [metrics_val],
        ):
            for k, v in metr.items():
                metrics[phase][k].append(v)
            metrics[phase]["steps"].append((0 + 1) * len(loader))

        with open(self.outdir / "losses.json", "wt") as f:
            json.dump(metrics, f)

        if tb_writer is not None:
            for phase, met in metrics.items():
                for k, v in met.items():
                    tb_writer[phase].add_scalar(k, v[-1], 0)

            for tbw in tb_writer.values():
                tbw.flush()

        checkpoint_dir = self.outdir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Saving checkpoint: epoch = {0}")
        self.save(
            prefix=f"epoch_{0:05d}",
            which="state",
            outdir=checkpoint_dir,
        )

        if tb_writer is not None:
            for tbw in tb_writer.values():
                tbw.close()
