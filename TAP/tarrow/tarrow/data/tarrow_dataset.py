import logging
from pathlib import Path
import bisect
from tqdm import tqdm
import tifffile
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from skimage.transform import downscale_local_mean
import skimage
from ..utils import normalize as utils_normalize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TarrowDataset(Dataset):
    def __init__(
        self,
        imgs,
        masks=None,                 # <---- optional, same shape as imgs
        split_start=0,
        split_end=1,
        n_images=None,
        n_frames=2,
        delta_frames=[1],
        subsample=1,
        size=None,
        mode="flip",
        permute=True,
        augmenter=None,
        normalize=None,
        channels=0,
        device="cpu",
        binarize=False,
        random_crop=True,
        reject_background=False,
        event_crop_prob=0.5,        # <---- probability for event crop
    ):
        super().__init__()
        self._split_start = split_start
        self._split_end = split_end
        self._n_images = n_images
        self._n_frames = n_frames
        self._delta_frames = delta_frames
        self._subsample = subsample

        assert mode in ["flip", "roll"]
        self._mode = mode

        self._permute = permute
        self._channels = channels
        self._device = device
        self._augmenter = augmenter
        if self._augmenter is not None:
            self._augmenter.to(device)

        # ----- Load images -----
        if isinstance(imgs, (str, Path)):
            imgs = self._load(
                path=imgs,
                split_start=split_start,
                split_end=split_end,
                n_images=n_images,
            )
        elif isinstance(imgs, (tuple, list, np.ndarray)) and isinstance(imgs[0], np.ndarray):
            imgs = np.asarray(imgs)[:n_images]
        else:
            raise ValueError(
                f"Cannot form a dataset from {imgs}. "
                "Input should be either a path to a sequence of 2d images, a single 2d+time image, or a list of 2d np.ndarrays."
            )

        if self._channels == 0:
            imgs = np.expand_dims(imgs, 1)
        else:
            imgs = imgs[:, : self._channels, ...]
        assert imgs.shape[1] == 1

        if binarize:
            imgs = (imgs > 0).astype(np.float32)
        else:
            if normalize is None:
                imgs = self._default_normalize(imgs)
            else:
                imgs = normalize(imgs)

        # --- subsample BEFORE converting to torch (skimage expects numpy) ---
        if not isinstance(self._subsample, int) or self._subsample < 1:
            raise NotImplementedError("Spatial subsampling only for positive integers.")
        if self._subsample > 1:
            # imgs shape: (T, C, H, W)
            T, C, H, W = imgs.shape
            # downscale_local_mean only supports spatial dims; do it per (T,C) slice
            out_H = H // self._subsample if H % self._subsample == 0 else H // self._subsample
            out_W = W // self._subsample if W % self._subsample == 0 else W // self._subsample
            ds = np.empty((T, C, out_H, out_W), dtype=np.float32)
            for t in range(T):
                for c in range(C):
                    ds[t, c] = downscale_local_mean(
                        imgs[t, c], (self._subsample, self._subsample)
                    )
            imgs = ds

        # convert to torch after subsampling
        imgs = torch.as_tensor(imgs)

        # --- Handle masks (optional) ---
        self._use_masks = masks is not None
        if self._use_masks:
            if isinstance(masks, (str, Path)):
                masks = self._load(
                    path=masks,
                    split_start=split_start,
                    split_end=split_end,
                    n_images=n_images,
                )
            elif isinstance(masks, (tuple, list, np.ndarray)) and isinstance(masks[0], np.ndarray):
                masks = np.asarray(masks)[:n_images]
            masks = (masks > 0).astype(np.uint8)   # Ensure binary
            if masks.ndim == 3:
                masks = masks[:, None, ...]  # add channel
            self.masks = torch.as_tensor(masks)
        else:
            self.masks = None

        # output crop size (cap to image size)
        if size is None:
            self._size = imgs[0, 0].shape
        else:
            self._size = tuple(min(a, b) for a, b in zip(size, imgs[0, 0].shape))

        # crop transform selector (we’ll handle params robustly in __getitem__)
        if random_crop:
            if reject_background:
                # keep original call; assumed defined elsewhere in codebase
                self._crop = self._reject_background()
            else:
                self._crop = transforms.RandomCrop(
                    self._size,
                    padding_mode="reflect",
                    pad_if_needed=True,
                )
        else:
            self._crop = transforms.CenterCrop(self._size)

        # ----------- Precompute time slices -----------
        self._imgs_sequences = []
        self._masks_sequences = [] if self._use_masks else None
        for delta in self._delta_frames:
            n, k = self._n_frames, delta
            tslices = tuple(
                slice(i, i + k * (n - 1) + 1, k)
                for i in range(len(imgs) - (n - 1) * k)
            )
            self._imgs_sequences.extend([imgs[ss] for ss in tslices])
            if self._use_masks:
                self._masks_sequences.extend([self.masks[ss] for ss in tslices])

        self._crops_per_image = max(
            1, int(np.prod(imgs.shape[1:3]) / np.prod(self._size))
        )
        self._event_crop_prob = event_crop_prob

    # ------------ Loader ------------
    def _load(self, path, split_start=0, split_end=1, n_images=None):
        """Load images from tif, npy, or image sequence folder"""
        path = Path(path)
        logger.info(f"Loading image(s) from {path}")
        # Support tiff stacks, npy, or image folder
        if path.suffix in [".tif", ".tiff"]:
            imgs = tifffile.imread(str(path))
        elif path.suffix in [".npy"]:
            imgs = np.load(str(path))
        elif path.is_dir():
            img_files = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg")) + sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
            imgs = np.stack([imageio.imread(str(f)) for f in img_files], axis=0)
        else:
            raise ValueError(f"Unknown file type or path: {path}")

        # Handle splitting
        total = imgs.shape[0]
        start = int(split_start * total)
        end = int(split_end * total)
        imgs = imgs[start:end]
        if n_images is not None:
            imgs = imgs[:n_images]
        return imgs

    def _default_normalize(self, imgs):
        """Default normalization using min-max scaling per-image."""
        imgs = imgs.astype(np.float32)
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
        return imgs

    def __len__(self):
        return len(self._imgs_sequences)

    def __getitem__(self, idx):
        import random
        x = self._imgs_sequences[idx]          # (T, C, H, W)
        mask = self._masks_sequences[idx] if self._use_masks else None

        # dimensions
        H, W = x.shape[-2], x.shape[-1]
        th, tw = self._size

        # -------- Event-based cropping (if masks given and sampled) --------
        do_event_crop = self._use_masks and (random.random() < self._event_crop_prob)
        if do_event_crop:
            # pick a random event pixel (use first frame’s mask for localization)
            event_pixels = torch.nonzero(mask[0] > 0, as_tuple=False)
            if len(event_pixels) > 0:
                center_y, center_x = event_pixels[random.randint(0, len(event_pixels)-1)].tolist()
                top = max(0, min(center_y - th // 2, H - th))
                left = max(0, min(center_x - tw // 2, W - tw))
                x_crop = x[..., top:top+th, left:left+tw]
                mask_crop = mask[..., top:top+th, left:left+tw] if self._use_masks else None
            else:
                # fallback to normal crop (handled below)
                do_event_crop = False

        # -------- Normal cropping path (random vs center) --------
        if not do_event_crop:
            if hasattr(self._crop, "get_params"):
                # RandomCrop path: sample params from a single frame (C,H,W),
                # then apply same crop to all time frames/channels.
                i, j, h, w = self._crop.get_params(x[0], output_size=(th, tw))
                x_crop = x[..., i:i+h, j:j+w]
                mask_crop = mask[..., i:i+h, j:j+w] if self._use_masks else None
            else:
                # CenterCrop path: CenterCrop has no get_params — compute indices directly.
                i = max(0, (H - th) // 2)
                j = max(0, (W - tw) // 2)
                x_crop = x[..., i:i+th, j:j+tw]
                mask_crop = mask[..., i:i+th, j:j+tw] if self._use_masks else None

        # -------- Temporal permutation / label construction --------
        if self._permute:
            if self._mode == "flip":
                label = torch.randint(0, 2, (1,))[0]
                if label == 1:
                    x_crop = torch.flip(x_crop, dims=(0,))
            elif self._mode == "roll":
                label = torch.randint(0, self._n_frames, (1,))[0]
                x_crop = torch.roll(x_crop, label.item(), dims=(0,))
            else:
                raise ValueError()
        else:
            label = torch.tensor(0, dtype=torch.long)

        x_crop, label = x_crop.to(self._device), label.to(self._device)
        if self._augmenter is not None:
            x_crop = self._augmenter(x_crop)

        # Return mask crop if available
        return (x_crop, mask_crop, label) if self._use_masks else (x_crop, label)


class ConcatDatasetWithIndex(ConcatDataset):
    """Additionally returns index"""

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], idx
