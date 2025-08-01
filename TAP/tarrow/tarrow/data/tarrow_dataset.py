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
        masks=None,                 # <---- NEW! (optional, same shape as imgs)
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
        event_crop_prob=0.5,        # <---- NEW! (probability for event crop)
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

        # ----- Loading images and masks -----
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
        imgs = torch.as_tensor(imgs)

        # --- Handle masks ---
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
            masks = torch.as_tensor(masks)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            self.masks = masks
        else:
            self.masks = None

        if not isinstance(subsample, int) or subsample < 1:
            raise NotImplementedError("Spatial subsampling only for positive integers.")
        if subsample > 1:
            factors = (1,) + (subsample,) * (imgs.dim() - 1)
            imgs = downscale_local_mean(imgs, factors)
        if size is None:
            self._size = imgs[0, 0].shape
        else:
            self._size = tuple(min(a, b) for a, b in zip(size, imgs[0, 0].shape))

        if random_crop:
            if reject_background:
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

    # ------------ ADDED _load FUNCTION ------------
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
        x = self._imgs_sequences[idx]
        mask = self._masks_sequences[idx] if self._use_masks else None

        # Event-based cropping
        do_event_crop = self._use_masks and (random.random() < self._event_crop_prob)
        if do_event_crop:
            # Pick a random event pixel
            event_pixels = torch.nonzero(mask[0] > 0, as_tuple=False)
            if len(event_pixels) > 0:
                center_y, center_x = event_pixels[random.randint(0, len(event_pixels)-1)].tolist()
                crop_h, crop_w = self._size
                top = max(0, min(center_y - crop_h // 2, x.shape[-2] - crop_h))
                left = max(0, min(center_x - crop_w // 2, x.shape[-1] - crop_w))
                x_crop = x[:, :, top:top+crop_h, left:left+crop_w]
                mask_crop = mask[:, :, top:top+crop_h, left:left+crop_w]
            else:
                # fallback to normal crop
                i, j, h, w = self._crop.get_params(x, output_size=self._size)
                x_crop = transforms.functional.crop(x, i, j, h, w)
                mask_crop = transforms.functional.crop(mask, i, j, h, w)
        else:
            i, j, h, w = self._crop.get_params(x, output_size=self._size)
            x_crop = transforms.functional.crop(x, i, j, h, w)
            mask_crop = transforms.functional.crop(mask, i, j, h, w) if self._use_masks else None

        # Data augmentation (optional)
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
