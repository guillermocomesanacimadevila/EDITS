# 02_data_prep.py

import os
import sys
import numpy as np
from pathlib import Path
from typing import Sequence

# Make tarrow module path dynamic
script_dir = Path(__file__).resolve().parent
tarrow_root = script_dir.parent / 'TAP' / 'tarrow'
sys.path.insert(0, str(tarrow_root))
sys.path.append(str(tarrow_root / 'tarrow'))

import torch
import yaml
import platform
import git
import logging
import configargparse
from datetime import datetime
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_yaml_safe(obj):
    """Recursively convert any non-primitive object to string for YAML safety."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: make_yaml_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_yaml_safe(v) for v in obj]
    else:
        return str(obj)

def save_config_metadata(args, output_dir):
    metadata = {
        "timestamp": str(datetime.now().isoformat()),
        "python_version": str(platform.python_version()),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(getattr(torch.version, "cuda", "unknown")),
    }
    try:
        repo = git.Repo(search_parent_directories=True)
        metadata["git_commit"] = str(repo.head.object.hexsha)
    except Exception:
        metadata["git_commit"] = "unknown"
    config_path = Path(output_dir) / "preprocessing_config.yaml"
    yaml_ready = make_yaml_safe({**vars(args), **metadata})
    with open(config_path, "w") as f:
        yaml.safe_dump(yaml_ready, f)
    print(f"Saved config to {config_path}")

def _get_paths_recursive(paths: Sequence[str], level: int):
    input_rec = paths
    for i in range(level):
        new_inps = []
        for i in input_rec:
            if Path(i).is_dir():
                children = [
                    x for x in Path(i).iterdir() if x.is_dir() or x.suffix == ".tif"
                ]
                new_inps.extend(children)
            if Path(i).suffix == ".tif":
                new_inps.append(Path(i))
        input_rec = new_inps
    return input_rec

class CellEventDataset(Dataset):
    """
    Return 2d + time crops with event type labels for training cell event classification model based on the dense
    representations produced by the time arrow prediction task.
    """
    def __init__(
            self,
            imgs,
            masks,
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
            crops_per_image=1,
            min_pixels=10
    ):
        super().__init__()
        import numpy as np
        from pathlib import Path
        from skimage.transform import downscale_local_mean
        from torchvision import transforms

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
        self._crops_per_image = crops_per_image
        self._min_pixels = min_pixels

        # Read and process imgs
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
            logger.debug("Binarize images")
            imgs = (imgs > 0).astype(np.float32)
        else:
            logger.debug("Normalize images")
            if normalize is None:
                imgs = self._default_normalize(imgs)
            else:
                imgs = normalize(imgs)

        # Read and process masks
        if isinstance(masks, (str, Path)):
            masks = self._load(
                path=masks,
                split_start=split_start,
                split_end=split_end,
                n_images=n_images,
            )
        elif isinstance(masks, (tuple, list, np.ndarray)) and isinstance(masks[0], np.ndarray):
            masks = np.asarray(masks)[:n_images]
        else:
            raise ValueError(
                f"Cannot form a dataset from {masks}. "
                "Input should be either a path to a sequence of 2d images, a single 2d+time image, or a list of 2d np.ndarrays."
            )

        # --- FIX FOR RGB MASKS ---
        if masks.ndim == 4 and masks.shape[-1] == 3:
            print(f"Converting mask from RGB (shape={masks.shape}) to grayscale by taking channel 0")
            masks = masks[..., 0]
            print(f"New mask shape: {masks.shape}")

        # --- CRITICAL BINARIZATION FIX (always binarize mask!) ---
        masks = (masks > 0).astype(np.uint8)

        imgs = torch.as_tensor(imgs)
        masks = torch.as_tensor(masks)

        print(f"subsample : {subsample}")

        if not isinstance(subsample, int) or subsample < 1:
            raise NotImplementedError(
                "Spatial subsampling only implemented for positive integer values."
            )
        if subsample > 1:
            factors = (1,) + (subsample,) * (imgs.dim() - 1)
            full_size = imgs[0].shape
            imgs = downscale_local_mean(imgs, factors)
            logger.debug(f"Subsampled from {full_size} to {imgs[0].shape}")
        if size is None:
            self._size = imgs[0, 0].shape
        else:
            self._size = tuple(min(a, b) for a, b in zip(size, imgs[0, 0].shape))

        if random_crop:
            if reject_background:
                self._crop = self._reject_background(random_seed=42)
            else:
                self._crop = transforms.RandomCrop(
                    self._size,
                    padding_mode="reflect",
                    pad_if_needed=True,
                )
        else:
            self._crop = transforms.CenterCrop(self._size)

        if imgs.ndim != 4:
            raise NotImplementedError(
                f"only 2D timelapses supported (total image shape: {imgs.shape})"
            )
        min_number = max(self._delta_frames) * (n_frames - 1) + 1
        if len(imgs) < min_number:
            raise ValueError(f"imgs should contain at least {min_number} elements")
        if len(imgs.shape[2:]) != len(self._size):
            raise ValueError(
                f"incompatible shapes between images and size last {n_frames} elements"
            )

        # Precompute the time slices, get image-mask pairs (I_t, I_{t+\delta t), M_t, M_{t+\delta t})
        self._imgs_masks_sequences = []
        for delta in self._delta_frames:
            n, k = self._n_frames, delta
            logger.debug(f"Creating delta {delta} crops")
            tslices = tuple(
                slice(i, i + k * (n - 1) + 1, k) for i in range(len(imgs) - (n - 1) * k)
            )
            imgs_masks_sequences = [(torch.as_tensor(imgs[ss]), torch.as_tensor(masks[ss])) for ss in tslices]
            self._imgs_masks_sequences.extend(imgs_masks_sequences)

    def _reject_background(self, random_seed, threshold=0.02, max_iterations=10):
        from torchvision import transforms
        torch.manual_seed(random_seed)
        rc = transforms.RandomCrop(
            self._size,
            padding_mode="reflect",
            pad_if_needed=True,
        )

        def smoother(img):
            import skimage
            import numpy as np
            img = skimage.util.img_as_ubyte(img.squeeze(1).numpy().clip(-1, 1))
            img = skimage.filters.rank.median(
                img, footprint=np.ones((self._n_frames, 3, 3))
            )
            return torch.as_tensor(skimage.util.img_as_float32(img)).unsqueeze(1)

        def crop(x):
            with torch.no_grad():
                for i in range(max_iterations):
                    out = rc(x)
                    mask = smoother(out)
                    if mask.std() > threshold:
                        return out
                return out

        return crop

    def _default_normalize(self, imgs):
        from tqdm import tqdm
        from utils import normalize as utils_normalize
        import numpy as np

        imgs_norm = []
        for img in tqdm(imgs, desc="normalizing images", leave=False):
            imgs_norm.append(utils_normalize(img, subsample=8))
        return np.stack(imgs_norm)

    def _load(self, path, split_start, split_end, n_images=None):
        from pathlib import Path
        import tifffile

        assert split_start >= 0
        assert split_end <= 1

        inp = Path(path).expanduser()

        if inp.is_dir():
            suffixes = ("png", "jpg", "tif", "tiff")
            for s in suffixes:
                fnames = sorted(Path(inp).glob(f"*.{s}"))
                if len(fnames) > 0:
                    break
            if len(fnames) == 0:
                raise ValueError(f"Could not find any images in {inp}")

            fnames = fnames[:n_images]
            imgs = self._load_image_folder(fnames, split_start, split_end)

        elif inp.suffix == ".tif" or inp.suffix == ".tiff":
            logger.info(f"Loading {inp}")
            imgs = tifffile.imread(str(inp))
            logger.info("Done")
            print(f'imags shape : {imgs.shape}')
            imgs = imgs[int(len(imgs) * split_start) : int(len(imgs) * split_end)]
            imgs = imgs[:n_images]
        else:
            raise ValueError(
                (
                    f"Cannot form a dataset from {inp}. "
                    "Input should be either a path to a sequence of 2d images, a single 2d+time image, or a list of 2d np.ndarrays."
                )
            )

        return imgs

    def _load_image_folder(
            self,
            fnames,
            split_start: float,
            split_end: float,
    ) -> np.ndarray:
        import tifffile
        from tqdm import tqdm
        from pathlib import Path
        import imageio
        idx_start = int(len(fnames) * split_start)
        idx_end = int(len(fnames) * split_end)
        fnames = fnames[idx_start:idx_end]

        logger.info(f"Load images {idx_start}-{idx_end}")
        imgs = []
        for f in tqdm(fnames, leave=False, desc="loading images"):
            f = Path(f)
            if f.suffix in (".tif", ".TIFF", ".tiff"):
                x = tifffile.imread(f)
            elif f.suffix in (".png", ".jpg", ".jpeg"):
                x = imageio.imread(f)
                if x.ndim == 3:
                    x = np.moveaxis(x[..., :3], -1, 0)
            else:
                continue
            x = np.squeeze(x)
            imgs.append(x)
        imgs = np.stack(imgs)
        return imgs

    def __len__(self):
        return len(self._imgs_masks_sequences)

    def mask_to_label(self, mask_input, binary_problem=True):
        """
        This version is robust: any nonzero pixel means "event" if there are at least min_pixels present.
        """
        if binary_problem:
            if torch.any(mask_input > 0):
                if torch.sum(mask_input).item() > self._min_pixels:
                    output_label = 1
                else:
                    output_label = 0
            else:
                output_label = 0
        else:
            if torch.any(mask_input == torch.tensor(1)):
                output_label = 1
            elif torch.any(mask_input == torch.tensor(2)):
                output_label = 2
            else:
                output_label = 0
        return torch.tensor(output_label)

    def label_image_pair(self, mask1, mask2, binary_problem=True):
        label_m1 = self.mask_to_label(mask1,  binary_problem)
        label_m2 = self.mask_to_label(mask2,  binary_problem)
        if (label_m1 + label_m2).item() >= 1:
            event_label = torch.tensor(1)
        else:
            event_label = torch.tensor(0)
        return event_label

    def count_event_labels_one_pair(self, image1, image2, tile_size):
        event_count = torch.tensor(0)
        image_h, image_w = image1.size()[0], image1.size()[1]
        from torchvision import transforms
        for i in range(0, image_h, tile_size):
            for j in range(0, image_w, tile_size):
                crop_1 = transforms.functional.crop(image1, i, j, tile_size, tile_size)
                crop_2 = transforms.functional.crop(image2, i, j, tile_size, tile_size)
                event_count += self.label_image_pair(mask1=crop_1, mask2=crop_2)
        return event_count

    def generate_one_datapoint(self, idx):
        if isinstance(idx, (list, tuple)):
            return list(self[_idx] for _idx in idx)

        x, y = self._imgs_masks_sequences[idx]
        i, j, h, w = self._crop.get_params(x, output_size=self._size)
        from torchvision import transforms
        x_crop = transforms.functional.crop(x, i, j, h, w)
        y_crop = transforms.functional.crop(y, i, j, h, w)

        event_label = self.label_image_pair(mask1=y_crop[0], mask2=y_crop[1])

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

        if self._augmenter is not None:
            x_crop = self._augmenter(x_crop)

        crop_coordinates = (torch.tensor(i), torch.tensor(j), torch.tensor(idx), label)
        crop_coordinates = tuple(t.to(self._device) for t in crop_coordinates)
        x_crop, event_label, label = x_crop.to(self._device), event_label.to(self._device), label.to(self._device)
        return x_crop, event_label, label, crop_coordinates

    def __getitem__(self, idx):
        x, y = self._imgs_masks_sequences[idx]
        total_event_count = self.count_event_labels_one_pair(y[0], y[1], tile_size=64)
        total_event_count = total_event_count.to(self._device)
        sample = [tuple(self.generate_one_datapoint(idx)) for i in range(self._crops_per_image)]
        return sample, total_event_count

def _build_dataset(
        imgs,
        masks,
        split,
        size,
        args,
        n_frames,
        delta_frames,
        augmenter=None,
        permute=True,
        random_crop=True,
        reject_background=False,
        crops_per_image=1,
        min_pixels=10
):
    return CellEventDataset(
        imgs=imgs,
        masks=masks,
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
        channels=0,
        binarize=args.binarize,
        random_crop=random_crop,
        reject_background=reject_background,
        crops_per_image=crops_per_image,
        min_pixels=min_pixels
    )

def flatten_data(input_data, crops_per_image):
    flat_data = []
    for i in range(len(input_data)):
        for j in range(crops_per_image):
            flat_data.append((input_data[i][0][j]))
    return flat_data

def get_argparser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--input_frame", type=str, required=True)
    parser.add_argument("--input_mask", type=str, required=True)
    parser.add_argument("--data_save_dir", type=str, required=True)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--min_pixels", type=int, default=10, help="Minimum number of positive pixels for a crop to be counted as an event. User-tunable for classifier.")
    parser.add_argument("--crops_per_image", type=int, default=1)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--n_images", type=int, default=None)
    return parser

def main():
    parser = get_argparser()
    args = parser.parse_args()

    if not args.input_frame:
        raise ValueError("Missing required field: input_frame (use CLI or YAML).")
    if not args.input_mask:
        raise ValueError("Missing required field: input_mask (use CLI or YAML).")
    if not args.data_save_dir:
        raise ValueError("Missing required field: data_save_dir (use CLI or YAML).")

    if not Path(args.input_frame).exists():
        raise FileNotFoundError(f"Input frame path not found: {args.input_frame}")
    if not Path(args.input_mask).exists():
        raise FileNotFoundError(f"Input mask path not found: {args.input_mask}")

    torch.manual_seed(args.data_seed)
    time_delta = [1]

    inputs_frame = [_get_paths_recursive(args.input_frame, 0)]
    inputs_mask = [_get_paths_recursive(args.input_mask, 0)]

    image_crops = ConcatDataset(
        (
            _build_dataset(
                imgs=inp,
                masks=mask,
                split=[0, 1],
                size=(args.size,) * args.ndim,
                args=args,
                n_frames=args.frames,
                delta_frames=time_delta,
                crops_per_image=args.crops_per_image,
                min_pixels=args.min_pixels
            )
            for inp, mask in zip(inputs_frame, inputs_mask)
        )
    )

    start_time = datetime.now()
    image_crops_flat = flatten_data(image_crops, args.crops_per_image)

    os.makedirs(args.data_save_dir, exist_ok=True)
    save_path = Path(args.data_save_dir) / "preprocessed_image_crops.pth"
    torch.save(image_crops_flat, save_path)
    print(f"Image crops saved to {save_path}!")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Data pre-processing completed in {elapsed:.2f} seconds.")

    save_config_metadata(args, args.data_save_dir)

if __name__ == '__main__':
    main()
