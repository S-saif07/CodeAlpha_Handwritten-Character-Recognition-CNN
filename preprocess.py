"""Dataset loading and image preprocessing.

Two jobs live here:
1. Load MNIST (or EMNIST) and turn it into train / validation / test arrays.
2. Turn a photo or scan of a handwritten character into the same format the
   model was trained on: a 28x28 grayscale image, white character on black.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split

IMG_SIZE = 28          # MNIST / EMNIST images are 28 x 28 pixels
CHAR_BOX = 20          # MNIST digits fit inside a 20 x 20 box, centred in 28 x 28
SEED = 42              # one seed everywhere, so results can be reproduced

MNIST_CLASSES: List[str] = [str(i) for i in range(10)]
# EMNIST "balanced": 10 digits + 26 uppercase + 11 lowercase letters that are
# not easily confused with their uppercase form = 47 classes.
EMNIST_BALANCED_CLASSES: List[str] = list(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
)


class PreprocessingError(Exception):
    """Raised when an input image cannot be turned into model input."""


# --------------------------------------------------------------------------
# Part 1: datasets
# --------------------------------------------------------------------------
def get_class_names(dataset: str) -> List[str]:
    """Return the list of label names for a dataset ('mnist' or 'emnist')."""
    if dataset == "mnist":
        return MNIST_CLASSES
    if dataset == "emnist":
        return EMNIST_BALANCED_CLASSES
    raise ValueError(f"Unknown dataset '{dataset}'. Use 'mnist' or 'emnist'.")


def load_dataset(dataset: str = "mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download (first run only) and return raw uint8 images and integer labels.

    Returns x_train, y_train, x_test, y_test with images shaped (N, 28, 28).
    """
    if dataset == "mnist":
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test

    if dataset == "emnist":
        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "EMNIST needs tensorflow-datasets: pip install tensorflow-datasets"
            ) from exc
        ds = tfds.load("emnist/balanced", split=["train", "test"],
                       batch_size=-1, as_supervised=True)
        (x_train, y_train), (x_test, y_test) = tfds.as_numpy(ds)
        # EMNIST images are stored rotated/flipped; swapping the two image
        # axes puts them upright (always eyeball a few images to confirm).
        x_train = np.transpose(x_train[..., 0], (0, 2, 1))
        x_test = np.transpose(x_test[..., 0], (0, 2, 1))
        return x_train, y_train, x_test, y_test

    raise ValueError(f"Unknown dataset '{dataset}'. Use 'mnist' or 'emnist'.")


def split_and_normalize(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    val_size: float = 0.1,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """Create train / validation / test sets with pixel values in [0, 1].

    The validation set is carved out of the *training* data only, and the test
    set is never touched until the final evaluation (this avoids data leakage).
    """
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )

    def prep(images: np.ndarray) -> np.ndarray:
        # 0..255 integers -> 0..1 floats, plus a channel axis: (N, 28, 28, 1)
        return (images.astype("float32") / 255.0)[..., np.newaxis]

    return {
        "x_train": prep(x_tr), "y_train": y_tr,
        "x_val": prep(x_val), "y_val": y_val,
        "x_test": prep(x_test), "y_test": y_test,
    }


# --------------------------------------------------------------------------
# Part 2: custom handwritten images
# --------------------------------------------------------------------------
def load_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image file as a 2-D grayscale uint8 array."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # respect phone-camera rotation
            if img.mode in ("RGBA", "LA") or "transparency" in img.info:
                # Transparent PNGs: paint them on a white background first.
                rgba = img.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                background.alpha_composite(rgba)
                img = background
            return np.array(img.convert("L"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PreprocessingError(f"'{path.name}' is not a readable image.") from exc


def ensure_white_on_black(gray: np.ndarray) -> np.ndarray:
    """MNIST is a white character on a black background; flip if ours is not.

    We look at the image border: if it is mostly bright, the page is white
    (dark ink on light paper), so we invert the image.
    """
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return 255 - gray if np.median(border) > 127 else gray


def crop_to_character(gray: np.ndarray) -> np.ndarray:
    """Remove the background, drop tiny noise specks and crop to the character."""
    if int(gray.max()) - int(gray.min()) < 30:
        raise PreprocessingError("The image has almost no contrast - is it blank?")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu picks the threshold between "ink" and "background" automatically.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep only reasonably large blobs so dust and noise do not enlarge the crop.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        raise PreprocessingError("No handwritten character found in the image.")
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= 0.05 * areas.max()]
    clean_mask = np.isin(labels, keep)

    ys, xs = np.where(clean_mask)
    ink_only = np.where(clean_mask, gray, 0).astype(np.uint8)
    return ink_only[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def resize_and_center(char: np.ndarray) -> np.ndarray:
    """Scale the character to fit a 20x20 box, then centre it in 28x28 by its mass."""
    h, w = char.shape
    scale = CHAR_BOX / max(h, w)  # keep the aspect ratio (a "1" stays thin)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    top, left = (IMG_SIZE - new_h) // 2, (IMG_SIZE - new_w) // 2
    canvas[top: top + new_h, left: left + new_w] = small

    # MNIST digits are centred by centre of mass, so we do the same.
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        shift_x = IMG_SIZE / 2.0 - m["m10"] / m["m00"]
        shift_y = IMG_SIZE / 2.0 - m["m01"] / m["m00"]
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, matrix, (IMG_SIZE, IMG_SIZE))
    return canvas


def preprocess_custom_image(
    source: Union[str, Path, np.ndarray], return_steps: bool = False
):
    """Full pipeline: file/array -> model input of shape (1, 28, 28, 1) in [0, 1].

    With ``return_steps=True`` also returns a dict of intermediate images,
    which is handy for showing what the pipeline did.
    """
    gray = load_image(source) if not isinstance(source, np.ndarray) else source
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)

    polarity_fixed = ensure_white_on_black(gray)
    cropped = crop_to_character(polarity_fixed)
    final = resize_and_center(cropped)

    # Stretch contrast so the brightest ink pixel is 1.0, like MNIST.
    scaled = final.astype("float32")
    scaled = scaled / scaled.max() if scaled.max() > 0 else scaled
    model_input = scaled[np.newaxis, ..., np.newaxis]

    if return_steps:
        return model_input, {"gray": gray, "polarity_fixed": polarity_fixed,
                             "cropped": cropped, "final_28x28": final}
    return model_input
