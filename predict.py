"""Predict a character from your own image.

Usage (from the project root):
    python -m src.predict --image samples/my_digit.png
    python -m src.predict --image samples/my_letter.png --dataset emnist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from src.preprocess import PreprocessingError, get_class_names, preprocess_custom_image


def load_trained_model(model_path: Union[str, Path]):
    """Load a saved .keras model (TensorFlow is imported only when needed)."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run `python -m src.train` first.")
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def predict_image(image_path: Union[str, Path], model, class_names: Sequence[str],
                  show: bool = True, save_path: Union[str, Path, None] = None) -> Dict:
    """Preprocess one image and predict. Returns label, confidence and top-3."""
    model_input, steps = preprocess_custom_image(image_path, return_steps=True)
    probs = model.predict(model_input, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    result = {"label": class_names[top[0]], "confidence": float(probs[top[0]]),
              "top3": [(class_names[i], float(probs[i])) for i in top]}

    if show or save_path:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
        axes[0].imshow(steps["gray"], cmap="gray")
        axes[0].set_title("Input image")
        axes[1].imshow(steps["final_28x28"], cmap="gray")
        axes[1].set_title(f"Model sees -> predicts '{result['label']}'\nconfidence {result['confidence']:.1%}")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a handwritten character.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    parser.add_argument("--model", default=None, help="default: models/<dataset>_cnn.keras")
    parser.add_argument("--save", default=None, help="optional path to save the result figure")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args()

    model_path = args.model or f"models/{args.dataset}_cnn.keras"
    try:
        model = load_trained_model(model_path)
        result = predict_image(args.image, model, get_class_names(args.dataset),
                               show=not args.no_show, save_path=args.save)
    except (FileNotFoundError, PreprocessingError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Prediction: {result['label']}  (confidence {result['confidence']:.1%})")
    print("Top 3:", ", ".join(f"{c} {p:.1%}" for c, p in result["top3"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
