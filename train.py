"""Train and evaluate the CNN.

Usage (from the project root):
    python -m src.train                      # MNIST digits
    python -m src.train --dataset emnist     # EMNIST balanced (47 classes)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.evaluate import evaluate_predictions, plot_history, plot_misclassified
from src.model import build_cnn
from src.preprocess import SEED, get_class_names, load_dataset, split_and_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a handwritten character CNN.")
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    parser.add_argument("--epochs", type=int, default=20, help="maximum epochs (early stopping may stop sooner)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)  # seeds Python, NumPy and TensorFlow

    class_names = get_class_names(args.dataset)
    model_path = Path(args.models_dir) / f"{args.dataset}_cnn.keras"
    results_dir = Path(args.results_dir) / args.dataset
    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data -----------------------------------------------------------------
    data = split_and_normalize(*load_dataset(args.dataset))
    print({k: v.shape for k, v in data.items()})

    # 2. Model ----------------------------------------------------------------
    model = build_cnn(num_classes=len(class_names))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # 3. Train ----------------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(data["x_train"], data["y_train"],
                        validation_data=(data["x_val"], data["y_val"]),
                        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    pd.DataFrame(history.history).to_csv(results_dir / "history.csv", index=False)
    plot_history(history.history, results_dir)

    # 4. Evaluate on the untouched test set, using the model re-loaded from disk
    model = tf.keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate(data["x_test"], data["y_test"], verbose=0)
    y_pred = model.predict(data["x_test"], verbose=0).argmax(axis=1)
    metrics = evaluate_predictions(data["y_test"], y_pred, class_names, results_dir)
    plot_misclassified(data["x_test"], data["y_test"], y_pred, class_names, out_dir=results_dir)

    metrics.update(dataset=args.dataset, test_loss=float(test_loss), test_accuracy=float(test_acc),
                   epochs_run=len(history.history["loss"]), seed=SEED,
                   tensorflow_version=tf.__version__)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
