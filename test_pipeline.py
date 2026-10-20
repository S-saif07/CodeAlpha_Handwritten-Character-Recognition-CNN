"""Tests that run without TensorFlow (preprocessing, evaluation, prediction logic)."""
import tempfile
import unittest
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.evaluate import evaluate_predictions, plot_history, top_confusions
from src.predict import predict_image
from src.preprocess import (MNIST_CLASSES, PreprocessingError, preprocess_custom_image,
                            split_and_normalize)


def make_digit_image(path, digit="5", dark_on_light=True, noise=False, angle=0):
    canvas = np.full((300, 300), 255, np.uint8)
    cv2.putText(canvas, digit, (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 6, 0, 14)
    if angle:
        m = cv2.getRotationMatrix2D((150, 150), angle, 1)
        canvas = cv2.warpAffine(canvas, m, (300, 300), borderValue=255)
    if not dark_on_light:
        canvas = 255 - canvas
    if noise:
        rng = np.random.default_rng(0)
        canvas = np.clip(canvas + rng.normal(0, 12, canvas.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), canvas)


class FakeModel:
    """Stands in for a Keras model: always says class 3 with 90% confidence."""
    def predict(self, x, verbose=0):
        probs = np.full((len(x), 10), 0.1 / 9)
        probs[:, 3] = 0.9
        return probs


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_output_format(self):
        make_digit_image(self.tmp / "a.png")
        x = preprocess_custom_image(self.tmp / "a.png")
        self.assertEqual(x.shape, (1, 28, 28, 1))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(0.0 <= x.min() and x.max() <= 1.0)
        self.assertAlmostEqual(float(x.max()), 1.0, places=5)
        img = x[0, ..., 0]
        self.assertEqual(img[:2].sum() + img[-2:].sum(), 0)  # empty margins
        ys, xs = np.mgrid[0:28, 0:28]
        cy, cx = (img * ys).sum() / img.sum(), (img * xs).sum() / img.sum()
        self.assertLess(abs(cy - 14), 1.5)
        self.assertLess(abs(cx - 14), 1.5)

    def test_polarity_is_handled(self):
        make_digit_image(self.tmp / "dark.png", dark_on_light=True)
        make_digit_image(self.tmp / "light.png", dark_on_light=False)
        a = preprocess_custom_image(self.tmp / "dark.png").ravel()
        b = preprocess_custom_image(self.tmp / "light.png").ravel()
        self.assertGreater(np.corrcoef(a, b)[0, 1], 0.9)

    def test_noisy_rotated_image(self):
        make_digit_image(self.tmp / "n.png", digit="7", noise=True, angle=8)
        x = preprocess_custom_image(self.tmp / "n.png")
        self.assertEqual(x.shape, (1, 28, 28, 1))
        self.assertGreater(float((x > 0.5).sum()), 20)  # a real stroke, not noise

    def test_bad_inputs(self):
        with self.assertRaises(FileNotFoundError):
            preprocess_custom_image(self.tmp / "missing.png")
        (self.tmp / "fake.png").write_text("not an image")
        with self.assertRaises(PreprocessingError):
            preprocess_custom_image(self.tmp / "fake.png")
        cv2.imwrite(str(self.tmp / "blank.png"), np.full((100, 100), 255, np.uint8))
        with self.assertRaises(PreprocessingError):
            preprocess_custom_image(self.tmp / "blank.png")

    def test_split_has_no_overlap_and_is_normalized(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 256, (500, 28, 28), dtype=np.uint8)
        y = np.repeat(np.arange(10), 50)
        d = split_and_normalize(x, y, x[:100], y[:100])
        self.assertEqual(len(d["x_train"]) + len(d["x_val"]), 500)
        self.assertEqual(d["x_train"].shape[1:], (28, 28, 1))
        self.assertLessEqual(float(d["x_train"].max()), 1.0)


class EvaluateAndPredictTests(unittest.TestCase):
    def test_evaluate_outputs(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 10, 300)
        y_pred = np.where(rng.random(300) < 0.8, y_true, rng.integers(0, 10, 300))
        out = Path(tempfile.mkdtemp())
        metrics = evaluate_predictions(y_true, y_pred, MNIST_CLASSES, out)
        history = {"accuracy": [0.5, 0.7], "val_accuracy": [0.4, 0.6],
                   "loss": [1.0, 0.6], "val_loss": [1.1, 0.7]}
        plot_history(history, out)
        for name in ("classification_report.txt", "confusion_matrix.png", "top_confusions.csv",
                     "accuracy_curve.png", "loss_curve.png"):
            self.assertTrue((out / name).exists(), name)
        self.assertAlmostEqual(metrics["accuracy"], float((y_true == y_pred).mean()))
        self.assertLessEqual(len(top_confusions(y_true, y_pred, MNIST_CLASSES)), 10)

    def test_predict_image_with_stand_in_model(self):
        tmp = Path(tempfile.mkdtemp())
        make_digit_image(tmp / "d.png")
        result = predict_image(tmp / "d.png", FakeModel(), MNIST_CLASSES, show=False,
                               save_path=tmp / "out.png")
        self.assertEqual(result["label"], "3")
        self.assertAlmostEqual(result["confidence"], 0.9, places=5)
        self.assertTrue((tmp / "out.png").exists())


if __name__ == "__main__":
    unittest.main()
