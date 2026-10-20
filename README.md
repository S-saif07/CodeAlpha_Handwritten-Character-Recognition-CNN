# Handwritten Character Recognition with a CNN

A beginner-friendly deep learning project that recognises handwritten digits (0-9) from images using a
Convolutional Neural Network built with TensorFlow/Keras, plus a preprocessing pipeline for your own
handwriting. The code is ready to extend to letters with EMNIST.

> \*\*Status:\*\* MNIST digit recognition is implemented. EMNIST support is wired in but \*\*not yet run or
> verified\*\*. Fill in the results section below with the numbers from your own run.

## Problem statement and objectives

Given a picture of a handwritten character, predict which character it is.

* Load and explore MNIST; normalise pixels to 0-1; build train / validation / test sets without leakage.
* Build, train and evaluate a CNN (accuracy, precision, recall, F1, confusion matrix, learning curves).
* Predict from a user-supplied image: grayscale, polarity fix, crop, resize, centre, normalise.

## Features

* Modular code in `src/` (data, model, training, evaluation, prediction) and a step-by-step notebook.
* Reproducible: fixed seed, stratified validation split, pinned structure, saved metrics.
* Custom-image prediction with friendly errors for missing, corrupt or blank images.
* Unit tests for preprocessing, evaluation and prediction logic (`python -m unittest discover -s tests -t .`).

## Dataset

* **MNIST**: 70,000 grayscale 28x28 images (60,000 train / 10,000 test), labels 0-9, pixel values 0-255.
Loaded with `tf.keras.datasets.mnist`. Source: http://yann.lecun.com/exdb/mnist/
* **EMNIST (extension)**: extended MNIST with letters. This project uses the `balanced` split (47 classes)
via `tensorflow-datasets`. Source: https://www.nist.gov/itl/products-and-services/emnist-dataset
* Data is downloaded automatically on first run and is not committed to the repo.

## Technologies

Python, TensorFlow/Keras, OpenCV, Pillow, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn.

## CNN architecture

|Layer|Purpose|
|-|-|
|Input 28x28x1|one grayscale image|
|2 x Conv2D(32, 3x3, ReLU) + MaxPool + Dropout(0.25)|learn simple features (edges), halve the image size|
|2 x Conv2D(64, 3x3, ReLU) + MaxPool + Dropout(0.25)|combine edges into strokes and loops|
|Flatten|turn the feature maps into one long vector|
|Dense(128, ReLU) + Dropout(0.5)|combine features into a decision|
|Dense(n\_classes, softmax)|probability for each class (10 for MNIST)|

Adam optimizer, sparse categorical cross-entropy loss, EarlyStopping + ModelCheckpoint on validation loss.

## Installation

```bash
git clone https://github.com/<your-username>/handwritten-character-recognition-cnn.git
cd handwritten-character-recognition-cnn
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage

Run all commands from the project root.

```bash
python -m src.train                          # train + evaluate on MNIST, saves model and results
python -m src.train --dataset emnist         # EMNIST balanced (experimental, see Limitations)
python -m unittest discover -s tests -t .    # run the tests
```

Or open `notebooks/handwritten\_character\_recognition.ipynb` in Jupyter or Google Colab
(Colab: clone the repo first; the first notebook cell explains how).

### Predict your own image

Write one digit in dark ink on plain paper, photograph or scan it, and save it in `samples/`:

```bash
python -m src.predict --image samples/my\_digit.png
```

The model file is created by training; it is not stored in Git (see `.gitignore`).

## Results

**Not yet filled in.** After running `python -m src.train`, copy the numbers from
`results/mnist/metrics.json` here:

|Metric|Value|
|-|-|
|Test accuracy|*your result*|
|Test loss|*your result*|
|Macro precision / recall / F1|*your result*|

Figures are written to `results/mnist/` (`accuracy\_curve.png`, `loss\_curve.png`, `confusion\_matrix.png`,
`misclassified\_examples.png`). Add the ones you want to show with `git add -f` and embed them here, plus a
screenshot of a custom prediction.

## Project structure

```
handwritten-character-recognition-cnn/
├── README.md  requirements.txt  .gitignore  LICENSE
├── notebooks/handwritten\_character\_recognition.ipynb
├── src/   preprocess.py  model.py  train.py  evaluate.py  predict.py
├── tests/ test\_pipeline.py
├── models/ results/ samples/        (.gitkeep placeholders)
```

## Limitations

* Trained on MNIST only: it recognises digits, not letters. Real-world handwriting that differs from MNIST
(thick pens, unusual styles, photos with shadows) can be predicted with lower accuracy.
* One character per image. Touching or multi-part characters can be cropped wrongly.
* Dark-on-light / light-on-dark is detected from the image border, which fails if the character touches the edge.
* The confidence score is the model's softmax output, not a guarantee of correctness.
* EMNIST support has not been run yet; check the image orientation visually before trusting it.

## Future improvements (not implemented)

1. Letters with EMNIST (code path exists; run, verify, report results).
2. Several characters per image (contour detection, one prediction per box).
3. Word segmentation into characters.
4. Word/sentence recognition with a CRNN + CTC loss on a word dataset (e.g. IAM).
5. Data augmentation (rotation, zoom, shift) to better handle real photos.
6. A Streamlit or Flask web interface.

## Author and license

Author : Siam Bin Saif



GitHub : https://github.com/S-saif07



LinkedIn link : https://www.linkedin.com/in/siam-saif-exe976/



>. Released under the MIT License (see `LICENSE`).

