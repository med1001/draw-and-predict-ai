# MNIST CNN -> TensorFlow Lite (ML Project)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project trains a CNN on MNIST handwritten digits, exports it to TensorFlow Lite, and runs inference in:

- a host test script with real MNIST test images (`test_tflite_host.py`)
- an interactive drawing app (`draw_and_predict.py`)

## Results (Typical)

From local CPU tests:

- Accuracy on MNIST test subset: around `98.8%` to `99.4%`
- Average TFLite inference time on desktop CPU: around `1-3 ms/image`
- Real-time interactive predictions in the Tkinter drawing app

## Demo / Screenshot

You can add your own screenshot for GitHub preview:

1. Run `python draw_and_predict.py`
2. Draw a digit and click **Predict**
3. Take a screenshot and save it to `assets/demo.png`
4. Save your screenshot as `assets/demo.png`

![Demo screenshot](assets/demo.png)

## Project Structure

- `train_mnist_cnn.py` - Train CNN, evaluate, export SavedModel, convert to TFLite
- `convert_to_tflite.py` - Convert exported SavedModel to TFLite
- `test_tflite_host.py` - Quick accuracy + latency check on MNIST test subset
- `draw_and_predict.py` - Draw digits and predict class/confidence
- `mnist_cnn_model.tflite` - Generated TFLite model
- `tutoriel_MNIST.md` - Extended tutorial notes

## Requirements

- Python 3.10+ recommended
- Windows, macOS, or Linux

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv edgeai_env
.\edgeai_env\Scripts\activate
```

macOS/Linux:

```bash
python -m venv edgeai_env
source edgeai_env/bin/activate
```

### 2) Install packages

```bash
pip install -r requirements.txt
```

### 3) Train and export the model

```bash
python train_mnist_cnn.py
```

Expected result:
- `mnist_cnn_model/` generated (SavedModel)
- `mnist_cnn_model.tflite` generated

### 4) Run host evaluation (real MNIST test images)

```bash
python test_tflite_host.py
```

Expected output format:

```text
Samples tested: 1000
Accuracy: 0.99xx (99.xx%)
Average inference time: X.XXX ms/sample
```

### 5) Run interactive app

```bash
python draw_and_predict.py
```

Draw a digit (0-9), click **Predict**, and check the terminal output for:
- predicted digit
- confidence
- inference time

## Typical Workflow

1. Train/update model: `python train_mnist_cnn.py`
2. (Optional) Re-convert model: `python convert_to_tflite.py`
3. Validate model quickly: `python test_tflite_host.py`
4. Demo interactively: `python draw_and_predict.py`

## Notes

- `tf.lite.Interpreter` currently works, but TensorFlow warns it will be replaced in future versions by LiteRT.
- If you only need the `.tflite` model for demos, you can skip committing `mnist_cnn_model/` and regenerate it when needed.

## How to Contribute

Contributions are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Make your changes (code, docs, tests, or examples)
4. Run the project checks locally:
   - `python train_mnist_cnn.py`
   - `python test_tflite_host.py`
   - `python draw_and_predict.py`
5. Commit with a clear message and open a Pull Request

Good first contributions:
- improve preprocessing for drawn digits
- add better evaluation metrics (confusion matrix, per-class accuracy)
- improve UX of the drawing app
- update docs/tutorial with examples and screenshots

## License

This project is licensed under the MIT License - see the `LICENSE` file.
