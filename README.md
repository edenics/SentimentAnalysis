# ESMatch: Semi-Supervised Text Classification with Entropy-Guided Soft Mask and Distribution Regularization

This repository contains the official implementation of **ESMatch (Entropy-guided Soft Mask Match)**, a semi-supervised text classification framework designed for scenarios with limited labeled samples and imbalanced class distributions. 

ESMatch mitigates pseudo-label confirmation bias, prediction overconfidence, and decision boundary shift through a dual-level constraint mechanism:
- **Sample Level**: An entropy-guided soft mask that dynamically adjusts weights for unlabeled samples based on class-level prediction uncertainty.
- **Distribution Level**: A distribution regularization constraint that prevents unsupervised marginal predictions from drifting excessively, using real-time supervised predictions as a reference.

---

## 🚀 Features

- **Multi-Method Benchmarking**: Includes implementations of ESMatch, FixMatch, FlexMatch, FreeMatch, and SoftMatch.
- **Robust Feature Extraction**: Integrated with pre-trained BERT-Base-Chinese models and PCA dimensionality reduction.
- **Ablation Studies**: Built-in control switches to easily verify the individual contributions of the soft mask and distribution regularization mechanisms.
- **Multi-Dataset Support**: Configured for benchmarks such as CSL (Academic Literature), KUAKE-QIC (Medical Intent), and Sogou News.

---

## 📦 Requirements & Dependencies

The codebase requires **Python 3.8+** (tested and verified on **Python 3.12**).

### Core Dependencies
No specific package versions are locked, but the following standard library and third-party packages must be installed:

- **`torch` (PyTorch)**: For deep learning model construction and joint loss optimizations.
- **`transformers` (Hugging Face)**: For loading pre-trained BERT models and text tokenization.
- **`scikit-learn`**: For PCA feature reduction and evaluation metrics (Accuracy, Weighted F1).
- **`pandas` & `numpy`**: For data manipulation, matrix operations, and loading datasets.
- **`tqdm`**: For progress bar visualization during training loops.
- **`joblib`**: For caching extracted BERT features to speed up repetitive experiments.

---

## 🛠️ Usage

### 1. Configuration
Open `main.py` and modify the parameters section to select datasets, algorithms, and training configurations:

```python
# Select methods and datasets to run
METHODS_TO_RUN = ['ESMatch']  # Options: 'SOFTMATCH', 'FLEXMATCH', 'FREEMATCH', 'ESMatch'
DATASETS_TO_RUN = ['qic']  # Options: 'qic', 'csl', 'sogou'

# Training size configurations
LABELED_CONFIGS = [250]    # Number of labeled samples per class config
NUM_RUNS = 100             # Number of independent random seeds to run for stability
```

### 2. Ablation Configuration
You can toggle the ablation switches in `main.py` to evaluate degraded variants of ESMatch:
```python
# 1. Disable Sigmoid soft-mask (degrades to traditional hard truncation)
ABLATION_WO_SOFT = False

# 3. Disable distribution regularization (degrades to pure pseudo-labeling)
ABLATION_WO_DA = False
```

### 3. Running the Code
Run the main script to start feature extraction and semi-supervised training:
```bash
python main.py
```

---

## 📁 Repository Structure

```text
├── main.py              # Main training and evaluation script
├── cache/               # Directory for cached BERT embeddings (auto-generated)
├── weights/             # Directory for model checkpoint checkpoints (auto-generated)
└── result/              # Directory containing output logs and csv reports (auto-generated)
```

> [!NOTE]
> During the first run, the script will download the `BERT-Base-Chinese` model from Hugging Face. Feature extraction caches will be saved in `cache/` to ensure subsequent runs are instantaneous.
