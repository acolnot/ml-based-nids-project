# Attacking a ML-Based Network Intrusion Detection System


## Prerequisites

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) as the package manager. Please ensure you have it installed on your machine.

## Local Setup 

### 1. Clone the repository & sync dependencies

```bash
git clone https://github.com/acolnot/ml-based-nids-project
cd ml-based-nids-project
uv sync
```

### 2. Download the dataset

In this project we are using the Intrusion detection evaluation dataset [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html). Because the raw dataset is too large to host on GitHub, you need to download it manually to your local machine before you can run any code.

1. Create a folder named `data/` in the root directory of the project.
2. Go to the [Kaggle CICIDS2017 Dataset page](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) and download **only** the file named `Wednesday-workingHours.pcap_ISCX.csv`. 
3. Move this downloaded CSV directly into your local `data/` folder.

### 3. Generate the cleaned data

Run the preprocessing pipeline to clean the data.

```bash
uv run src/clean_data.py
```

## Train the PyTorch MLP

The training pipeline:

- creates stratified 70/15/15 train, validation, and test splits;
- fits standardization using only the training split;
- trains a `256 -> 128 -> 64` MLP with batch normalization, ReLU, and dropout;
- uses validation-loss early stopping; and
- reports accuracy, precision, recall, and F1.

```bash
uv run python -m src.train
```

For a short end-to-end smoke test:

```bash
uv run python -m src.train --max-rows 20000 --epochs 2 --output-dir artifacts/smoke
```
