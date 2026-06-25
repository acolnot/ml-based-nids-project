from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import NIDSMLP


SEED = 42
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5
THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NIDS MLP.")
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/Cleaned_Wednesday.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mlp"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(
    path: Path, max_rows: int | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `uv run src/clean_data.py` first."
        )

    data = pd.read_csv(path)
    if max_rows is not None and max_rows < len(data):
        data = data.sample(max_rows, random_state=SEED)

    labels = data.pop("Label").to_numpy(dtype=np.float32)
    features = data.to_numpy(dtype=np.float32)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Label must contain only 0 and 1")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values")

    return features, labels, data.columns.tolist()


def prepare_data(
    features: np.ndarray, labels: np.ndarray
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    StandardScaler,
]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        stratify=labels,
        random_state=SEED,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE),
        stratify=y_temp,
        random_state=SEED,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_validation = scaler.transform(x_validation).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    return (
        (x_train, y_train),
        (x_validation, y_validation),
        (x_test, y_test),
        scaler,
    )


def make_loader(
    data: tuple[np.ndarray, np.ndarray], batch_size: int, shuffle: bool
) -> DataLoader:
    features, labels = data
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, object]:
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced NaN or infinite probabilities")

    labels = labels.astype(np.int64)
    predictions = (probabilities >= THRESHOLD).astype(np.int64)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def evaluate(
    model: NIDSMLP,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    with torch.inference_mode():
        for features, labels in loader:
            # Asynchronous transfers are safe for pinned CUDA memory, but not MPS.
            non_blocking = device.type == "cuda"
            features = features.to(device, non_blocking=non_blocking)
            device_labels = labels.to(device, non_blocking=non_blocking)
            logits = model(features)

            total_loss += loss_function(logits, device_labels).item() * len(labels)
            all_labels.append(labels.numpy().astype(np.int64))
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy().copy())

    metrics = calculate_metrics(
        np.concatenate(all_labels), np.concatenate(all_probabilities)
    )
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train(
    model: NIDSMLP,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], int]:
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= len(train_loader.dataset)
        validation = evaluate(model, validation_loader, loss_function, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation": validation}
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.5f} | "
            f"val loss {validation['loss']:.5f} | val F1 {validation['f1']:.4f}"
        )

        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return history, best_epoch


def save_results(
    output_dir: Path,
    model: NIDSMLP,
    scaler: StandardScaler,
    feature_names: list[str],
    history: list[dict[str, object]],
    best_epoch: int,
    test_metrics: dict[str, object],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_names),
            "hidden_dims": (256, 128, 64),
            "dropout": 0.2,
            "feature_names": feature_names,
            "target_column": "Label",
            "threshold": THRESHOLD,
            "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
            "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
        },
        output_dir / "model.pt",
    )

    report = {
        "config": {
            "data_path": str(args.data_path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": SEED,
        },
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "training_history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    features, labels, feature_names = load_data(args.data_path, args.max_rows)
    print(
        f"Loaded {len(labels):,} rows with {len(feature_names)} features "
        f"({int(labels.sum()):,} malicious, {int((1 - labels).sum()):,} benign)."
    )

    train_data, validation_data, test_data, scaler = prepare_data(features, labels)
    print(
        f"Split sizes — train: {len(train_data[1]):,}, "
        f"validation: {len(validation_data[1]):,}, test: {len(test_data[1]):,}"
    )

    train_loader = make_loader(train_data, args.batch_size, shuffle=True)
    validation_loader = make_loader(validation_data, args.batch_size, shuffle=False)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False)

    model = NIDSMLP(input_dim=len(feature_names)).to(device)
    history, best_epoch = train(
        model, train_loader, validation_loader, args.epochs, device
    )

    test_metrics = evaluate(
        model, test_loader, nn.BCEWithLogitsLoss(), device
    )
    print("\nTest results")
    for name in ("loss", "accuracy", "precision", "recall", "f1"):
        print(f"  {name}: {test_metrics[name]:.5f}")

    save_results(
        args.output_dir,
        model,
        scaler,
        feature_names,
        history,
        best_epoch,
        test_metrics,
        args,
    )
    print(f"Saved model and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
