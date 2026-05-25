"""Run MNIST ablation tests for Dropout and BatchNorm.

The script compares four model variants:
- baseline: Dropout + BatchNorm
- no_dropout: BatchNorm only
- no_batchnorm: Dropout only
- no_dropout_no_batchnorm: neither regularization layer

Defaults use the full MNIST train/test split and 20 epochs for the report run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data import load_mnist  # noqa: E402
from network import NeuralNetwork  # noqa: E402
from optimizers import Adam  # noqa: E402
from training import evaluate, train  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    use_dropout: bool
    use_batchnorm: bool


VARIANTS = [
    Variant("baseline_dropout_batchnorm", use_dropout=True, use_batchnorm=True),
    Variant("no_dropout", use_dropout=False, use_batchnorm=True),
    Variant("no_batchnorm", use_dropout=True, use_batchnorm=False),
    Variant("no_dropout_no_batchnorm", use_dropout=False, use_batchnorm=False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=60_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis" / "dropout_batchnorm_ablation_results.csv",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=ROOT / "analysis" / "plots",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    (x_train, y_train), (x_test, y_test) = load_mnist(str(ROOT / "data"))
    x_train = x_train[: args.train_size]
    y_train = y_train[: args.train_size]
    x_test = x_test[: args.test_size]
    y_test = y_test[: args.test_size]

    rows = []
    for variant in VARIANTS:
        set_seed(args.seed)
        model = NeuralNetwork(
            use_dropout=variant.use_dropout,
            use_batchnorm=variant.use_batchnorm,
            dropout_ratio=args.dropout_ratio,
        )
        optimizer = Adam(lr=args.lr)

        started_at = time.perf_counter()
        loss_history = train(
            model,
            optimizer,
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        elapsed_sec = time.perf_counter() - started_at
        train_accuracy, total_params = evaluate(model, x_train, y_train)
        test_accuracy, _ = evaluate(model, x_test, y_test)

        row = {
            "variant": variant.name,
            "use_dropout": variant.use_dropout,
            "use_batchnorm": variant.use_batchnorm,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_size": len(x_train),
            "test_size": len(x_test),
            "lr": args.lr,
            "dropout_ratio": args.dropout_ratio if variant.use_dropout else 0.0,
            "final_loss": loss_history[-1],
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "total_params": total_params,
            "elapsed_sec": elapsed_sec,
            "loss_history": json.dumps([round(loss, 6) for loss in loss_history]),
        }
        rows.append(row)
        print(
            f"{variant.name}: test={test_accuracy:.2f}% "
            f"train={train_accuracy:.2f}% loss={loss_history[-1]:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_plots(rows, args.plot_dir)
    print(f"Wrote {args.output}")
    print(f"Wrote plots to {args.plot_dir}")


def write_plots(rows: list[dict], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = [short_label(row["variant"]) for row in rows]
    test_acc = [row["test_accuracy"] for row in rows]
    train_acc = [row["train_accuracy"] for row in rows]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, train_acc, width, label="Train")
    plt.bar(x + width / 2, test_acc, width, label="Test")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Dropout / BatchNorm Ablation Accuracy")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "accuracy_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for row in rows:
        losses = json.loads(row["loss_history"])
        plt.plot(range(1, len(losses) + 1), losses, marker="o", label=short_label(row["variant"]))
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training Loss by Variant")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curves.png", dpi=160)
    plt.close()


def short_label(variant: str) -> str:
    return {
        "baseline_dropout_batchnorm": "Dropout + BN",
        "no_dropout": "BN only",
        "no_batchnorm": "Dropout only",
        "no_dropout_no_batchnorm": "No Dropout / BN",
    }[variant]


if __name__ == "__main__":
    main()
