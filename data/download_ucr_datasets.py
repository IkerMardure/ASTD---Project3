"""Download selected UCR datasets using aeon.

Usage:
    python data/download_ucr_datasets.py
"""

from __future__ import annotations

from pathlib import Path

from aeon.datasets import load_classification

DATASETS = [
    "ItalyPowerDemand",
    "GunPoint",
    "ECG5000",
    "InlineSkate",
    "ElectricDevices",
]


def main() -> None:
    target_dir = Path("data")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading into: {target_dir.resolve()}")
    for name in DATASETS:
        X_train, y_train = load_classification(
            name, split="TRAIN", extract_path=str(target_dir)
        )
        X_test, y_test = load_classification(
            name, split="TEST", extract_path=str(target_dir)
        )

        print(
            f"{name}: "
            f"train={X_train.shape}/{y_train.shape}, "
            f"test={X_test.shape}/{y_test.shape}"
        )


if __name__ == "__main__":
    main()
