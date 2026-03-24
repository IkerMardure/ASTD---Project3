import os
import json
import numpy as np
from aeon.datasets import load_from_ts_file

BASE_PATH = "./"  # folder where datasets are located


def load_ucr_ts_file(filepath):
    """
    Load a .ts file using aeon.

    Returns:
        X: time series data
        y: labels
    """
    X, y = load_from_ts_file(filepath)

    # Typical shape:
    # univariate: (n_instances, 1, series_length)
    # multivariate: (n_instances, n_channels, series_length)
    return X, y


def get_series_length(X):
    """
    Extract the series length from the dataset.
    Assumes shape (n_instances, n_channels, series_length).
    """
    return X.shape[2]


def process_dataset(dataset_path):
    """
    Process a dataset and extract statistics.
    """
    dataset_name = os.path.basename(dataset_path)

    train_file = os.path.join(dataset_path, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(dataset_path, f"{dataset_name}_TEST.ts")

    # Load data
    X_train, y_train = load_ucr_ts_file(train_file)
    X_test, y_test = load_ucr_ts_file(test_file)

    # Dataset sizes
    train_size = len(X_train)
    test_size = len(X_test)

    # Length of each time series
    series_length = get_series_length(X_train)

    # Number of unique classes (use both train and test for safety)
    num_classes = len(np.unique(np.concatenate([y_train, y_test])))

    # Total number of samples
    total_size = train_size + test_size

    return {
        "dataset": dataset_name,
        "train_size": train_size,
        "test_size": test_size,
        "series_length": series_length,
        "num_classes": num_classes,
        "total_size": total_size
    }


def main():
    datasets = [
        "ECG5000",
        "ElectricDevices",
        "GunPoint",
        "InlineSkate",
        "ItalyPowerDemand"
    ]

    results = {}

    for ds in datasets:
        dataset_path = os.path.join(BASE_PATH, ds)

        # Optional safety check
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} not found")
            continue

        results[ds] = process_dataset(dataset_path)

    # Save results to JSON file
    with open("dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("JSON generated: dataset_summary.json")
    print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
