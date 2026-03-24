import os
import json
import numpy as np

BASE_PATH = "./"  # carpeta donde están los datasets

def load_ucr_file(filepath):
    data = np.loadtxt(filepath)
    y = data[:, 0]      # labels
    X = data[:, 1:]     # series
    return X, y

def process_dataset(dataset_path):
    dataset_name = os.path.basename(dataset_path)

    train_file = os.path.join(dataset_path, f"{dataset_name}_TRAIN.tsv")
    test_file = os.path.join(dataset_path, f"{dataset_name}_TEST.tsv")

    X_train, y_train = load_ucr_file(train_file)
    X_test, y_test = load_ucr_file(test_file)

    train_size = len(X_train)
    test_size = len(X_test)

    # longitud de la serie (todas suelen tener la misma)
    series_length = X_train.shape[1]

    # número de clases (train + test por si acaso)
    num_classes = len(np.unique(np.concatenate([y_train, y_test])))

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
    datasets = ["ECG5000", "ElectricDevices", "GunPoint", "InlineSkate", "ItalyPowerDemand"]

    results = {}

    for ds in datasets:
        dataset_path = os.path.join(BASE_PATH, ds)
        results[ds] = process_dataset(dataset_path)

    # guardar JSON
    with open("dataset_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print("JSON generado: dataset_summary.json")


if __name__ == "__main__":
    main()
