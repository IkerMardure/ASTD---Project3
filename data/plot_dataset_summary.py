import json
import matplotlib.pyplot as plt
import numpy as np


def load_dataset_summary(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    json_path = "dataset_summary.json"
    datasets = load_dataset_summary(json_path)

    # Sort datasets by total size
    sorted_items = sorted(datasets.items(), key=lambda x: x[1]["total_size"])

    names = [item[0] for item in sorted_items]
    train_sizes = [item[1]["train_size"] for item in sorted_items]
    test_sizes = [item[1]["test_size"] for item in sorted_items]
    total_sizes = [item[1]["total_size"] for item in sorted_items]

    x = np.arange(len(names))

    plt.figure(figsize=(12, 6))

    # Stacked bars (mejoradas)
    plt.bar(x, train_sizes,
            label="Train size",
            color="#f8c8dc",   # rosa palo
            edgecolor="black")

    plt.bar(x, test_sizes,
            bottom=train_sizes,
            label="Test size",
            color="#ff1493",   # fucsia
            edgecolor="black")

    # Línea roja
    plt.plot(x, total_sizes,
             marker="o",
             linewidth=2.5,
             color="red",
             label="Total size")

    # Etiquetas
    offset = max(total_sizes) * 0.015
    for i, total in enumerate(total_sizes):
        plt.text(i, total + offset, str(total), ha="center", fontsize=9)

    plt.xticks(x, names, rotation=20)
    plt.ylabel("Number of time series")
    plt.title("Dataset size comparison")

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig("dataset_size_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()