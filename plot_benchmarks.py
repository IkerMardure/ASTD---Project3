import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_benchmark_results(results_csv_path: Path) -> pd.DataFrame:
    if not results_csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {results_csv_path}")

    df = pd.read_csv(results_csv_path)

    # Normalize classifier names trimming spaces
    df["classifier"] = df["classifier"].astype(str).str.strip()

    return df


def plot_accuracy_across_datasets(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    techniques = [
        "TSF (ours)",
        "1NN-DTW",
        "1NN-ED",
        "BOSS-ensemble",
        "Shapelet Transform (ST)",
        "Rocket",
        "RISE",
        "catch22",
    ]

    pivot = df.pivot_table(
        index="dataset",
        columns="classifier",
        values="accuracy",
        aggfunc="mean",
    )

    pivot = pivot.reindex(sorted(pivot.index))

    short_names = {
        "ECG5000": "ECG",
        "ElectricDevices": "ED",
        "GunPoint": "GP",
        "InlineSkate": "IS",
        "ItalyPowerDemand": "IPD",
    }
    x_labels = [short_names.get(ds, ds) for ds in pivot.index]

    plt.figure(figsize=(10, 7), facecolor="none")
    plt.xlabel("Dataset", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, ha="right", fontsize=14)

    for technique in techniques:
        if technique not in pivot.columns:
            continue

        y = (pivot[technique].values * 100).astype(float)
        y_smooth = pd.Series(y).rolling(window=3, min_periods=1, center=True).mean().values
        is_tsf = technique == "TSF (ours)"

        plt.plot(
            range(len(x_labels)),
            y_smooth,
            marker="o",
            linewidth=3.5 if is_tsf else 1.8,
            markersize=8 if is_tsf else 6,
            label=technique,
            color="#006400" if is_tsf else "#90EE90",
            alpha=1.0 if is_tsf else 0.8,
        )

    plt.ylim(60, 100)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(title="Technique", loc="lower left", bbox_to_anchor=(1.02, 0.03), frameon=False)
    plt.tight_layout()

    plt.savefig(out_path, dpi=180, transparent=True)
    plt.close()


def plot_gunpoint_train_test_time(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gunpoint = df[df["dataset"].str.lower() == "gunpoint"]
    if gunpoint.empty:
        raise ValueError("No hay datos para GunPoint en el CSV")

    gunpoint = gunpoint.copy()
    gunpoint.sort_values("accuracy", ascending=False, inplace=True)

    techniques = gunpoint["classifier"].tolist()
    fit_time = gunpoint["fit_time_s"].astype(float).tolist()
    predict_time = gunpoint["predict_time_s"].astype(float).tolist()

    x = np.arange(len(techniques))
    width = 0.38

    short_tech = {
        "TSF (ours)": "TSF",
        "1NN-DTW": "DTW",
        "1NN-ED": "ED",
        "BOSS-ensemble": "BOSS",
        "Shapelet Transform (ST)": "ST",
        "Rocket": "ROCKET",
        "RISE": "RISE",
        "catch22": "C22",
    }
    x_labels = [short_tech.get(t, t) for t in techniques]

    plt.figure(figsize=(9, 5), facecolor="none")
    plt.xlabel("Technique", fontsize=16)
    plt.ylabel("Time (seconds)", fontsize=16)

    fit_colors = ["#006400" if t == "TSF (ours)" else "#90EE90" for t in techniques]
    predict_colors = ["#006400" if t == "TSF (ours)" else "#90EE90" for t in techniques]
    predict_alpha = 0.45

    p1 = plt.bar(x, fit_time, width, label="Train", color=fit_colors)
    p2 = plt.bar(x, predict_time, width, bottom=fit_time, label="Test", color=predict_colors, alpha=predict_alpha)

    # Use normal linear seconds scale for train+test time
    plt.yscale("linear")
    plt.xticks(x, x_labels, rotation=35, ha="right", fontsize=14)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(title="Phase", frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    for i, (ft, pt) in enumerate(zip(fit_time, predict_time)):
        plt.text(x[i], ft / 2, f"{ft:.3f}", ha="center", va="center", fontsize=8, color="white" if techniques[i] == "TSF (ours)" else "black")
        plt.text(x[i], ft + pt / 2, f"{pt:.3f}", ha="center", va="center", fontsize=8, color="white" if techniques[i] == "TSF (ours)" else "black")

    plt.savefig(out_path, dpi=180, transparent=True)
    plt.close()


def plot_gunpoint_accuracy(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gunpoint = df[df["dataset"].str.lower() == "gunpoint"]
    if gunpoint.empty:
        raise ValueError("No hay datos para GunPoint en el CSV")

    gunpoint = gunpoint.copy()
    gunpoint.sort_values("accuracy", ascending=False, inplace=True)

    techniques = gunpoint["classifier"].tolist()
    accuracy = gunpoint["accuracy"].astype(float).tolist()

    x = np.arange(len(techniques))

    short_tech = {
        "TSF (ours)": "TSF",
        "1NN-DTW": "DTW",
        "1NN-ED": "ED",
        "BOSS-ensemble": "BOSS",
        "Shapelet Transform (ST)": "ST",
        "Rocket": "ROCKET",
        "RISE": "RISE",
        "catch22": "C22",
    }
    x_labels = [short_tech.get(t, t) for t in techniques]

    plt.figure(figsize=(9, 5), facecolor="none")
    plt.xlabel("Technique", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)

    colors = ["#006400" if t == "TSF (ours)" else "#90EE90" for t in techniques]
    accuracy_pct = [a * 100 for a in accuracy]
    plt.bar(x, accuracy_pct, color=colors)

    plt.ylim(0, 100)
    plt.xticks(x, x_labels, rotation=35, ha="right", fontsize=14)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    for i, acc in enumerate(accuracy_pct):
        plt.text(x[i], acc + 1, f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.savefig(out_path, dpi=180, transparent=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Genera gráficas de benchmark a partir de benchmark_comparison.csv")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/benchmark_comparison.csv"),
        help="Ruta al CSV de resultados",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualization"),
        help="Directorio de salida para PNG",
    )
    args = parser.parse_args()

    df = load_benchmark_results(args.results_csv)

    plot_accuracy_across_datasets(df, args.out_dir / "accuracy_across_datasets.png")
    plot_gunpoint_train_test_time(df, args.out_dir / "gunpoint_train_test_time.png")
    plot_gunpoint_accuracy(df, args.out_dir / "gunpoint_accuracy.png")

    print("Gráficas generadas:")
    print(" -", args.out_dir / "accuracy_across_datasets.png")
    print(" -", args.out_dir / "gunpoint_train_test_time.png")
    print(" -", args.out_dir / "gunpoint_accuracy.png")


if __name__ == "__main__":
    main()
