"""Dash app for Time Series classification metrics (Dash + Plotly).

Full dynamic migration, no static HTML pages.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.visualize_TS import generate_dataset_graph, load_ucr_txt_dataset
    from utils.visualize_predictions import (
        load_predictions_csv,
        plot_confusion_matrix,
        plot_overlay_by_correctness,
    )
except ModuleNotFoundError:
    # Fallback for environments where a third-party 'utils' package shadows
    # the local project folder.
    local_utils_dir = PROJECT_ROOT / "utils"
    if str(local_utils_dir) not in sys.path:
        sys.path.insert(0, str(local_utils_dir))

    from visualize_TS import generate_dataset_graph, load_ucr_txt_dataset
    from visualize_predictions import (
        load_predictions_csv,
        plot_confusion_matrix,
        plot_overlay_by_correctness,
    )


def _load_ucr_stats(data_dir: Path, dataset_name: str) -> dict[str, int]:
    train_path = data_dir / dataset_name / f"{dataset_name}_TRAIN.txt"
    test_path = data_dir / dataset_name / f"{dataset_name}_TEST.txt"

    n_train = 0
    n_test = 0
    length = None

    if train_path.exists():
        train_arr = pd.read_csv(train_path, sep=r"\s+", header=None).values
        if train_arr.ndim == 1:
            train_arr = train_arr.reshape(1, -1)
        n_train = int(train_arr.shape[0])
        length = int(train_arr.shape[1] - 1)

    if test_path.exists():
        test_arr = pd.read_csv(test_path, sep=r"\s+", header=None).values
        if test_arr.ndim == 1:
            test_arr = test_arr.reshape(1, -1)
        n_test = int(test_arr.shape[0])
        if length is None:
            length = int(test_arr.shape[1] - 1)

    if length is None:
        return {"nTrain": None, "nTest": None, "nSeries": None, "length": None}

    return {
        "nTrain": n_train,
        "nTest": n_test,
        "nSeries": n_train + n_test,
        "length": length,
    }


def collect_results(results_csv: Path, data_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(results_csv)
    if df.empty:
        raise ValueError(f"Results file '{results_csv}' is empty")

    datasets = sorted(df["dataset"].unique())
    dataset_stats = []
    for ds in datasets:
        stats = _load_ucr_stats(data_dir, ds)
        dataset_stats.append({"name": ds, **stats})

    classifiers = sorted(df["classifier"].unique())

    metrics_by_dataset: dict[str, dict[str, Any]] = {}
    records = []
    classifier_to_idx = {cls: idx for idx, cls in enumerate(classifiers)}

    for ds in datasets:
        ds_rows = df[df["dataset"] == ds]
        metrics_by_dataset[ds] = {}
        for _, row in ds_rows.iterrows():
            cls = row["classifier"]
            train_t = float(row.get("fit_time_s", 0.0) or 0.0)
            test_t = float(row.get("predict_time_s", 0.0) or 0.0)

            metrics_by_dataset[ds][cls] = {
                "accuracy": float(row.get("accuracy", 0.0) or 0.0),
                "fit_time_s": train_t,
                "predict_time_s": test_t,
            }

            records.append({
                "ds": ds,
                "nSeries": int(next((x["nSeries"] for x in dataset_stats if x["name"] == ds), 0) or 0),
                "length": int(next((x["length"] for x in dataset_stats if x["name"] == ds), 0) or 0),
                "technique": cls,
                "techIdx": classifier_to_idx.get(cls, 0),
                "trainT": train_t,
                "testT": test_t,
            })

    return {
        "datasets": dataset_stats,
        "classifiers": classifiers,
        "metrics_by_dataset": metrics_by_dataset,
        "raw": df,
        "timing_records": records,
    }


def collect_hyperparameter_results(hp_dir: Path) -> list[dict[str, Any]]:
    json_files = sorted(Path(hp_dir).glob("*_*_results.json"))
    results = []
    for jf in json_files:
        try:
            raw = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        dataset = jf.stem.split("_")[0] if "_" in jf.stem else "unknown"
        results.append({
            "file": str(jf),
            "dataset": dataset,
            "method": raw.get("method", "unknown"),
            "best_score": float(raw.get("best_score", 0.0)),
            "best_std": float(raw.get("best_std", 0.0)),
            "elapsed_seconds": float(raw.get("elapsed_seconds", 0.0)),
            "best_params": raw.get("best_params", {}),
            "all_results": raw.get("all_results", []),
        })
    return results


def build_metrics_figures(payload: dict[str, Any], dataset: str):
    dataset_metrics = payload["metrics_by_dataset"].get(dataset, {})
    if not dataset_metrics:
        return go.Figure(), go.Figure(), {"best": "-", "worst": "-", "mean": "-"}

    accuracies = {cls: data["accuracy"] * 100 for cls, data in dataset_metrics.items()}
    best_cls = max(accuracies, key=lambda x: accuracies[x])
    worst_cls = min(accuracies, key=lambda x: accuracies[x])

    bar_colors = []
    for cls in accuracies.keys():
        if cls == best_cls:
            bar_colors.append("#2E8B57")
        elif cls == worst_cls:
            bar_colors.append("#C0392B")
        else:
            bar_colors.append("#4C78A8")

    bar_fig = go.Figure(go.Bar(x=list(accuracies.keys()), y=list(accuracies.values()), marker_color=bar_colors))
    bar_fig.update_layout(title=f"Accuracy for {dataset}", xaxis_title="Classifier", yaxis_title="Accuracy (%)", yaxis=dict(range=[0, 100]))

    datasets_list = [d["name"] for d in payload["datasets"]]
    line_series = []
    for cls in payload["classifiers"]:
        y = [payload["metrics_by_dataset"].get(ds, {}).get(cls, {}).get("accuracy", 0.0) * 100 for ds in datasets_list]
        line_series.append(go.Scatter(x=datasets_list, y=y, mode="lines+markers", name=cls))

    line_fig = go.Figure(line_series)
    line_fig.update_layout(title="Accuracy across datasets", xaxis_title="Dataset", yaxis_title="Accuracy (%)", yaxis=dict(range=[0, 100]))

    mean_val = sum(accuracies.values()) / len(accuracies)

    summary = {
        "best": f"{best_cls} ({accuracies[best_cls]:.2f}%)",
        "worst": f"{worst_cls} ({accuracies[worst_cls]:.2f}%)",
        "mean": f"{mean_val:.2f}%",
    }

    return bar_fig, line_fig, summary


def build_timing_figures(payload: dict[str, Any], dataset: str, metric_mode: str):
    timing_records = payload.get("timing_records", [])
    if not timing_records:
        empty_fig = go.Figure(); empty_fig.update_layout(title="No timing data");
        return empty_fig, empty_fig, empty_fig

    techniques = payload["classifiers"]
    palette = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#BA7517", "#5B3D99", "#4A7D39", "#C32E4B", "#2A6D99", "#9E6712"]

    # Bar chart: metric_mode controls what is shown
    train_vals = []
    test_vals = []
    for tech in techniques:
        rec = next((r for r in timing_records if r["ds"] == dataset and r["technique"] == tech), None)
        train_vals.append(rec["trainT"] if rec else 0)
        test_vals.append(rec["testT"] if rec else 0)

    bar_fig = go.Figure()
    if metric_mode == "train":
        bar_fig.add_trace(go.Bar(y=techniques, x=train_vals, name="Train", orientation="h", marker=dict(color=palette[0], opacity=0.9)))
        bar_title = "Train time by technique"
    elif metric_mode == "predict":
        bar_fig.add_trace(go.Bar(y=techniques, x=test_vals, name="Predict", orientation="h", marker=dict(color=palette[1], opacity=0.9)))
        bar_title = "Predict time by technique"
    else:
        bar_fig.add_trace(go.Bar(y=techniques, x=train_vals, name="Train", orientation="h", marker=dict(color=palette[0], opacity=1.0)))
        bar_fig.add_trace(go.Bar(y=techniques, x=test_vals, name="Test", orientation="h", marker=dict(color=palette[1], opacity=0.35)))
        bar_fig.update_layout(barmode="stack")
        bar_title = "Train+Test time by technique"

    bar_fig.update_layout(title=bar_title, xaxis_title="Seconds", yaxis_title="Technique", yaxis={'automargin': True})

    # Bubble chart: all datasets in one view, each technique as a trace
    if metric_mode not in ["train", "predict", "total"]:
        metric_mode = "total"

    bubble_fig = go.Figure()
    for idx, tech in enumerate(techniques):
        x_vals = []
        y_vals = []
        sizes = []
        hover_text = []

        for d in payload["datasets"]:
            ds_name = d["name"]
            nseries = d.get("nSeries", 0) or 0
            length = d.get("length", 0) or 0
            if nseries <= 0 or length <= 0:
                continue

            rec = next((r for r in timing_records if r["ds"] == ds_name and r["technique"] == tech), None)
            if not rec:
                continue

            if metric_mode == "train":
                val = rec.get("trainT", 0)
            elif metric_mode == "predict":
                val = rec.get("testT", 0)
            else:
                val = rec.get("trainT", 0) + rec.get("testT", 0)

            x_vals.append(nseries)
            y_vals.append(length)
            sizes.append(max(10, val * 8))
            hover_text.append(f"{ds_name} / {tech}: {val:.3f}s")

        if x_vals:
            bubble_fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=tech,
                marker=dict(size=sizes, sizemode="area", color=palette[idx % len(palette)], opacity=0.7, line=dict(width=1, color="black")),
                text=hover_text,
                hoverinfo="text",
            ))

    bubble_fig.update_layout(
        title="Dataset size vs series length (bubble=time, all datasets)",
        xaxis_title="Number of time series",
        yaxis_title="Series length",
        legend_title="Technique",
    )

    # Profile chart with metric_mode control
    profile_fig = go.Figure()
    sorted_datasets = sorted(payload["datasets"], key=lambda x: x.get("nSeries", 0))
    labels = [f"{d['name']} (N={d.get('nSeries', 0)})" for d in sorted_datasets]

    use_train = metric_mode in ["train", "total"]
    use_predict = metric_mode in ["predict", "total"]

    for idx, tech in enumerate(techniques):
        train_vals = [next((r["trainT"] for r in timing_records if r["ds"] == d["name"] and r["technique"] == tech), 0) for d in sorted_datasets]
        test_vals = [next((r["testT"] for r in timing_records if r["ds"] == d["name"] and r["technique"] == tech), 0) for d in sorted_datasets]

        if use_train:
            profile_fig.add_trace(go.Bar(name=f"train ({tech})", y=labels, x=train_vals, orientation="h", marker=dict(color=palette[idx % len(palette)], opacity=1.0)))
        if use_predict:
            opacity = 0.35 if metric_mode == "total" else 0.9
            profile_fig.add_trace(go.Bar(name=f"test ({tech})", y=labels, x=test_vals, orientation="h", marker=dict(color=palette[idx % len(palette)], opacity=opacity)))

    profile_title = "Dataset profile (stacked)"
    if metric_mode == "train":
        profile_title = "Dataset profile - Train time"
    elif metric_mode == "predict":
        profile_title = "Dataset profile - Predict time"

    # Siempre stacked para consistencia visual
    profile_fig.update_layout(title=profile_title, barmode="stack", xaxis_title="Time (s)")

    return bar_fig, bubble_fig, profile_fig


def build_hyperparam_figures(hp_results: list[dict[str, Any]], dataset: str):
    rows = [r for r in hp_results if r["dataset"] == dataset]
    if not rows:
        fig = go.Figure(); fig.update_layout(title="No hyperparameter data"); return [], fig, "No data"

    table_data = []
    summary_lines = []

    for method in sorted({r["method"] for r in rows}):
        method_rows = [r for r in rows if r["method"] == method]
        best = max(method_rows, key=lambda x: x.get("best_score", 0.0))
        table_data.append({
            "method": method,
            "best_score": best["best_score"],
            "elapsed_seconds": best["elapsed_seconds"],
            "best_params": json.dumps(best.get("best_params", {}), ensure_ascii=False),
        })
        summary_lines.append(f"{method}: {best['best_score']*100:.3f}% @ {best['elapsed_seconds']:.2f}s")

    scatter_fig = go.Figure()
    entries_by_method = {method: [] for method in sorted({r["method"] for r in rows})}
    for r in rows:
        for entry in r.get("all_results", []):
            entries_by_method[r["method"]].append((r.get("elapsed_seconds", 0.0), float(entry.get("mean_score", 0.0)) * 100))

    base_colors = ["rgb(56, 133, 221)", "rgb(29, 158, 117)", "rgb(216, 90, 48)", "rgb(212, 83, 126)", "rgb(186, 117, 23)", "rgb(91, 61, 153)", "rgb(74, 125, 57)", "rgb(195, 46, 75)", "rgb(42, 109, 153)", "rgb(158, 103, 18)"]
    for idx, (method, points) in enumerate(entries_by_method.items()):
        if not points:
            continue
        x_vals, y_vals = zip(*points)
        best_val = max(y_vals)
        sizes = [18 if y == best_val else 10 for y in y_vals]
        opacities = [1.0 if y == best_val else 0.5 for y in y_vals]
        color = base_colors[idx % len(base_colors)]
        scatter_fig.add_trace(go.Scatter(
            x=list(x_vals),
            y=list(y_vals),
            mode="markers",
            name=method,
            marker=dict(
                size=sizes,
                color=color,
                opacity=opacities,
                line=dict(width=1, color="black"),
            ),
        ))

    scatter_fig.update_layout(title="Elapsed vs Accuracy (HP search)", xaxis_title="Elapsed seconds", yaxis_title="Accuracy (%)")
    return table_data, scatter_fig, "\n".join(summary_lines)


def build_global_metrics_conclusion(payload: dict[str, Any]) -> str:
    df = payload.get("raw")
    if df is None or df.empty:
        return "No results available to derive global conclusions."

    # Best classifier per dataset (by accuracy)
    winners = (
        df.loc[df.groupby("dataset")["accuracy"].idxmax(), ["dataset", "classifier"]]
        .groupby("classifier")
        .size()
        .sort_values(ascending=False)
    )
    top_global = winners.index[0] if not winners.empty else "N/A"
    top_global_wins = int(winners.iloc[0]) if not winners.empty else 0

    mean_acc = df.groupby("classifier")["accuracy"].mean().sort_values(ascending=False)
    tsf_mean = float(mean_acc.get("TSF (ours)", float("nan")) * 100) if "TSF (ours)" in mean_acc else None
    tsf_rank = int(mean_acc.index.get_loc("TSF (ours)") + 1) if "TSF (ours)" in mean_acc.index else None

    lines = [
        f"Global conclusion: {top_global} is the most consistent top-accuracy classifier, winning {top_global_wins} dataset(s).",
    ]

    if tsf_mean is not None and tsf_rank is not None:
        lines.append(
            f"TSF (ours) reaches {tsf_mean:.2f}% mean accuracy overall and ranks #{tsf_rank} by average accuracy among available classifiers."
        )

    lines.append("Recommendation: use this page to select high-accuracy candidates first, then validate deployment cost in the Timing page.")
    return " ".join(lines)


def build_global_timing_conclusion(payload: dict[str, Any]) -> str:
    df = payload.get("raw")
    if df is None or df.empty:
        return "No results available to derive global timing conclusions."

    df_local = df.copy()
    df_local["total_s"] = df_local["fit_time_s"].fillna(0.0) + df_local["predict_time_s"].fillna(0.0)

    fastest = (
        df_local.loc[df_local.groupby("dataset")["total_s"].idxmin(), ["dataset", "classifier"]]
        .groupby("classifier")
        .size()
        .sort_values(ascending=False)
    )
    fastest_global = fastest.index[0] if not fastest.empty else "N/A"
    fastest_wins = int(fastest.iloc[0]) if not fastest.empty else 0

    mean_total = df_local.groupby("classifier")["total_s"].mean().sort_values(ascending=True)
    tsf_mean_total = float(mean_total.get("TSF (ours)", float("nan"))) if "TSF (ours)" in mean_total else None
    tsf_time_rank = int(mean_total.index.get_loc("TSF (ours)") + 1) if "TSF (ours)" in mean_total.index else None

    lines = [
        f"Global conclusion: {fastest_global} is the fastest overall by total time in {fastest_wins} dataset(s).",
    ]

    if tsf_mean_total is not None and tsf_time_rank is not None:
        lines.append(
            f"TSF (ours) averages {tsf_mean_total:.2f}s total runtime and ranks #{tsf_time_rank} in speed."
        )

    lines.append("Recommendation: TSF is a balanced option in several datasets, but always confirm whether its accuracy gain compensates runtime against faster baselines.")
    return " ".join(lines)


def build_global_hyperparam_conclusion(hp_results: list[dict[str, Any]]) -> str:
    if not hp_results:
        return "No hyperparameter search files found, so no global optimization conclusion can be drawn yet."

    df = pd.DataFrame(hp_results)
    if df.empty:
        return "No hyperparameter search files found, so no global optimization conclusion can be drawn yet."

    best_by_method = df.groupby("method")["best_score"].mean().sort_values(ascending=False)
    fastest_by_method = df.groupby("method")["elapsed_seconds"].mean().sort_values(ascending=True)

    best_method = str(best_by_method.index[0]) if not best_by_method.empty else "N/A"
    fastest_method = str(fastest_by_method.index[0]) if not fastest_by_method.empty else "N/A"

    return (
        f"Global conclusion: for TSF hyperparameter tuning, {best_method} gives the best average optimization score, "
        f"while {fastest_method} is the fastest search strategy on average. "
        "Recommendation: pick the best-accuracy method when quality is critical, or the fastest method for quick iteration, then retrain TSF and compare against benchmark classifiers."
    )


def _sanitize_filename(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def _image_to_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_image_block(title: str, path: Path) -> html.Div:
    uri = _image_to_data_uri(path)
    if uri is None:
        content = html.Div(f"Image not found: {path.as_posix()}", style={"color": "#a33", "fontSize": "12px"})
    else:
        content = html.Img(src=uri, style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "6px"})

    return html.Div(
        [
            html.H4(title, style={"marginBottom": "8px"}),
            content,
        ],
        style={"marginBottom": "14px"},
    )


def generate_visual_assets(
    dataset_name: str,
    classifier_name: str,
    data_dir: Path,
    predictions_dir: Path,
    viz_dir: Path,
) -> tuple[list[html.Div], str]:
    dataset_viz_dir = viz_dir / dataset_name
    classifier_viz_dir = dataset_viz_dir / _sanitize_filename(classifier_name)
    dataset_viz_dir.mkdir(parents=True, exist_ok=True)
    classifier_viz_dir.mkdir(parents=True, exist_ok=True)

    status: list[str] = []

    try:
        train_file = data_dir / dataset_name / f"{dataset_name}_TRAIN.txt"
        X_train, y_train = load_ucr_txt_dataset(train_file)
        generate_dataset_graph(
            X_train,
            dataset_name=dataset_name,
            labels=y_train,
            max_series=16,
            save=True,
            out_dir=dataset_viz_dir,
        )
        status.append("TS plots generated")
    except Exception as exc:
        status.append(f"TS plots error: {type(exc).__name__}: {exc}")

    pred_file = predictions_dir / dataset_name / f"{_sanitize_filename(classifier_name)}.csv"
    try:
        if pred_file.exists():
            plot_overlay_by_correctness(
                dataset_name=dataset_name,
                predictions_csv=pred_file,
                data_dir=data_dir,
                split="TEST",
                max_series=60,
                save=True,
                out_dir=classifier_viz_dir,
            )
            _, y_true, y_pred = load_predictions_csv(pred_file)
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                normalize=False,
                save=True,
                out_dir=classifier_viz_dir,
                dataset_name=dataset_name,
            )
            status.append("Prediction plots generated")
        else:
            status.append(f"Predictions CSV not found: {pred_file}")
    except Exception as exc:
        status.append(f"Prediction plots error: {type(exc).__name__}: {exc}")

    blocks = [
        _build_image_block("Dataset grid (visualize_TS)", dataset_viz_dir / f"{dataset_name}.png"),
        _build_image_block("Dataset overlay (visualize_TS)", dataset_viz_dir / f"{dataset_name}_overlay.png"),
        _build_image_block("Prediction overlay (visualize_predictions)", classifier_viz_dir / f"{dataset_name}_overlay_correctness.png"),
        _build_image_block("Confusion matrix (visualize_predictions)", classifier_viz_dir / f"{dataset_name}_confusion_matrix.png"),
    ]

    return blocks, " | ".join(status)


def generate_prediction_visual_assets(
    dataset_name: str,
    classifier_name: str,
    data_dir: Path,
    predictions_dir: Path,
    viz_dir: Path,
) -> tuple[list[html.Div], str]:
    dataset_viz_dir = viz_dir / dataset_name
    classifier_viz_dir = dataset_viz_dir / _sanitize_filename(classifier_name)
    classifier_viz_dir.mkdir(parents=True, exist_ok=True)

    status: list[str] = []
    pred_file = predictions_dir / dataset_name / f"{_sanitize_filename(classifier_name)}.csv"

    try:
        if pred_file.exists():
            plot_overlay_by_correctness(
                dataset_name=dataset_name,
                predictions_csv=pred_file,
                data_dir=data_dir,
                split="TEST",
                max_series=60,
                save=True,
                out_dir=classifier_viz_dir,
            )
            _, y_true, y_pred = load_predictions_csv(pred_file)
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                normalize=False,
                save=True,
                out_dir=classifier_viz_dir,
                dataset_name=dataset_name,
            )
            status.append("Prediction plots generated")
        else:
            status.append(f"Predictions CSV not found: {pred_file}")
    except Exception as exc:
        status.append(f"Prediction plots error: {type(exc).__name__}: {exc}")

    blocks = [
        _build_image_block(
            "Prediction overlay (visualize_predictions)",
            classifier_viz_dir / f"{dataset_name}_overlay_correctness.png",
        ),
        _build_image_block(
            "Confusion matrix (visualize_predictions)",
            classifier_viz_dir / f"{dataset_name}_confusion_matrix.png",
        ),
    ]
    return blocks, " | ".join(status)


def generate_dataset_visual_assets(
    dataset_name: str,
    data_dir: Path,
    viz_dir: Path,
    label_filter: Any = "__all__",
    full_individual: bool = False,
) -> tuple[html.Div, html.Div, str]:
    dataset_viz_dir = viz_dir / dataset_name
    dataset_viz_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / dataset_name / f"{dataset_name}_TRAIN.txt"
    if not train_file.exists():
        empty = html.Div(f"Dataset split file not found: {train_file.as_posix()}", style={"color": "#a33", "fontSize": "12px"})
        return empty, empty, "Dataset file missing"

    try:
        X_train, y_train = load_ucr_txt_dataset(train_file)
        include_labels = None if label_filter == "__all__" else [label_filter]
        preview_n = min(32, int(X_train.shape[0]))
        max_series = int(X_train.shape[0]) if full_individual else preview_n
        generate_dataset_graph(
            X_train,
            dataset_name=dataset_name,
            labels=y_train,
            include_labels=include_labels,
            max_series=max_series,
            save=True,
            out_dir=dataset_viz_dir,
        )
        selected_label_txt = "all labels" if label_filter == "__all__" else f"label={label_filter}"
        mode = "full individual plots" if full_individual else f"preview ({preview_n} series)"
        status = f"TS plots generated: {mode}, {selected_label_txt}"
    except Exception as exc:
        empty = html.Div(f"Error generating dataset plots: {type(exc).__name__}: {exc}", style={"color": "#a33", "fontSize": "12px"})
        return empty, empty, f"TS plots error: {type(exc).__name__}: {exc}"

    overlay_block = _build_image_block(
        "Dataset overlay (visualize_TS)",
        dataset_viz_dir / f"{dataset_name}_overlay.png",
    )
    grid_block = _build_image_block(
        "Individual time series (visualize_TS)",
        dataset_viz_dir / f"{dataset_name}.png",
    )
    return overlay_block, grid_block, status


def collect_prediction_availability(predictions_dir: Path) -> dict[str, set[str]]:
    """Map dataset -> available sanitized classifier names from predictions CSV files."""
    availability: dict[str, set[str]] = {}
    if not predictions_dir.exists():
        return availability

    for ds_dir in predictions_dir.iterdir():
        if not ds_dir.is_dir():
            continue
        csv_names = {
            p.stem for p in ds_dir.glob("*.csv")
        }
        availability[ds_dir.name] = csv_names
    return availability


def collect_dataset_labels(data_dir: Path, dataset_name: str) -> list[Any]:
    train_file = data_dir / dataset_name / f"{dataset_name}_TRAIN.txt"
    if not train_file.exists():
        return []
    try:
        _X, y = load_ucr_txt_dataset(train_file)
    except Exception:
        return []
    # Preserve order of first appearance
    return list(dict.fromkeys(y.tolist()))


def create_dash_app(
    results_csv: Path,
    data_dir: Path,
    hp_dir: Path,
    predictions_dir: Path,
    viz_dir: Path,
):
    payload = collect_results(results_csv, data_dir)
    hp_results = collect_hyperparameter_results(hp_dir)
    metrics_conclusion = build_global_metrics_conclusion(payload)
    timing_conclusion = build_global_timing_conclusion(payload)
    hp_conclusion = build_global_hyperparam_conclusion(hp_results)
    prediction_availability = collect_prediction_availability(predictions_dir)
    dataset_names = [d["name"] for d in payload["datasets"]]
    datasets_with_predictions = [d for d in dataset_names if prediction_availability.get(d)]
    default_viz_dataset = datasets_with_predictions[0] if datasets_with_predictions else (dataset_names[0] if dataset_names else None)

    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div([
        html.Div([html.H1("Time Series Classifier Dashboard (3 pages)")], className="header"),
        dcc.Tabs(id="tabs", value="tab-metrics", children=[
            dcc.Tab(label="Metrics", value="tab-metrics"),
            dcc.Tab(label="Timing", value="tab-timing"),
            dcc.Tab(label="Hyperparameter", value="tab-hyperparam"),
            dcc.Tab(label="Datasets", value="tab-datasets"),
        ], className="tabs-bar"),
        html.Div(id="tab-content", style={"marginTop": "20px", "padding": "12px"}),
        html.Div([
            html.Hr(),
            html.P("This dashboard provides interactive analysis of classifier performance, timing, and hyperparameter search results for time series datasets."),
            html.P("Created by Ane Miren Arregi, Iker Bereziartua and Eneko Zabaleta as part of ASTD Project 3."),
        ], style={"marginTop": "40px", "fontSize": "12px", "color": "#666", "textAlign": "center"}),
    ], style={"background": "#f4f4f3"})

    button_style = {
        "padding": "10px 16px",
        "fontSize": "14px",
        "fontWeight": "600",
        "borderRadius": "8px",
        "border": "1px solid #b9b9b9",
        "background": "#ffffff",
        "cursor": "pointer",
    }

    @app.callback(Output("tab-content", "children"), [Input("tabs", "value")])
    def render_tab(tab):
        if tab == "tab-metrics":
            ds_opts = [{"label": d["name"], "value": d["name"]} for d in payload["datasets"]]
            return html.Div([
                html.Div([html.P("Metrics tab provides global accuracy trends first, then a selected-dataset breakdown with data size details." )], style={"marginBottom": "10px", "color": "#444"}),
                dcc.Graph(id="metrics-line"),
                html.Div([
                    html.Div([
                        html.Label("Dataset"),
                        dcc.Dropdown(id="metrics-dataset", options=ds_opts, value=ds_opts[0]["value"], clearable=False),
                        html.Div(id="metrics-dataset-info", style={"marginTop": "10px", "marginBottom": "8px", "color": "#444"}),
                        html.Div(id="metrics-summary", style={"marginTop": "4px"}),
                    ], style={"flex": "0 0 32%", "paddingRight": "12px"}),
                    html.Div([
                        dcc.Graph(id="metrics-bar"),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "alignItems": "flex-start"}),
                html.Div(
                    metrics_conclusion,
                    style={"marginTop": "14px", "padding": "12px", "background": "#f9f9f9", "border": "1px solid #ddd", "borderRadius": "8px", "color": "#333"},
                ),
                html.Div([
                    html.Div([
                        html.Label("Classifier"),
                        dcc.Dropdown(id="metrics-classifier", clearable=False),
                    ], style={"width": "38%", "display": "inline-block", "verticalAlign": "top"}),
                    html.Div(id="metrics-pred-status", style={"width": "58%", "display": "inline-block", "marginLeft": "4%", "marginTop": "22px", "color": "#444", "fontSize": "12px"}),
                ], style={"marginTop": "12px", "marginBottom": "8px"}),
                html.Div(id="metrics-pred-images"),
            ])
        if tab == "tab-timing":
            ds_opts = [{"label": d["name"], "value": d["name"]} for d in payload["datasets"]]
            metric_opts = [{"label": "Train", "value": "train"}, {"label": "Predict", "value": "predict"}, {"label": "Total", "value": "total"}]
            return html.Div([
                html.Div([html.P("Timing tab shows model training and prediction timings for the selected dataset. Use metric selector to switch between train/predict/total timing charts." )], style={"marginBottom": "10px", "color": "#444"}),
                html.Div([
                    html.Div([html.Label("Dataset"), dcc.Dropdown(id="timing-dataset", options=ds_opts, value=ds_opts[0]["value"], clearable=False)], style={"width": "45%", "display": "inline-block"}),
                    html.Div([html.Label("Metric"), dcc.Dropdown(id="timing-metric", options=metric_opts, value="total", clearable=False)], style={"width": "45%", "display": "inline-block", "marginLeft": "20px"}),
                ]),
                html.Div([html.Button("Mostrar/bloquear otros gráficos", id="toggle-graphs", n_clicks=0, style=button_style)], style={"margin":"10px 0"}),
                html.Div([html.P("Bubble chart: each point represents a dataset/technique pair. X=dataset size, Y=series length, size=time depending on selected metric.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"})]),
                dcc.Graph(id="timing-bubble"),
                html.Div([
                    html.Div([html.P("Stacked horizontal bar chart: technique timings according to selected metric.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"}), dcc.Graph(id="timing-bar", style={"height": "480px"})], style={"flex": "1", "paddingRight": "10px"}),
                    html.Div([html.P("Profile chart: per-dataset total or individual timing bars by technique.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"}), dcc.Graph(id="timing-profile", style={"height": "480px"})], style={"flex": "1", "paddingLeft": "10px"}),
                ], id="timing-extra-graphs", style={"display":"flex", "flexDirection":"row", "alignItems": "flex-start"}),
                html.Div(
                    timing_conclusion,
                    style={"marginTop": "14px", "padding": "12px", "background": "#f9f9f9", "border": "1px solid #ddd", "borderRadius": "8px", "color": "#333"},
                ),
            ])
        ds_opts = [{"label": d["name"], "value": d["name"]} for d in payload["datasets"]]
        if tab == "tab-hyperparam":
            return html.Div([
                html.Div([html.P("Hyperparameter tab visualizes best obtained score and elapsed time for optimization methods. Use this view to compare search quality and training speed.")], style={"marginBottom": "10px", "color": "#444"}),
                html.Div([html.Label("Dataset"), dcc.Dropdown(id="hp-dataset", options=ds_opts, value=ds_opts[0]["value"], clearable=False)], style={"width": "30%", "marginBottom": "10px"}),
                dash_table.DataTable(id="hp-table", columns=[{"name": "Method", "id": "method"}, {"name": "Best Score", "id": "best_score"}, {"name": "Elapsed s", "id": "elapsed_seconds"}, {"name": "Best params", "id": "best_params"}], style_table={"overflowX": "auto"}, style_cell={"textAlign": "left", "fontSize": "13px"}),
                html.Div([html.P("Scatter chart: each point is an hyperparameter trial; x=elapsed time, y=accuracy; bigger points are best accuracy for method.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"})]),
                dcc.Graph(id="hp-scatter"),
                html.Pre(id="hp-summary", style={"whiteSpace": "pre-wrap", "marginTop": "12px"}),
                html.Div(
                    hp_conclusion,
                    style={"marginTop": "14px", "padding": "12px", "background": "#f9f9f9", "border": "1px solid #ddd", "borderRadius": "8px", "color": "#333"},
                ),
            ])

        return html.Div([
            html.Div([html.P("Datasets tab embeds outputs from visualize_TS.py for selected dataset." )], style={"marginBottom": "10px", "color": "#444"}),
            html.Div([
                html.Div([html.Label("Dataset"), dcc.Dropdown(id="datasets-dataset", options=ds_opts, value=default_viz_dataset, clearable=False)], style={"width": "40%", "display": "inline-block"}),
                html.Div([html.Button("Show all individual time series", id="datasets-plot-all", n_clicks=0, style=button_style)], style={"width": "40%", "display": "inline-block", "marginLeft": "20px", "paddingTop": "22px"}),
            ], style={"marginBottom": "12px"}),
            html.Div([
                html.Label("Label filter"),
                dcc.RadioItems(
                    id="datasets-label-filter",
                    inline=True,
                    inputStyle={"transform": "scale(1.45)", "marginRight": "8px"},
                    labelStyle={
                        "display": "inline-flex",
                        "alignItems": "center",
                        "marginRight": "20px",
                        "fontSize": "16px",
                        "fontWeight": "600",
                    },
                ),
            ], style={"marginBottom": "10px", "color": "#444", "fontSize": "12px"}),
            html.Div("Plots are displayed below this control panel.", style={"marginBottom": "8px", "color": "#555", "fontSize": "12px"}),
            html.Div(id="datasets-status", style={"marginBottom": "10px", "color": "#444", "fontSize": "12px"}),
            html.Div([
                html.Div(id="datasets-overlay-image", style={"flex": "1.25"}),
                html.Div(id="datasets-individual-image", style={"display": "none", "flex": "1"}),
            ], style={"display": "flex", "gap": "12px", "alignItems": "flex-start"}),
        ])

    @app.callback(
        [Output("datasets-label-filter", "options"), Output("datasets-label-filter", "value")],
        [Input("datasets-dataset", "value")],
    )
    def update_dataset_label_options(dataset_value):
        labels = collect_dataset_labels(data_dir, dataset_value)
        options = [{"label": "All", "value": "__all__"}] + [
            {"label": str(lbl), "value": lbl} for lbl in labels
        ]
        return options, "__all__"

    @app.callback(
        [Output("metrics-classifier", "options"), Output("metrics-classifier", "value")],
        [Input("metrics-dataset", "value")],
    )
    def update_metrics_classifier_options(dataset_value):
        available = prediction_availability.get(dataset_value, set())
        all_classifiers = payload["classifiers"]
        preferred = [c for c in all_classifiers if _sanitize_filename(c) in available]
        if preferred:
            return [{"label": c, "value": c} for c in preferred], preferred[0]
        return [{"label": c, "value": c} for c in all_classifiers], all_classifiers[0] if all_classifiers else None

    @app.callback(
        [
            Output("metrics-bar", "figure"),
            Output("metrics-line", "figure"),
            Output("metrics-summary", "children"),
            Output("metrics-dataset-info", "children"),
            Output("metrics-pred-images", "children"),
            Output("metrics-pred-status", "children"),
        ],
        [Input("metrics-dataset", "value"), Input("metrics-classifier", "value")],
    )
    def update_metrics(dataset_value, classifier_value):
        fig_bar, fig_line, summary = build_metrics_figures(payload, dataset_value)
        summary_el = html.Div([html.P(f"Best classifier: {summary['best']}"), html.P(f"Worst classifier: {summary['worst']}"), html.P(f"Mean accuracy: {summary['mean']}")])
        ds_stats = next((d for d in payload["datasets"] if d["name"] == dataset_value), None)
        if ds_stats is None:
            dataset_info = html.P("Dataset details not available")
        else:
            n_train = ds_stats.get("nTrain")
            n_test = ds_stats.get("nTest")
            n_series = ds_stats.get("nSeries")
            length = ds_stats.get("length")
            dataset_info = html.P(
                f"{dataset_value}: train={n_train}, test={n_test}, total={n_series}, series length={length}",
            )

        pred_images: Any = []
        pred_status = ""
        if classifier_value:
            pred_blocks, pred_status = generate_prediction_visual_assets(
                dataset_name=dataset_value,
                classifier_name=classifier_value,
                data_dir=data_dir,
                predictions_dir=predictions_dir,
                viz_dir=viz_dir,
            )
            available = prediction_availability.get(dataset_value, set())
            if _sanitize_filename(classifier_value) not in available:
                pred_status = (
                    pred_status
                    + " | No prediction CSV for this dataset/classifier pair. "
                    + "Run: python experiments/main_run.py --mode predict --datasets "
                    + dataset_value
                )

            pred_images = html.Div(
                [
                    html.Div(pred_blocks[0], style={"flex": "1.25"}),
                    html.Div(pred_blocks[1], style={"flex": "0.75"}),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "start",
                },
            )

        return fig_bar, fig_line, summary_el, dataset_info, pred_images, pred_status

    @app.callback(
        [
            Output("timing-bar", "figure"),
            Output("timing-bubble", "figure"),
            Output("timing-profile", "figure"),
        ],
        [Input("timing-dataset", "value"), Input("timing-metric", "value")]
    )
    def update_timing(dataset_value, metric_value):
        return build_timing_figures(payload, dataset_value, metric_value)

    @app.callback(
        Output("timing-extra-graphs", "style"),
        [Input("toggle-graphs", "n_clicks")],
        prevent_initial_call=True
    )
    def toggle_timing_extra(n_clicks):
        if n_clicks % 2 == 0:
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        [
            Output("hp-table", "data"),
            Output("hp-scatter", "figure"),
            Output("hp-summary", "children"),
        ],
        [Input("hp-dataset", "value")],
    )
    def update_hp(dataset_value):
        table_data, scatter_fig, summary_text = build_hyperparam_figures(hp_results, dataset_value)
        return table_data, scatter_fig, summary_text

    @app.callback(
        [Output("datasets-overlay-image", "children"), Output("datasets-status", "children")],
        [Input("datasets-dataset", "value"), Input("datasets-label-filter", "value")],
    )
    def update_dataset_overlay(dataset_value, label_filter):
        overlay_block, _grid_block, status = generate_dataset_visual_assets(
            dataset_name=dataset_value,
            data_dir=data_dir,
            viz_dir=viz_dir,
            label_filter=label_filter,
            full_individual=False,
        )
        return [overlay_block], status

    @app.callback(
        [
            Output("datasets-individual-image", "children"),
            Output("datasets-individual-image", "style"),
            Output("datasets-plot-all", "children"),
        ],
        [
            Input("datasets-plot-all", "n_clicks"),
            Input("datasets-dataset", "value"),
            Input("datasets-label-filter", "value"),
        ],
    )
    def update_dataset_individual(n_clicks, dataset_value, label_filter):
        if not n_clicks or n_clicks % 2 == 0:
            return [], {"display": "none"}, "Show all individual time series"

        _overlay_block, grid_block, _status = generate_dataset_visual_assets(
            dataset_name=dataset_value,
            data_dir=data_dir,
            viz_dir=viz_dir,
            label_filter=label_filter,
            full_individual=True,
        )
        return [grid_block], {"display": "block", "flex": "1"}, "Hide individual time series"

    return app


def main():
    parser = argparse.ArgumentParser(description="Run Dash dashboard server")
    parser.add_argument("--results", default="results/benchmark_comparison.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--hp-dir", default="results")
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--viz-dir", default="visualization")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_dash_app(
        Path(args.results),
        Path(args.data_dir),
        Path(args.hp_dir),
        Path(args.predictions_dir),
        Path(args.viz_dir),
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
