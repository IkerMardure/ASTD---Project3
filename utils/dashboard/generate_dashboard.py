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


def collect_results(results_csv: Path, data_dir: Path, wilcoxon_csv: Path | None = None) -> dict[str, Any]:
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
                "f1_weighted": float(row.get("f1_weighted", 0.0) or 0.0),
                "precision_weighted": float(row.get("precision_weighted", 0.0) or 0.0),
                "recall_weighted": float(row.get("recall_weighted", 0.0) or 0.0),
                "balanced_accuracy": float(row.get("balanced_accuracy", 0.0) or 0.0),
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

    # Load Wilcoxon results if available
    wilcoxon_data = None
    if wilcoxon_csv is not None:
        wilcoxon_data = load_wilcoxon_csv(wilcoxon_csv)

    return {
        "datasets": dataset_stats,
        "classifiers": classifiers,
        "metrics_by_dataset": metrics_by_dataset,
        "raw": df,
        "timing_records": records,
        "wilcoxon_results": wilcoxon_data,
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


def load_wilcoxon_csv(wilcoxon_csv: Path) -> pd.DataFrame | None:
    """Load Wilcoxon test results CSV if available."""
    if not wilcoxon_csv.exists():
        return None
    try:
        df = pd.read_csv(wilcoxon_csv)
        return df
    except Exception:
        return None


def get_metric_metadata() -> dict[str, dict[str, str]]:
    """Return metadata for available metrics (labels, formatters, etc.)."""
    return {
        "accuracy": {
            "label": "Accuracy",
            "unit": "%",
            "range": [0, 100],
            "higher_is_better": True,
        },
        "f1_weighted": {
            "label": "F1-weighted",
            "unit": "",
            "range": [0, 1],
            "higher_is_better": True,
        },
        "precision_weighted": {
            "label": "Precision-weighted",
            "unit": "",
            "range": [0, 1],
            "higher_is_better": True,
        },
        "recall_weighted": {
            "label": "Recall-weighted",
            "unit": "",
            "range": [0, 1],
            "higher_is_better": True,
        },
        "balanced_accuracy": {
            "label": "Balanced Accuracy",
            "unit": "",
            "range": [0, 1],
            "higher_is_better": True,
        },
    }


def format_params_dict(params_dict: dict[str, Any]) -> str:
    """Format hyperparameter dict to readable key=value string."""
    if not params_dict:
        return "(no parameters)"
    
    items = []
    for k, v in sorted(params_dict.items()):
        if isinstance(v, float):
            items.append(f"{k}={v:.4g}")
        else:
            items.append(f"{k}={v}")
    return ", ".join(items)


def select_series_by_mode(
    X: Any, y: Any, mode: str, n_series: int, random_state: int = 42
) -> tuple[Any, Any]:
    """
    Select time series from dataset X, y by selected mode.
    
    Args:
        X: numpy array of shape (n_samples, n_timestamps)
        y: numpy array of shape (n_samples,) with class labels
        mode: "all" (first n_series), "balanced" (equal per class), "random" (shuffle & sample)
        n_series: max number of series to return (-1 means all)
        random_state: seed for reproducibility
    
    Returns:
        (X_selected, y_selected) tuple
    """
    import numpy as np
    
    if n_series == -1:
        n_series = len(X)
    
    if mode == "balanced":
        classes = np.unique(y)
        per_class = max(1, n_series // len(classes))
        indices = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            count = min(per_class, len(cls_idx))
            indices.extend(cls_idx[:count])
        indices = indices[:n_series]
        return X[indices], y[indices]
    
    elif mode == "random":
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), min(n_series, len(X)), replace=False)
        return X[idx], y[idx]
    
    else:  # "all" (sequential first n_series)
        return X[:n_series], y[:n_series]


def build_wilcoxon_panel(wilcoxon_df: pd.DataFrame | None, metric_name: str = "f1_weighted") -> html.Div:
    """Build Wilcoxon statistical test results panel."""
    if wilcoxon_df is None or wilcoxon_df.empty:
        return html.Div(
            html.P("Wilcoxon test results not available. Run benchmark with statistic mode enabled.", style={"color": "#999"}),
            style={"padding": "12px", "background": "#f5f5f5", "borderRadius": "8px", "border": "1px dashed #ddd"}
        )
    
    # Filter for the current metric if available
    if "metric" in wilcoxon_df.columns:
        df_filtered = wilcoxon_df[wilcoxon_df.get("metric", "f1_weighted") == metric_name].copy() if metric_name in wilcoxon_df["metric"].values else wilcoxon_df.copy()
    else:
        df_filtered = wilcoxon_df.copy()
    
    if df_filtered.empty:
        return html.Div(
            html.P(f"No Wilcoxon results for metric '{metric_name}'", style={"color": "#999"}),
            style={"padding": "12px", "background": "#f5f5f5", "borderRadius": "8px", "border": "1px dashed #ddd"}
        )
    
    # Build table data
    table_data = []
    for _, row in df_filtered.iterrows():
        p_value = float(row.get("p_value", 1.0))
        is_significant = bool(row.get("significant", False))
        table_data.append({
            "Baseline": str(row.get("classifier", "?")),
            "p-value": f"{p_value:.4f}",
            "Significant": "✓ Yes" if is_significant else "✗ No",
            "Mean Δ": f"{float(row.get('mean_delta', 0)):.4f}",
            "TSF Better": int(row.get("candidate_better_count", 0)),
            "Baseline Better": int(row.get("reference_better_count", 0)),
        })
    
    # Build styled table
    table_children = [
        html.H4("Wilcoxon Test Results (TSF vs Baselines)", style={"marginBottom": "10px", "fontSize": "14px", "fontWeight": "600"}),
        html.P(f"Paired Wilcoxon signed-rank test (α=0.05, metric: {metric_name})", style={"fontSize": "12px", "color": "#666", "marginBottom": "10px"}),
        dash_table.DataTable(
            data=table_data,
            columns=[{"name": c, "id": c} for c in table_data[0].keys()] if table_data else [],
            style_table={"overflowX": "auto", "marginBottom": "8px"},
            style_cell={"padding": "8px", "fontSize": "12px", "textAlign": "left"},
            style_data_conditional=[
                {
                    "if": {"column_id": "Significant", "filter_query": "{Significant} contains 'Yes'"},
                    "backgroundColor": "#d4edda",
                    "color": "#155724",
                    "fontWeight": "600",
                },
                {
                    "if": {"column_id": "Significant", "filter_query": "{Significant} contains 'No'"},
                    "backgroundColor": "#f8f9fa",
                    "color": "#666",
                },
            ],
            style_header={
                "backgroundColor": "#f5f5f5",
                "fontWeight": "600",
                "border": "1px solid #ddd",
                "fontSize": "12px",
            },
            style_cell_conditional=[
                {"if": {"column_id": "TSF Better"}, "textAlign": "center"},
                {"if": {"column_id": "Baseline Better"}, "textAlign": "center"},
            ],
        ),
    ]
    
    return html.Div(
        table_children,
        style={"marginTop": "12px", "padding": "12px", "background": "#fafaf8", "border": "1px solid #e8e7e3", "borderRadius": "8px"}
    )


def build_metrics_delta_chart(payload: dict[str, Any], dataset: str, metric_name: str = "accuracy") -> go.Figure:
    """Build delta chart showing performance gap vs TSF (ours)."""
    metric_values = {}
    dataset_metrics = payload["metrics_by_dataset"].get(dataset, {})
    
    for cls, data in dataset_metrics.items():
        val = data.get(metric_name, 0.0)
        metric_values[cls] = val * 100
    
    tsf_value = metric_values.get("TSF (ours)", 0.0)
    if tsf_value == 0:
        return go.Figure().add_annotation(text="TSF not available in dataset")
    
    # Calculate deltas (negative = worse, positive = better than TSF)
    deltas = {cls: val - tsf_value for cls, val in metric_values.items() if cls != "TSF (ours)"}
    
    if not deltas:
        empty = go.Figure()
        empty.update_layout(title="No baseline classifiers to compare")
        return empty
    
    bar_colors = ["#2E8B57" if v >= 0 else "#C0392B" for v in deltas.values()]
    fig = go.Figure(go.Bar(
        x=list(deltas.keys()),
        y=list(deltas.values()),
        marker_color=bar_colors,
        text=[f"{v:+.2f}" for v in deltas.values()],
        textposition="auto",
    ))
    fig.update_layout(
        title="Performance gap vs TSF (ours)",
        yaxis_title="Δ (%)",
        xaxis_title="Baseline Classifier",
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


def build_insights_summary(payload: dict[str, Any], dataset: str, metric_name: str = "accuracy") -> html.Div:
    """Generate top insights for current dataset and metric."""
    dataset_metrics = payload["metrics_by_dataset"].get(dataset, {})
    if not dataset_metrics:
        return html.Div("No data available for insights.", style={"color": "#999", "fontSize": "13px"})
    
    metric_values = {cls: data.get(metric_name, 0.0) * 100 for cls, data in dataset_metrics.items()}
    
    best_cls = max(metric_values, key=lambda x: metric_values[x])
    worst_cls = min(metric_values, key=lambda x: metric_values[x])
    tsf_value = metric_values.get("TSF (ours)")
    
    insights = []
    
    # Insight 1: Best performer
    insights.append(f"✓ Best: {best_cls} ({metric_values[best_cls]:.2f}%)")
    
    # Insight 2: TSF rank
    if tsf_value is not None:
        sorted_metrics = sorted(metric_values.values(), reverse=True)
        rank = sorted_metrics.index(tsf_value) + 1
        insights.append(f"• TSF ranks #{rank}/{len(metric_values)} on {metric_name}")
        
        margin_vs_best = metric_values[best_cls] - tsf_value
        if margin_vs_best > 1:
            insights.append(f"  Gap to best: {margin_vs_best:.2f}%")
        else:
            insights.append(f"  TSF is competitive (gap: {margin_vs_best:.2f}%)")
    
    # Insight 3: Performance range
    perf_range = metric_values[best_cls] - metric_values[worst_cls]
    insights.append(f"• Performance range: {perf_range:.2f}% ({worst_cls} vs {best_cls})")
    
    return html.Div([
        html.H5("Top Insights", style={"marginBottom": "8px", "fontSize": "13px", "fontWeight": "600"}),
        html.Ul([html.Li(insight, style={"fontSize": "12px", "marginBottom": "4px"}) for insight in insights], style={"paddingLeft": "20px"})
    ], style={"padding": "10px", "background": "#f5f5f5", "borderRadius": "6px", "marginTop": "10px"})


def build_metrics_figures(payload: dict[str, Any], dataset: str, metric_name: str = "accuracy"):
    """Build metrics bar and line charts for selected metric."""
    metric_meta = get_metric_metadata()
    meta = metric_meta.get(metric_name, metric_meta["accuracy"])
    
    dataset_metrics = payload["metrics_by_dataset"].get(dataset, {})
    if not dataset_metrics:
        return go.Figure(), go.Figure(), {"best": "-", "worst": "-", "mean": "-"}

    # Normalize metric value: all metrics displayed as 0-100
    metric_values = {}
    for cls, data in dataset_metrics.items():
        val = data.get(metric_name, 0.0)
        # All metrics are scaled to 0-100 for consistent display
        metric_values[cls] = val * 100
    
    best_cls = max(metric_values, key=lambda x: metric_values[x])
    worst_cls = min(metric_values, key=lambda x: metric_values[x])

    bar_colors = []
    for cls in metric_values.keys():
        if cls == best_cls:
            bar_colors.append("#2E8B57")
        elif cls == worst_cls:
            bar_colors.append("#C0392B")
        else:
            bar_colors.append("#4C78A8")

    bar_fig = go.Figure(go.Bar(x=list(metric_values.keys()), y=list(metric_values.values()), marker_color=bar_colors))
    bar_fig.update_layout(
        title=f"{meta['label']} for {dataset}",
        xaxis_title="Classifier",
        yaxis_title=f"{meta['label']} (%)",
        yaxis=dict(range=[0, 100])
    )

    datasets_list = [d["name"] for d in payload["datasets"]]
    line_series = []
    for cls in payload["classifiers"]:
        y_vals = []
        for ds in datasets_list:
            val = payload["metrics_by_dataset"].get(ds, {}).get(cls, {}).get(metric_name, 0.0)
            # All metrics scaled to 0-100
            y_vals.append(val * 100)
        line_series.append(go.Scatter(x=datasets_list, y=y_vals, mode="lines+markers", name=cls))

    line_fig = go.Figure(line_series)
    line_fig.update_layout(
        title=f"{meta['label']} across datasets",
        xaxis_title="Dataset",
        yaxis_title=f"{meta['label']} (%)",
        yaxis=dict(range=[0, 100])
    )

    mean_val = sum(metric_values.values()) / len(metric_values) if metric_values else 0

    summary = {
        "best": f"{best_cls} ({metric_values[best_cls]:.2f}%)" if metric_values else "-",
        "worst": f"{worst_cls} ({metric_values[worst_cls]:.2f}%)" if metric_values else "-",
        "mean": f"{mean_val:.2f}%" if metric_values else "-",
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
            "best_score": f"{best['best_score']*100:.2f}%",
            "elapsed_seconds": f"{best['elapsed_seconds']:.2f}",
            "best_params": format_params_dict(best.get("best_params", {})),
        })
        summary_lines.append(f"{method}: {best['best_score']*100:.2f}% @ {best['elapsed_seconds']:.2f}s")

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
        f"📊 {top_global} is the most consistent high-accuracy classifier, winning on {top_global_wins} dataset(s).",
    ]

    if tsf_mean is not None and tsf_rank is not None:
        lines.append(
            f"✓ TSF (our method) achieves {tsf_mean:.2f}% mean accuracy and ranks #{tsf_rank} among all classifiers."
        )

    lines.append("💡 Recommendation: Identify high-accuracy candidates here, then verify their computational cost in the Timing page before deployment.")
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
        f"⚡ {fastest_global} is the fastest overall, achieving minimum execution time in {fastest_wins} dataset(s).",
    ]

    if tsf_mean_total is not None and tsf_time_rank is not None:
        lines.append(
            f"✓ TSF (our method) averages {tsf_mean_total:.2f}s total runtime and ranks #{tsf_time_rank} in speed."
        )

    lines.append("💡 Recommendation: TSF offers balanced speed and accuracy in many datasets. Compare its runtime against baseline methods to ensure the accuracy gain justifies computational cost.")
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
        f"🎯 For TSF hyperparameter tuning: {best_method} delivers the best optimization scores on average, "
        f"while {fastest_method} is the fastest search strategy. "
        "💡 Choose accuracy-focused methods for critical deployments or time-efficient methods for rapid prototyping. After tuning, validate the optimized TSF against benchmark classifiers."
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
    n_series: int = 16,
    selection_mode: str = "all",
    random_state: int = 42,
) -> tuple[html.Div, html.Div, str, html.Div]:
    dataset_viz_dir = viz_dir / dataset_name
    dataset_viz_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / dataset_name / f"{dataset_name}_TRAIN.txt"
    if not train_file.exists():
        empty = html.Div(f"Dataset split file not found: {train_file.as_posix()}", style={"color": "#a33", "fontSize": "12px"})
        return empty, empty, "Dataset file missing", empty

    try:
        import numpy as np
        X_train, y_train = load_ucr_txt_dataset(train_file)
        
        # Store original data for info generation
        X_train_orig = X_train.copy()
        y_train_orig = y_train.copy()
        
        # Apply label filter if specified
        if label_filter != "__all__":
            mask = y_train == label_filter
            X_train = X_train[mask]
            y_train = y_train[mask]
        
        include_labels = None if label_filter == "__all__" else [label_filter]
        
        # Determine how many series to plot
        if full_individual or n_series == -1:
            target_n_series = len(X_train)
        elif n_series > 0:
            target_n_series = min(n_series, len(X_train))
        else:
            target_n_series = min(32, len(X_train))
        
        # Apply selection mode (sequential, balanced, random)
        X_selected, y_selected = select_series_by_mode(X_train, y_train, selection_mode, target_n_series, random_state=random_state)
        
        generate_dataset_graph(
            X_selected,
            dataset_name=dataset_name,
            labels=y_selected,
            include_labels=None,  # Already filtered above
            max_series=len(X_selected),
            save=True,
            out_dir=dataset_viz_dir,
        )
        
        # Generate dataset information
        series_length = X_train_orig.shape[1] if X_train_orig.ndim > 1 else len(X_train_orig)
        total_ts = len(X_train_orig)
        unique_classes, class_counts = np.unique(y_train_orig, return_counts=True)
        class_info = ", ".join([f"Class {cls}: {count} series" for cls, count in sorted(zip(unique_classes, class_counts))])
        
        info_items = [
            html.P(f"📊 Total time series: {total_ts}", style={"margin": "4px 0"}),
            html.P(f"📏 Series length: {series_length} timestamps", style={"margin": "4px 0"}),
            html.P(f"🏷️ Classes: {len(unique_classes)} ({class_info})", style={"margin": "4px 0"}),
        ]
        dataset_info_html = html.Div(info_items)
        
        selected_label_txt = "all labels" if label_filter == "__all__" else f"label={label_filter}"
        mode_name = {"all": "sequential", "balanced": "balanced", "random": "random"}.get(selection_mode, "sequential")
        mode_txt = "full individual plots" if full_individual else f"preview ({len(X_selected)} series, {mode_name})"
        actual_classes = len(np.unique(y_selected))
        status = f"Displaying: {mode_txt}, {actual_classes} class(es), {selected_label_txt}"
    except Exception as exc:
        empty = html.Div(f"Error generating dataset plots: {type(exc).__name__}: {exc}", style={"color": "#a33", "fontSize": "12px"})
        return empty, empty, f"TS plots error: {type(exc).__name__}: {exc}", empty

    overlay_block = _build_image_block(
        f"{dataset_name} - Overlay Visualization",
        dataset_viz_dir / f"{dataset_name}_overlay.png",
    )
    grid_block = _build_image_block(
        "Individual time series (visualize_TS)",
        dataset_viz_dir / f"{dataset_name}.png",
    )
    return overlay_block, grid_block, status, dataset_info_html


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
    wilcoxon_csv: Path | None = None,
):
    payload = collect_results(results_csv, data_dir, wilcoxon_csv)
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
        html.Div([html.H1("Time Series Classifier Dashboard")], className="header"),
        dcc.Tabs(id="tabs", value="tab-metrics", children=[
            dcc.Tab(label="Metrics", value="tab-metrics"),
            dcc.Tab(label="Timing", value="tab-timing"),
            dcc.Tab(label="Hyperparameter", value="tab-hyperparam"),
            dcc.Tab(label="Datasets", value="tab-datasets"),
        ], className="tabs-bar"),
        html.Div(
            [
                html.P("Welcome. This dashboard is designed for interactive exploration of time-series classification benchmarks.", style={"margin": "0 0 6px 0", "fontWeight": "600"}),
                html.P("How to explore: start with Metrics (quality), continue with Timing (cost), check Hyperparameter (search behavior), and finish with Datasets (raw signal structure).", style={"margin": "0"}),
            ],
            style={"marginTop": "14px", "padding": "10px 12px", "border": "1px solid #d9d9d9", "borderRadius": "8px", "background": "#fbfbfb", "color": "#333"},
        ),
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
            metric_meta = get_metric_metadata()
            metric_opts = [{"label": meta["label"], "value": key} for key, meta in metric_meta.items()]
            return html.Div([
                html.Div([
                    html.P("Metrics page: compare model quality across datasets and classifiers.", style={"margin": "0 0 4px 0", "fontWeight": "600"}),
                    html.P("Use Metric and Dataset selectors to update all main charts. The line chart summarizes behavior across datasets, the bar chart compares classifiers in one dataset, and the Wilcoxon panel highlights statistical significance.", style={"margin": "0"}),
                ], style={"marginBottom": "10px", "color": "#444"}),
                html.Div([
                    html.Div([
                        html.Label("Metric"),
                        dcc.Dropdown(id="metrics-metric", options=metric_opts, value="accuracy", clearable=False),
                    ], style={"flex": "0 0 22%"}),
                    html.Div([
                        html.Label("Dataset"),
                        dcc.Dropdown(id="metrics-dataset", options=ds_opts, value=ds_opts[0]["value"], clearable=False),
                    ], style={"flex": "0 0 22%"}),
                    html.Div([
                        html.Div(id="metrics-dataset-info", style={"marginTop": "2px", "color": "#444", "fontSize": "13px"}),
                        html.Div(id="metrics-summary", style={"marginTop": "4px"}),
                    ], style={"flex": "1", "paddingLeft": "8px"}),
                ], style={"display": "flex", "gap": "10px", "alignItems": "flex-end", "marginTop": "6px"}),
                html.Div(
                    "Interpretation tip: higher values are better for Metrics. Use the delta chart to see how each baseline differs from TSF in the selected dataset.",
                    style={"marginTop": "8px", "padding": "8px 10px", "border": "1px solid #e0e0e0", "borderRadius": "6px", "background": "#fcfcfc", "color": "#4a4a4a", "fontSize": "12px"},
                ),
                html.Div([
                    html.Div([
                        dcc.Graph(id="metrics-line"),
                    ], style={"flex": "1"}),
                    html.Div([
                        dcc.Graph(id="metrics-bar"),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px", "alignItems": "flex-start", "marginTop": "10px"}),
                html.Div(
                    metrics_conclusion,
                    style={"marginTop": "14px", "padding": "12px", "background": "#f9f9f9", "border": "1px solid #ddd", "borderRadius": "8px", "color": "#333"},
                ),
                html.Div(id="metrics-wilcoxon-panel", style={"marginTop": "14px"}),
                html.Div([
                    html.Div([dcc.Graph(id="metrics-delta-chart")], style={"flex": "1"}),
                    html.Div(id="metrics-insights", style={"flex": "1", "paddingLeft": "12px"}),
                ], style={"display": "flex", "gap": "12px", "marginTop": "14px", "alignItems": "flex-start"}),
                html.Div([
                    html.Div([
                        html.Label("Classifier"),
                        dcc.Dropdown(id="metrics-classifier", clearable=False),
                    ], style={"width": "35%", "display": "inline-block", "verticalAlign": "top"}),
                    html.Div(id="metrics-pred-status", style={"width": "61%", "display": "inline-block", "marginLeft": "4%", "marginTop": "22px", "color": "#444", "fontSize": "12px"}),
                ], style={"marginTop": "12px", "marginBottom": "8px"}),
                html.Div(id="metrics-pred-images"),
            ])
        if tab == "tab-timing":
            ds_opts = [{"label": d["name"], "value": d["name"]} for d in payload["datasets"]]
            metric_opts = [{"label": "Train", "value": "train"}, {"label": "Predict", "value": "predict"}, {"label": "Total", "value": "total"}]
            return html.Div([
                html.Div([
                    html.P("Timing page: analyze computational cost.", style={"margin": "0 0 4px 0", "fontWeight": "600"}),
                    html.P("Select a dataset and timing metric (Train, Predict, Total). Bubble size encodes runtime, horizontal bars compare techniques, and profile bars show per-dataset timing details.", style={"margin": "0"}),
                ], style={"marginBottom": "10px", "color": "#444"}),
                html.Div([
                    html.Div([html.Label("Dataset"), dcc.Dropdown(id="timing-dataset", options=ds_opts, value=ds_opts[0]["value"], clearable=False)], style={"width": "45%", "display": "inline-block"}),
                    html.Div([html.Label("Metric"), dcc.Dropdown(id="timing-metric", options=metric_opts, value="total", clearable=False)], style={"width": "45%", "display": "inline-block", "marginLeft": "20px"}),
                ]),
                html.Div([html.P("Bubble chart: each point represents a dataset/technique pair. X=dataset size, Y=series length, size=time depending on selected metric.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"})]),
                dcc.Graph(id="timing-bubble"),
                html.Div([
                    html.Div([html.P("Stacked horizontal bar chart: technique timings according to selected metric.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"}), dcc.Graph(id="timing-bar", style={"height": "480px"})], style={"flex": "1", "paddingRight": "10px"}),
                    html.Div([html.P("Profile chart: per-dataset total or individual timing bars by technique.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"}), dcc.Graph(id="timing-profile", style={"height": "480px"})], style={"flex": "1", "paddingLeft": "10px"}),
                ], id="timing-extra-graphs", style={"display":"flex", "flexDirection":"row", "alignItems": "flex-start", "marginTop": "14px"}),
                html.Div(
                    timing_conclusion,
                    style={"marginTop": "14px", "padding": "12px", "background": "#f9f9f9", "border": "1px solid #ddd", "borderRadius": "8px", "color": "#333"},
                ),
            ])
        ds_opts = [{"label": d["name"], "value": d["name"]} for d in payload["datasets"]]
        if tab == "tab-hyperparam":
            return html.Div([
                html.Div([
                    html.P("Hyperparameter page: compare optimization strategies.", style={"margin": "0 0 4px 0", "fontWeight": "600"}),
                    html.P("For the selected dataset, the table reports each method's best score, runtime, and best parameter configuration. The scatter chart helps you inspect score-vs-time behavior during search.", style={"margin": "0"}),
                ], style={"marginBottom": "10px", "color": "#444"}),
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
            html.Div([
                html.P("Datasets page: inspect raw time-series patterns.", style={"margin": "0 0 4px 0", "fontWeight": "600"}),
                html.P("Choose dataset, sample size (N series), and sampling mode. Use label filtering to focus on a specific class. Overlay plots summarize global shape and variability; this helps contextualize model behavior seen in other tabs.", style={"margin": "0"}),
            ], style={"marginBottom": "10px", "color": "#444"}),
            html.Div([
                html.Div([html.Label("Dataset"), dcc.Dropdown(id="datasets-dataset", options=ds_opts, value=default_viz_dataset, clearable=False)], style={"width": "30%", "display": "inline-block"}),
                html.Div([html.Label("N series"), dcc.Dropdown(id="datasets-n-series", options=[{"label": "8 (small preview)", "value": 8}, {"label": "16 (preview)", "value": 16}, {"label": "32 (medium)", "value": 32}, {"label": "64 (large)", "value": 64}, {"label": "All", "value": -1}], value=16, clearable=False)], style={"width": "20%", "display": "inline-block", "marginLeft": "15px"}),
            ], style={"marginBottom": "12px"}),
            html.Div([
                html.Label("Selection mode"),
                dcc.RadioItems(
                    id="datasets-selection-mode",
                    options=[
                        {"label": "Sequential", "value": "all"},
                        {"label": "Balanced per class", "value": "balanced"},
                        {"label": "Random", "value": "random"},
                    ],
                    value="all",
                    inline=True,
                    inputStyle={"transform": "scale(1.45)", "marginRight": "8px"},
                    labelStyle={
                        "display": "inline-flex",
                        "alignItems": "center",
                        "marginRight": "20px",
                        "fontSize": "14px",
                    },
                ),
                html.Div([
                    html.Label("Random seed:", style={"marginLeft": "40px", "marginRight": "8px", "fontSize": "13px"}),
                    dcc.Input(id="datasets-random-seed", type="number", placeholder="42", value=42, style={"width": "60px", "padding": "4px", "fontSize": "12px", "borderRadius": "4px", "border": "1px solid #ccc"}),
                ], style={"display": "inline-flex", "alignItems": "center", "marginLeft": "10px"}),
            ], style={"marginBottom": "10px", "color": "#444", "fontSize": "12px"}),
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
            html.Div("Plots are displayed below this control panel. Note: selecting large N or All may take longer to render; the loading spinner indicates active processing.", style={"marginBottom": "8px", "color": "#555", "fontSize": "12px"}),
            html.Div(id="datasets-info", style={"marginBottom": "10px", "padding": "10px", "background": "#f9f9f9", "borderRadius": "6px", "border": "1px solid #ddd"}),
            html.Div(id="datasets-status", style={"marginBottom": "10px", "color": "#444", "fontSize": "12px"}),
            dcc.Loading(
                id="datasets-loading",
                type="circle",
                children=html.Div(id="datasets-overlay-image", style={"marginTop": "12px"}),
            ),
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
        [Input("metrics-metric", "value"), Input("metrics-dataset", "value"), Input("metrics-classifier", "value")],
    )
    def update_metrics(metric_value, dataset_value, classifier_value):
        fig_bar, fig_line, summary = build_metrics_figures(payload, dataset_value, metric_value)
        metric_meta = get_metric_metadata()
        meta = metric_meta.get(metric_value, metric_meta["accuracy"])
        summary_el = html.Div([html.P(f"Best classifier: {summary['best']}"), html.P(f"Worst classifier: {summary['worst']}"), html.P(f"Mean {meta['label']}: {summary['mean']}")])
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
        Output("metrics-wilcoxon-panel", "children"),
        [Input("metrics-metric", "value")],
    )
    def update_wilcoxon_panel(metric_value):
        wilcoxon_df = payload.get("wilcoxon_results")
        return build_wilcoxon_panel(wilcoxon_df, metric_value)

    @app.callback(
        Output("metrics-delta-chart", "figure"),
        [Input("metrics-metric", "value"), Input("metrics-dataset", "value")],
    )
    def update_delta_chart(metric_value, dataset_value):
        return build_metrics_delta_chart(payload, dataset_value, metric_value)

    @app.callback(
        Output("metrics-insights", "children"),
        [Input("metrics-metric", "value"), Input("metrics-dataset", "value")],
    )
    def update_insights(metric_value, dataset_value):
        return build_insights_summary(payload, dataset_value, metric_value)

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
        [Output("datasets-overlay-image", "children"), Output("datasets-status", "children"), Output("datasets-info", "children")],
        [Input("datasets-dataset", "value"), Input("datasets-label-filter", "value"), Input("datasets-n-series", "value"), Input("datasets-selection-mode", "value"), Input("datasets-random-seed", "value")],
    )
    def update_dataset_overlay(dataset_value, label_filter, n_series, selection_mode, random_seed):
        overlay_block, _grid_block, status, dataset_info_html = generate_dataset_visual_assets(
            dataset_name=dataset_value,
            data_dir=data_dir,
            viz_dir=viz_dir,
            label_filter=label_filter,
            full_individual=False,
            n_series=n_series,
            selection_mode=selection_mode,
            random_state=int(random_seed) if random_seed else 42,
        )
        return [overlay_block], status, dataset_info_html

    return app


def main():
    parser = argparse.ArgumentParser(description="Run Dash dashboard server")
    parser.add_argument("--results", default="results/benchmark_comparison.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--hp-dir", default="results")
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--viz-dir", default="visualization")
    parser.add_argument("--wilcoxon", default="results/benchmark_comparison_wilcoxon.csv", help="Path to Wilcoxon test results CSV")
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
        Path(args.wilcoxon) if args.wilcoxon else None,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
