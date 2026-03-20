"""Dash app for Time Series classification metrics (Dash + Plotly).

Full dynamic migration, no static HTML pages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go


def _load_ucr_stats(data_dir: Path, dataset_name: str) -> dict[str, int]:
    train_path = data_dir / dataset_name / f"{dataset_name}_TRAIN.txt"
    if not train_path.exists():
        return {"nSeries": None, "length": None}

    arr = pd.read_csv(train_path, sep=r"\s+", header=None).values
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {"nSeries": int(arr.shape[0]), "length": int(arr.shape[1] - 1)}


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

    bar_fig = go.Figure(go.Bar(x=list(accuracies.keys()), y=list(accuracies.values()), marker_color="royalblue"))
    bar_fig.update_layout(title=f"Accuracy for {dataset}", xaxis_title="Classifier", yaxis_title="Accuracy (%)", yaxis=dict(range=[0, 100]))

    datasets_list = [d["name"] for d in payload["datasets"]]
    line_series = []
    for cls in payload["classifiers"]:
        y = [payload["metrics_by_dataset"].get(ds, {}).get(cls, {}).get("accuracy", 0.0) * 100 for ds in datasets_list]
        line_series.append(go.Scatter(x=datasets_list, y=y, mode="lines+markers", name=cls))

    line_fig = go.Figure(line_series)
    line_fig.update_layout(title="Accuracy across datasets", xaxis_title="Dataset", yaxis_title="Accuracy (%)", yaxis=dict(range=[0, 100]))

    best_cls = max(accuracies, key=lambda x: accuracies[x])
    worst_cls = min(accuracies, key=lambda x: accuracies[x])
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


def create_dash_app(results_csv: Path, data_dir: Path, hp_dir: Path):
    payload = collect_results(results_csv, data_dir)
    hp_results = collect_hyperparameter_results(hp_dir)

    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div([
        html.Div([html.H1("Time Series Classifier Dashboard (3 pages)")], className="header"),
        dcc.Tabs(id="tabs", value="tab-metrics", children=[
            dcc.Tab(label="Metrics", value="tab-metrics"),
            dcc.Tab(label="Timing", value="tab-timing"),
            dcc.Tab(label="Hyperparameter", value="tab-hyperparam"),
        ], className="tabs-bar"),
        html.Div(id="tab-content", style={"marginTop": "20px", "padding": "12px"}),
        html.Div([
            html.Hr(),
            html.P("This dashboard provides interactive analysis of classifier performance, timing, and hyperparameter search results for time series datasets."),
            html.P("Created by Ane Miren Arregi, Iker Bereziartua and Eneko Zabaleta as part of ASTD Project 3."),
        ], style={"marginTop": "40px", "fontSize": "12px", "color": "#666", "textAlign": "center"}),
    ], style={"background": "#f4f4f3"})

    @app.callback(Output("tab-content", "children"), [Input("tabs", "value")])
    def render_tab(tab):
        if tab == "tab-metrics":
            ds_opts = [{"label": d["name"], "value": d["name"]} for d in payload["datasets"]]
            cls_opts = [{"label": c, "value": c} for c in payload["classifiers"]]
            return html.Div([
                html.Div([html.P("Metrics tab provides accuracy comparison of classifiers for selected dataset. Choose dataset and classifier to inspect performance metrics and trends." )], style={"marginBottom": "10px", "color": "#444"}),
                html.Div([
                    html.Div([html.Label("Dataset"), dcc.Dropdown(id="metrics-dataset", options=ds_opts, value=ds_opts[0]["value"], clearable=False)], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"}),
                    html.Div([html.Label("Classifier"), dcc.Dropdown(id="metrics-classifier", options=cls_opts, value=cls_opts[0]["value"], clearable=False)], style={"width": "45%", "display": "inline-block", "marginLeft": "20px", "verticalAlign": "top"}),
                ]),
                html.Div(id="metrics-summary", style={"marginTop": "12px"}),
                dcc.Graph(id="metrics-bar"),
                dcc.Graph(id="metrics-line"),
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
                html.Div([html.Button("Mostrar/bloquear otros gráficos", id="toggle-graphs", n_clicks=0)], style={"margin":"10px 0"}),
                html.Div([html.P("Bubble chart: each point represents a dataset/technique pair. X=dataset size, Y=series length, size=time depending on selected metric.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"})]),
                dcc.Graph(id="timing-bubble"),
                html.Div([
                    html.Div([html.P("Stacked horizontal bar chart: technique timings according to selected metric.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"}), dcc.Graph(id="timing-bar", style={"height": "480px"})], style={"flex": "1", "paddingRight": "10px"}),
                    html.Div([html.P("Profile chart: per-dataset total or individual timing bars by technique.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"}), dcc.Graph(id="timing-profile", style={"height": "480px"})], style={"flex": "1", "paddingLeft": "10px"}),
                ], id="timing-extra-graphs", style={"display":"flex", "flexDirection":"row", "alignItems": "flex-start"}),
            ])
        ds_opts = [{"label": d["name"], "value": d["name"]} for d in payload["datasets"]]
        return html.Div([
            html.Div([html.P("Hyperparameter tab visualizes best obtained score and elapsed time for optimization methods. Use this view to compare search quality and training speed.")], style={"marginBottom": "10px", "color": "#444"}),
            html.Div([html.Label("Dataset"), dcc.Dropdown(id="hp-dataset", options=ds_opts, value=ds_opts[0]["value"], clearable=False)], style={"width": "30%", "marginBottom": "10px"}),
            dash_table.DataTable(id="hp-table", columns=[{"name": "Method", "id": "method"}, {"name": "Best Score", "id": "best_score"}, {"name": "Elapsed s", "id": "elapsed_seconds"}, {"name": "Best params", "id": "best_params"}], style_table={"overflowX": "auto"}, style_cell={"textAlign": "left", "fontSize": "13px"}),
            html.Div([html.P("Scatter chart: each point is an hyperparameter trial; x=elapsed time, y=accuracy; bigger points are best accuracy for method.", style={"fontStyle": "italic", "marginBottom": "8px", "color": "#333"})]),
            dcc.Graph(id="hp-scatter"),
            html.Pre(id="hp-summary", style={"whiteSpace": "pre-wrap", "marginTop": "12px"}),
        ])

    @app.callback([Output("metrics-bar", "figure"), Output("metrics-line", "figure"), Output("metrics-summary", "children")], [Input("metrics-dataset", "value")])
    def update_metrics(dataset_value):
        fig_bar, fig_line, summary = build_metrics_figures(payload, dataset_value)
        summary_el = html.Div([html.P(f"Best classifier: {summary['best']}"), html.P(f"Worst classifier: {summary['worst']}"), html.P(f"Mean accuracy: {summary['mean']}")])
        return fig_bar, fig_line, summary_el

    @app.callback(
        [Output("timing-bar", "figure"), Output("timing-bubble", "figure"), Output("timing-profile", "figure")],
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

    @app.callback([Output("hp-table", "data"), Output("hp-scatter", "figure"), Output("hp-summary", "children")], [Input("hp-dataset", "value")])
    def update_hp(dataset_value):
        table_data, scatter_fig, summary_text = build_hyperparam_figures(hp_results, dataset_value)
        return table_data, scatter_fig, summary_text

    return app


def main():
    parser = argparse.ArgumentParser(description="Run Dash dashboard server")
    parser.add_argument("--results", default="results/benchmark_comparison.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--hp-dir", default="results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_dash_app(Path(args.results), Path(args.data_dir), Path(args.hp_dir))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
