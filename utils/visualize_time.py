"""Generate an interactive HTML dashboard for runtime analysis.

This script produces a single self-contained HTML file (Chart.js) that lets you
explore runtime behavior across datasets and classifiers.

Usage example:
  python utils/visualize_time.py --results results/benchmark_comparison.csv \
      --out visualization/time_vs_size.html

The generated HTML embeds the data so it works when opened directly in a browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TS Classifier Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: system-ui, sans-serif;
    font-size: 13px;
    background: #f5f5f4;
    color: #2c2c2a;
    padding: 24px;
  }

  h1 { font-size: 16px; font-weight: 500; margin-bottom: 4px; }
  .subtitle { font-size: 12px; color: #73726c; margin-bottom: 16px; }

  /* ── controls ── */
  .controls {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }
  .controls label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #5f5e5a;
  }
  select {
    font-size: 12px;
    padding: 4px 8px;
    border: 0.5px solid #b4b2a9;
    border-radius: 6px;
    background: #fff;
    color: #2c2c2a;
    cursor: pointer;
  }

  /* ── panel grid ── */
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    justify-items: center;
  }

  .panel {
    background: #fff;
    border: 0.5px solid #d3d1c7;
    border-radius: 10px;
    padding: 16px;
    width: 100%;
    max-width: 760px;
  }
  .panel--span {
    grid-column: span 2;
    justify-self: stretch;
    max-width: none;
    width: 100%;
  }

  /* panel title + description */
  .panel-header { margin-bottom: 10px; }
  .panel-title  { font-size: 12px; font-weight: 500; color: #2c2c2a; }
  .panel-desc   { font-size: 11px; color: #888780; margin-top: 2px; line-height: 1.4; }

  /* canvas wrapper with fixed height */
  .chart-wrap   { position: relative; height: 260px; }

  /* bubble legend (top chart) */
  .bubble-legend {
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 11px;
    color: #555;
    margin-top: 10px;
  }
  .bubble-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .bubble-legend-swatch {
    border-radius: 50%;
    background: rgba(60, 120, 220, 0.35);
    border: 1px solid rgba(60, 120, 220, 0.8);
  }

  /* ── manual legend ── */
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: #5f5e5a;
  }
  .legend-swatch {
    width: 10px; height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }
</style>
</head>
<body>

<h1>Time Series Classifier — timing analysis</h1>
<p class="subtitle">Data loaded from your results CSV. Use the dataset selector to switch.</p>

<div class="controls">
  <label>
    Dataset
    <select id="datasetSelect"></select>
  </label>
  <label>
    Summary metric
    <select id="metricBy">
      <option value="train">Train</option>
      <option value="test">Test</option>
      <option value="total">Train + Test</option>
    </select>
  </label>
</div>

<div class="grid">

  <!-- ── PANEL 1: barplot (mean) ── -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title" id="barsTitle">Mean time per technique</div>
      <div class="panel-desc">
        Each bar is the mean across all datasets.
      </div>
    </div>
    <div class="chart-wrap"><canvas id="cBars"></canvas></div>
  </div>

  <!-- ── PANEL 2: dataset size vs length ── -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">Dataset size vs series length (bubble = time)</div>
      <div class="panel-desc">
        X axis = number of time series in the dataset.
        Y axis = length of each time series.
        Bubble size reflects the selected time metric (train/test/total) for each technique.
      </div>
    </div>
    <div class="chart-wrap"><canvas id="cLogLog"></canvas></div>
    <div id="bubbleLegend" class="bubble-legend"></div>
  </div>

  <!-- ── PANEL 3: dataset profile (stacked) ── -->
  <div class="panel panel--span">
    <div class="panel-header">
      <div class="panel-title" id="profileTitle">Dataset profile (stacked)</div>
      <div class="panel-desc">
        X = dataset (sorted by number of series). Each stacked segment is a technique.
        In Train+Test mode, each technique is split into Train and Test.
      </div>
    </div>
    <div class="chart-wrap"><canvas id="cProfile"></canvas></div>
  </div>

</div>

<!-- ── leyenda global ── -->
<div class="legend" id="legend"></div>

<script>
// Dataset + results cargados desde Python
const DATASETS   = __DATASETS__;
const TECHNIQUES = __TECHNIQUES__;
const TECH_COLORS = __TECH_COLORS__;
const TECH_FILL   = __TECH_FILL__;
const DATA       = __DATA__;

// Generador determinista simple
const mkRng = seed => {
  let s = seed;
  return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
};
const r = mkRng(42);

/* ═══════════════════════════════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════════════════════════════ */

const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

const toRgba = (hex, alpha) => {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const ALPHA_TRAIN = 0.65;
const ALPHA_TEST  = 0.35;
const MIN_BUBBLE_RATIO = 0.35;


let charts = {};

function buildLogLog(datasetName, metricMode) {
  charts.loglog?.destroy();

  // Position bubbles by dataset size/length, and size them by time.
  const datasets = [];
  const datasetMap = new Map();
  DATA.forEach(d => {
    const key = `${d.ds}||${d.technique}`;
    if (!datasetMap.has(key)) datasetMap.set(key, []);
    datasetMap.get(key).push(d);
  });

  const maxTime = Math.max(...DATA.map(d => Math.max(d.trainT, d.testT)));
  const scaleSize = v => 4 + 18 * Math.sqrt((v || 0) / (maxTime || 1));

  const buildBubbleLegend = () => {
    const legend = document.getElementById('bubbleLegend');
    const sample = [0.25, 0.5, 1].map(f => ({
      label: `${Math.round(maxTime * f)}s`,
      r: scaleSize(maxTime * f),
    }));
    legend.innerHTML = sample.map(s =>
      `<span class="bubble-legend-item">
         <span class="bubble-legend-swatch" style="width:${s.r * 2}px; height:${s.r * 2}px"></span>
         ${s.label}
       </span>`
    ).join('');
  };

  const makePoints = (technique, timeFn) => {
    return DATASETS.map(ds => {
      const key = `${ds.name}||${technique}`;
      const recs = datasetMap.get(key) ?? [];
      const time = recs.length ? mean(recs.map(r => timeFn(r))) : null;
      const r = scaleSize(time);
      return {
        x: ds.nSeries,
        y: ds.length,
        r,
        _d: { ds: ds.name, technique, time, train: recs.length ? mean(recs.map(r => r.trainT)) : null, test: recs.length ? mean(recs.map(r => r.testT)) : null },
      };
    });
  };

  if (metricMode === 'total') {
    const testDatasets = TECHNIQUES.map((t, ti) => ({
      label: `${t} (test)`,
      data: makePoints(t, r => r.testT),
      backgroundColor: toRgba(TECH_COLORS[ti], ALPHA_TEST),
      borderColor: toRgba(TECH_COLORS[ti], Math.min(ALPHA_TEST + 0.4, 1)),
      borderWidth: 1,
    }));

    const trainDatasets = TECHNIQUES.map((t, ti) => ({
      label: `${t} (train)`,
      data: makePoints(t, r => r.trainT),
      backgroundColor: toRgba(TECH_COLORS[ti], ALPHA_TRAIN),
      borderColor: toRgba(TECH_COLORS[ti], Math.min(ALPHA_TRAIN + 0.4, 1)),
      borderWidth: 1,
    }));

    // Ensure the smaller bubble remains visible even if the other is much larger.
    for (let ti = 0; ti < testDatasets.length; ti += 1) {
      const testPoints = testDatasets[ti].data;
      const trainPoints = trainDatasets[ti].data;
      for (let i = 0; i < testPoints.length; i += 1) {
        const rt = trainPoints[i].r;
        const rs = testPoints[i].r;
        const mx = Math.max(rt, rs);
        const min = Math.max(4, mx * MIN_BUBBLE_RATIO);
        if (rt < min) trainPoints[i].r = min;
        if (rs < min) testPoints[i].r = min;
      }
    }

    datasets.push(...testDatasets);
    datasets.push(...trainDatasets);
    buildBubbleLegend();
  } else {
    const alpha = metricMode === 'test' ? ALPHA_TEST : ALPHA_TRAIN;
    const suffix = metricMode === 'test' ? '(test)' : '(train)';

    datasets.push(...TECHNIQUES.map((t, ti) => ({
      label: `${t} ${suffix}`,
      data: makePoints(t, r => (metricMode === 'test' ? r.testT : r.trainT)),
      backgroundColor: toRgba(TECH_COLORS[ti], alpha),
      borderColor: toRgba(TECH_COLORS[ti], Math.min(alpha + 0.4, 1)),
      borderWidth: 1,
    })));
  }

  charts.loglog = new Chart(document.getElementById('cLogLog'), {
    type: 'bubble',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw._d;
              return [
                `${d.technique} — ${d.ds}`,
                `train=${d.trainT.toFixed(2)}s  test=${d.testT.toFixed(3)}s`,
              ];
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'number of time series', font: { size: 11 } },
          ticks: { font: { size: 10 } },
        },
        y: {
          type: 'linear',
          title: { display: true, text: 'time series length', font: { size: 11 } },
          ticks: { font: { size: 10 } },
        },
      },
      layout: { padding: 10 },
    }
  });
}

function buildBars(datasetName, metric) {
  charts.bars?.destroy();

  const rows = DATA.filter(d => d.ds === datasetName);
  const metricLabel = metric === 'test' ? 'test time' : metric === 'total' ? 'train+test time' : 'train time';

  let datasets;
  if (metric === 'total') {
    const trainMeans = TECHNIQUES.map((_, ti) => {
      const vals = rows.filter(d => d.techIdx === ti).map(d => d.trainT);
      return mean(vals);
    });
    // no error bars, std not shown
    const trainStds = TECHNIQUES.map(() => 0);

    const testMeans = TECHNIQUES.map((_, ti) => {
      const vals = rows.filter(d => d.techIdx === ti).map(d => d.testT);
      return mean(vals);
    });
    const testStds = TECHNIQUES.map(() => 0);

    datasets = [
      {
        label: 'train',
        data: trainMeans,
        backgroundColor: TECH_COLORS.map(c => toRgba(c, ALPHA_TRAIN)),
        borderColor: TECH_COLORS.map(c => toRgba(c, Math.min(ALPHA_TRAIN + 0.4, 1))),
        borderWidth: 1,
        borderRadius: 3,
        stack: 'a',
      },
      {
        label: 'test',
        data: testMeans,
        backgroundColor: TECH_COLORS.map(c => toRgba(c, ALPHA_TEST)),
        borderColor: TECH_COLORS.map(c => toRgba(c, Math.min(ALPHA_TEST + 0.4, 1))),
        borderWidth: 1,
        borderRadius: 3,
        stack: 'a',
      },
    ];
  } else {
    const field = metric === 'test' ? 'testT' : 'trainT';

    const means = TECHNIQUES.map((_, ti) => {
      const vals = rows.filter(d => d.techIdx === ti).map(d => d[field]);
      return mean(vals);
    });
    datasets = [
      {
        label: metric,
        data: means,
        backgroundColor: TECH_COLORS.map(c => toRgba(c, metric === 'test' ? ALPHA_TEST : ALPHA_TRAIN)),
        borderColor: TECH_COLORS.map(c => toRgba(c, Math.min((metric === 'test' ? ALPHA_TEST : ALPHA_TRAIN) + 0.4, 1))),
        borderWidth: 1,
        borderRadius: 3,
      },
    ];
  }

  charts.bars = new Chart(document.getElementById('cBars'), {
    type: 'bar',
    data: { labels: TECHNIQUES, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      animation: { duration: 300 },
      datasets: {
        barThickness: 18,
        maxBarThickness: 24,
        barPercentage: 0.85,
        categoryPercentage: 0.9,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const i = ctx.dataIndex;
              const ds = ctx.dataset;
              const mean = ds.data[i];
              return `${ds.label}: mean=${mean.toFixed(1)}s`;
            }
          }
        }
      },
      scales: {
        x: {
          stacked: metric === 'total',
          title: { display: true, text: `${metricLabel} (s)`, font: { size: 11 } },
          ticks: { font: { size: 10 } },
          grace: '10%',
        },
        y: { stacked: metric === 'total', ticks: { font: { size: 10 } } },
      },
      layout: { padding: { right: 16 } },
    }
  });
}

function buildProfile(datasetName, metric) {
  charts.profile?.destroy();

  const sorted = [...DATASETS].sort((a, b) => a.nSeries - b.nSeries);
  const labels  = sorted.map(ds => `${ds.name}\n(N=${ds.nSeries})`);

  // Ensure the profile chart has enough vertical space as datasets grow.
  // For just a few datasets, keep the chart compact.
  const profileWrap = document.getElementById('cProfile').parentElement;
  const baseHeight = labels.length <= 2 ? 180 : 260;
  const targetHeight = Math.max(baseHeight, labels.length * 40 + 60);
  profileWrap.style.height = `${targetHeight}px`;

  const metricLabel = metric === 'test' ? 'test time' : metric === 'total' ? 'train+test time' : 'train time';

  const datasets = [];

  // For stacked bars we want a per-technique stack with train/test split.
  // In non-total mode we show just train or just test per technique.
  if (metric === 'total') {
    TECHNIQUES.forEach((t, ti) => {
      const trainValues = sorted.map(ds => {
        const row = DATA.find(d => d.ds === ds.name && d.techIdx === ti);
        return row ? row.trainT : null;
      });
      const testValues = sorted.map(ds => {
        const row = DATA.find(d => d.ds === ds.name && d.techIdx === ti);
        return row ? row.testT : null;
      });

      datasets.push({
        label: `train (${t})`,
        data: trainValues,
        backgroundColor: toRgba(TECH_COLORS[ti], ALPHA_TRAIN),
        borderColor: toRgba(TECH_COLORS[ti], Math.min(ALPHA_TRAIN + 0.4, 1)),
        borderWidth: 1,
        stack: 'a',
      });

      datasets.push({
        label: `test (${t})`,
        data: testValues,
        backgroundColor: toRgba(TECH_COLORS[ti], ALPHA_TEST),
        borderColor: toRgba(TECH_COLORS[ti], Math.min(ALPHA_TEST + 0.4, 1)),
        borderWidth: 1,
        stack: 'a',
      });
    });
  } else {
    const field = metric === 'test' ? 'testT' : 'trainT';

    datasets.push(...TECHNIQUES.map((t, ti) => ({
      label: t,
      data: sorted.map(ds => {
        const row = DATA.find(d => d.ds === ds.name && d.techIdx === ti);
        return row ? row[field] : null;
      }),
      backgroundColor: metric === 'test' ? toRgba(TECH_COLORS[ti], 0.3) : toRgba(TECH_COLORS[ti], 0.5),
      borderColor: toRgba(TECH_COLORS[ti], 0.7),
      borderWidth: 1,
      stack: 'a',
    })));
  }

  charts.profile = new Chart(document.getElementById('cProfile'), {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          stacked: true,
          title: { display: true, text: `${metricLabel} (s)`, font: { size: 11 } },
          ticks: { font: { size: 10 } },
        },
        y: {
          stacked: false,
          ticks: { font: { size: 9 }, maxRotation: 0, autoSkip: false },
        },
      },
      datasets: { barThickness: 32, maxBarThickness: 48, barPercentage: 0.92, categoryPercentage: 0.95 },
      layout: { padding: 8 },
    }
  });
}

function buildLegend() {
  document.getElementById('legend').innerHTML = TECHNIQUES.map((t, i) =>
    `<span class="legend-item">
       <span class="legend-swatch" style="background:${TECH_COLORS[i]}"></span>${t}
     </span>`
  ).join('');
}

function rebuild() {
  const datasetName = document.getElementById('datasetSelect').value;
  const metricMode  = document.getElementById('metricBy').value;
  const metricName  = metricMode === 'test' ? 'test' : metricMode === 'total' ? 'train+test' : 'train';

  document.getElementById('barsTitle').textContent = `Mean ${metricName} time by technique`;
  document.getElementById('profileTitle').textContent = `Dataset profile (stacked) — ${metricName}`;

  buildLogLog(datasetName, metricMode);
  buildBars(datasetName, metricMode);
  buildProfile(datasetName, metricMode);
  buildLegend();
}

function init() {
  const select = document.getElementById('datasetSelect');
  const metricLabel = document.getElementById('metricBy').parentElement;

  DATASETS.forEach(ds => {
    const o = document.createElement('option');
    o.value = ds.name;
    o.textContent = ds.name;
    select.appendChild(o);
  });

  select.addEventListener('change', rebuild);
  document.getElementById('metricBy').addEventListener('change', rebuild);

  // If only one dataset is present, the metric selector is less useful.
  if (DATASETS.length <= 1) {
    metricLabel.style.display = 'none';
    document.getElementById('metricBy').value = 'train';
  }

  select.value = DATASETS[0]?.name ?? '';
  rebuild();
}

init();
</script>
</body>
</html>
"""


def _load_ucr_stats(data_dir: Path, dataset_name: str) -> tuple[int, int]:
    train_path = data_dir / dataset_name / f"{dataset_name}_TRAIN.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found: {train_path}")

    # UCR .txt files are whitespace-delimited and may include variable spacing.
    arr = pd.read_csv(train_path, sep=r"\s+", header=None).values
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n_series = arr.shape[0]
    series_length = arr.shape[1] - 1
    return n_series, series_length


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #rrggbb to rgba(r,g,b,a)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_palette(n: int) -> list[str]:
    # A default palette with good contrast for ~20 categories
    base = [
        "#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#BA7517",
        "#5B3D99", "#4A7D39", "#C32E4B", "#2A6D99", "#9E6712",
        "#B02E77", "#2F5F3C", "#6B4A99", "#1D7199", "#9A2F25",
        "#707070", "#2C7B32", "#D0693C", "#7A2E99", "#277F5B",
    ]
    if n <= len(base):
        return base[:n]
    # cycle if more needed
    return [base[i % len(base)] for i in range(n)]


def generate_html(
    results_csv: Path,
    data_dir: Path,
    out_path: Path,
    dataset_filter: str | None = None,
) -> Path:
    df = pd.read_csv(results_csv)

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]

    if df.empty:
        raise ValueError(f"No results found for dataset filter: {dataset_filter}")

    datasets = []
    dataset_names = sorted(df["dataset"].unique())
    for ds in dataset_names:
        n_series, length = _load_ucr_stats(data_dir, ds)
        datasets.append({"name": ds, "nSeries": int(n_series), "length": int(length)})

    classifiers = sorted(df["classifier"].unique())

    classifier_to_idx = {c: i for i, c in enumerate(classifiers)}

    records: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        cls = r["classifier"]
        if cls not in classifier_to_idx:
            continue
        rec = {
            "ds": r["dataset"],
            "nSeries": int(next(x["nSeries"] for x in datasets if x["name"] == r["dataset"])),
            "length": int(next(x["length"] for x in datasets if x["name"] == r["dataset"])),
            "technique": cls,
            "techIdx": classifier_to_idx[cls],
            "trainT": float(r.get("fit_time_s") or 0.0),
            "testT": float(r.get("predict_time_s") or 0.0),
        }
        records.append(rec)

    palette = _build_palette(len(classifiers))
    fills = [_hex_to_rgba(c, 0.55) for c in palette]

    html = HTML_TEMPLATE
    html = html.replace("__DATASETS__", json.dumps(datasets, indent=2, ensure_ascii=False))
    html = html.replace("__TECHNIQUES__", json.dumps(classifiers, indent=2, ensure_ascii=False))
    html = html.replace("__TECH_COLORS__", json.dumps(palette, indent=2, ensure_ascii=False))
    html = html.replace("__TECH_FILL__", json.dumps(fills, indent=2, ensure_ascii=False))
    html = html.replace("__DATA__", json.dumps(records, indent=2, ensure_ascii=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate an interactive HTML dashboard showing runtime vs dataset size.")
    parser.add_argument("--results", default="results/benchmark_comparison.csv", help="CSV file with timings (output of main_run.py)")
    parser.add_argument("--data-dir", default="data", help="Directory containing UCR datasets")
    parser.add_argument("--out", default="visualization/time_vs_size.html", help="Output HTML file path")
    parser.add_argument("--dataset", help="Optionally restrict to one dataset")
    args = parser.parse_args()

    out = generate_html(
        results_csv=Path(args.results),
        data_dir=Path(args.data_dir),
        out_path=Path(args.out),
        dataset_filter=args.dataset,
    )
    print(f"Wrote dashboard to: {out}")


if __name__ == "__main__":
    main()
