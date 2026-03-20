from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>Hyperparameter Search Dashboard</title>
<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js'></script>
<style>
  body { margin: 0; font-family: system-ui, sans-serif; background: #f4f4f4; color: #222; }
  .container { max-width: 1200px; margin: 0 auto; padding: 16px; }
  h1 { margin-bottom: 12px; font-size: 22px; }
  .controls { margin-bottom: 14px; display: flex; gap: 10px; align-items: center; }
  select { font-size: 13px; padding: 6px 8px; }
  .layout { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .panel { background: white; border: 1px solid #d1d1d0; border-radius: 10px; padding: 12px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 8px; }
  th, td { text-align: left; border: 1px solid #ddd; padding: 6px; font-size: 13px; }
  th { background: #f9f9f9; }
  .subtitle { font-size: 12px; color: #555; margin-bottom: 8px; }
</style>
</head>
<body>
<div class='container'>
  <h1>Hyperparameter Search Dashboard</h1>
  <div class='controls'>
    <label>Dataset:<select id='datasetSelect'></select></label>
  </div>
  <div class='layout'>
    <div class='panel'>
      <h2>Result table</h2>
      <p class='subtitle'>Método - best accuracy - best params</p>
      <div id='tableHolder'></div>
    </div>
    <div class='panel'>
      <h2>Elapsed vs Accuracy</h2>
      <div style='position: relative; width: 100%; height: 260px;'>
        <canvas id='scatterChart'></canvas>
      </div>
    </div>
  </div>
  <div class='panel' style='margin-top:12px;'>
    <h2>Legend and best results</h2>
    <div id='bestInfo'></div>
  </div>
</div>
<script>
const RAW = __RAW_RESULTS__;

function uniqueValues(arr){return [...new Set(arr)];}

function buildDatasetSelector(){
  const select = document.getElementById('datasetSelect');
  const datasets = uniqueValues(RAW.map(x=>x.dataset));
  select.innerHTML = datasets.map(d=>`<option value="${d}">${d}</option>`).join('');
  select.addEventListener('change', renderForDataset);
  if(datasets.length>0) renderForDataset();
}

function renderForDataset(){
  const dataset = document.getElementById('datasetSelect').value;
  const rows = RAW.filter(r=>r.dataset===dataset);
  if(!rows.length){document.getElementById('tableHolder').innerHTML='<p>No hay datos</p>'; return;}

  const methods = uniqueValues(rows.map(r=>r.method));
  const tableRows = methods.map(m=>{
    const subset = rows.filter(r=>r.method===m);
    const best = subset.reduce((a,b)=>b.best_score>a.best_score ? b : a);
    return `<tr><td>${m}</td><td>${(best.best_score*100).toFixed(3)}%</td><td><pre style='margin:0;font-size:11px;'>${JSON.stringify(best.best_params,null,2)}</pre></td></tr>`;
  }).join('');

  document.getElementById('tableHolder').innerHTML = `<table><thead><tr><th>Método</th><th>Best accuracy</th><th>Best params</th></tr></thead><tbody>${tableRows}</tbody></table>`;
  renderChart(rows);
  renderBestInfo(rows);
}

let chartInstance=null;
function renderChart(rows){
  const methods = uniqueValues(rows.map(r=>r.method));
  const colors = {grid_search: '#1f77b4', random_search: '#ff7f0e', optuna_search: '#2ca02c'};

  const dataSets = methods.map(m=>{
    const subset = rows.filter(r=>r.method===m);
    if(subset.length===0) return null;

    const bestObj = subset.reduce((a,b)=>b.best_score > a.best_score ? b : a);
    const base = Number(bestObj.elapsed_seconds || 0);

    // puntos de todas combinaciones y mejor
    const allPoints = [];
    subset.forEach(r=>{
      (r.all_results || []).forEach(c=>{
        allPoints.push({
          x: base,
          y: Number(c.mean_score || 0)*100,
          radius: (Number(c.mean_score || 0) === bestObj.best_score ? 8 : 5),
          opacity: (Number(c.mean_score || 0) === bestObj.best_score ? 1.0 : 0.35),
          border: Number(c.mean_score || 0) === bestObj.best_score,
        });
      });
    });

    // punto best extra
    allPoints.push({x: base, y: bestObj.best_score*100, radius: 10, opacity:1, border:true});

    return {
      label: m,
      data: allPoints.map(p=>({x:p.x,y:p.y})),
      pointRadius: allPoints.map(p=>p.radius),
      pointBackgroundColor: allPoints.map(p=>`rgba(${hexToRgb(colors[m] || '#555')},${p.opacity})`),
      pointBorderColor: allPoints.map(p=>p.border ? '#000' : (colors[m] || '#555')),
      pointBorderWidth: allPoints.map(p=>p.border ? 1.8 : 0.8),
      showLine:false,
    };
  }).filter(x=>x!==null);

  const yValues = dataSets.flatMap(ds=>ds.data.map(pt=>pt.y));
  const yMin = Math.max(0, Math.min(...yValues) - 1);
  const yMax = Math.min(100, Math.max(...yValues) + 1);

  const ctx = document.getElementById('scatterChart');
  if(chartInstance) chartInstance.destroy();
  chartInstance = new Chart(ctx, {
    type:'scatter',
    data:{datasets:dataSets},
    options:{
      responsive:true,
      maintainAspectRatio:false,
      plugins:{legend:{position:'top'}},
      scales:{x:{title:{display:true,text:'Elapsed seconds'},beginAtZero:true},y:{title:{display:true,text:'Accuracy (%)'},min:yMin,max:yMax}}
    }
  });
}

function renderBestInfo(rows){
  const methods = uniqueValues(rows.map(r=>r.method));
  let html='';
  methods.forEach(m=>{
    const subset = rows.filter(r=>r.method===m);
    const best = subset.reduce((a,b)=>b.best_score>a.best_score ? b : a);
    html += `<p><strong>${m}</strong>: best ${(best.best_score*100).toFixed(3)}%, time ${best.elapsed_seconds.toFixed(2)}s</p>`;
  });
  document.getElementById('bestInfo').innerHTML = html;
}

function hexToRgb(hex){
  const h=hex.replace('#','');
  const bigint=parseInt(h,16);
  const r=(bigint>>16)&255;
  const g=(bigint>>8)&255;
  const b=bigint&255;
  return `${r},${g},${b}`;
}

buildDatasetSelector = buildDatasetSelector || buildDatasetSelector;
buildDatasetSelector();
</script>
</body>
</html>
"""


def _load_hp_results(path: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(path.glob("*_*_quick_results.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        dataset = p.name.split("_")[0] if "_" in p.name else "unknown"
        out.append({
            "dataset": dataset,
            "method": raw.get("method", "unknown"),
            "best_score": float(raw.get("best_score", 0.0)),
            "best_std": float(raw.get("best_std", 0.0)),
            "elapsed_seconds": float(raw.get("elapsed_seconds", 0.0)),
            "best_params": raw.get("best_params", {}),
            "all_results": raw.get("all_results", []),
            "file": str(p),
        })
    return out


def generate_dashboard(hp_dir: Path, out_html: Path) -> Path:
    hp_data = _load_hp_results(hp_dir)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__RAW_RESULTS__", json.dumps(hp_data, ensure_ascii=False))
    out_html.write_text(html, encoding="utf-8")
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hyperparameter search dashboard")
    parser.add_argument("--hp-dir", default="results", help="Hiperdirectory with JSON results")
    parser.add_argument("--out", default="visualization/hyperparam_dashboard.html", help="Output HTML path")
    args = parser.parse_args()

    generated = generate_dashboard(Path(args.hp_dir), Path(args.out))
    print(f"Got dashboard: {generated}")


if __name__ == "__main__":
    main()
