import argparse
import json
from pathlib import Path
from typing import Any, Dict


def choose_best_run(runs: list[Dict[str, Any]]) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None

    for run in runs:
        if best is None:
            best = run
            continue

        current_score = float(run.get("best_score", -float("inf")))
        best_score = float(best.get("best_score", -float("inf")))

        if current_score > best_score:
            best = run
            continue

        if current_score == best_score:
            current_n_est = int(run.get("best_params", {}).get("n_estimators", float("inf")))
            best_n_est = int(best.get("best_params", {}).get("n_estimators", float("inf")))
            if current_n_est < best_n_est:
                best = run

    assert best is not None
    return best


def extract_best_of_best(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Input JSON must contain 'runs' as a list")

    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for run in runs:
        dataset = run.get("dataset")
        if dataset is None:
            continue
        grouped.setdefault(dataset, []).append(run)

    result: Dict[str, Any] = {
        "input_file": str(input_path),
        "dataset_count": len(grouped),
        "best_runs": {},
    }

    for dataset, dataset_runs in sorted(grouped.items()):
        best_run = choose_best_run(dataset_runs)
        result["best_runs"][dataset] = best_run

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae el mejor resultado por dataset de un resumen de búsqueda de hiperparámetros TSF")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/tsf_hyperparameter_search_summary.json"),
        help="ruta del archivo JSON de resumen (por defecto: results/tsf_hyperparameter_search_summary.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ruta de salida para guardar el resultado JSON (por defecto: imprime en stdout)",
    )

    args = parser.parse_args()
    output_data = extract_best_of_best(args.input)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Mejores hiperparámetros escritos en {args.output}")
    else:
        print(json.dumps(output_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
