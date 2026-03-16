# ASTD---Project3

Implementación de Time Series Forest Classifier usando la librería aeon.

## Requisitos

Instala dependencias desde la raíz del proyecto:

```bash
pip install -r requirements.txt
```

## Ejecución rápida

Ejecuta un experimento sintético de prueba:

```bash
python experiments/main_run.py
```

Parámetros opcionales:

```bash
python experiments/main_run.py --n-train 100 --n-test 40 --n-timestamps 120 --n-estimators 300 --seed 7
```

## Descargar datasets recomendados (UCR)

Para descargar ItalyPowerDemand, GunPoint, ECG5000, InlineSkate y ElectricDevices en la carpeta `data/`:

```bash
python data/download_ucr_datasets.py
```

El script descargará y validará tanto TRAIN como TEST para cada dataset.

## Archivo principal del clasificador

El wrapper del modelo TSF está en:

- classifiers/tsf_classifier.py