# RUL-CMAPSS — Predicción de RUL (FD001)

Autores: [Tu nombre]  
Repositorio original: https://github.com/Jucept/RUL-CMAPSS

Resumen
- Pipeline reproducible para predicción de Remaining Useful Life (RUL) sobre el dataset NASA C-MAPSS FD001.
- Incluye preprocesado, generación de features (rolling mean/std/slope), validaciones básicas, entrenamiento XGBoost y scripts de evaluación.

Estructura (relevante)
- src/ : código fuente (pipeline.py, train_xgb.py, preds.py, scripts/)
- notebooks/ : notebooks de EDA y análisis (EDA.ipynb, analysis_fastmode.ipynb)
- data/ : instrucciones para descargar los archivos raw (no se incluyen los datos)
- models/ : predicciones y métricas (CSV). Model binaries se omiten del push.
- pipeline_outputs/ : artefactos (features.json, metadata.json, plots, validation logs) — no subir datos masivos.
- diagnostics/ : CSVs con análisis por unit.
- docs/ : informe final, presentación y enlaces.

Reproducibilidad (resumen)
1. Preparar datos: colocar train_FD001.txt, test_FD001.txt, RUL_FD001.txt en data/raw/
2. Crear entorno: `pip install -r requirements.txt`
3. Ejecutar pipeline:  
   `python src/run_pipeline.py --raw_dir "data/raw"`
4. Entrenar modelo:  
   `python src/train_xgb.py`
5. Evaluar:  
   `python src/scripts/eval_from_preds.py --preds_dir models --out models/metrics_summary.csv`

Métricas reportadas
- RMSE, MAE, PHM08 (proxy en FAST_MODE = -RMSE). Ver `models/metrics_summary.csv`.

Limitaciones y próximos pasos
- PHM08 oficial no implementado (proxy usado).  
- En FAST_MODE se usan features reducidos; para producción habilitar pruning y búsqueda de features.

Referencias
- Saxena et al., NASA C-MAPSS dataset.