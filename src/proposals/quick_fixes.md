# Quick fixes (FAST_MODE iteration)

## Context
Resultados iniciales (FAST_MODE): RMSE  42.2, MAE  27.0 (fold 0). Baseline rápido sin pruning ni features avanzadas.

## Fix 1  Monotonic smoothing por unidad (postprocess)
- Objetivo: reducir outliers y retornos no plausibles en la predicción RUL.
- Descripción:
  - Para cada unidad, aplicar un suavizado exponencial o isotónica sobre la serie de predicciones por ciclo.
  - Garantizar monotonía no creciente en RUL (o no negativa): clip y proyectar para evitar saltos.
- Implementación rápida:
  - Aplicar scipy.signal.savgol_filter o pandas.Series.ewm(alpha=0.2) sobre predicciones por unit-cycles.
  - Luego forzar RUL_smoothed = cummax(RUL_smoothed[::-1])[::-1] si queremos que RUL decrezca de forma suave.
- Riesgo / beneficio: muy barato, suele mejorar RMSE/MAE para outliers aislados.

## Fix 2  Añadir slopes y ventanas extensas / cortas
- Objetivo: capturar dinámicas rápidas y tendencias largas que afectan fallo.
- Descripción:
  - Añadir windows cortas (w=3) y largas (w=50) para rm, rstd y slope.
  - Mantener FAST_MODE acumulando en dict para evitar fragmentación.
- Implementación rápida:
  - En pipeline.py añadir windows = [3,5,20,50] en el modo estándar y recomputar solo dichas columnas.
  - Evaluar correlación rápida y descartar solo features con varianza cero; dejar pruning más exhaustivo para experimentos posteriores.
- Riesgo / beneficio: coste moderado en CPU pero suele mejorar sensibilidad a cambios bruscos y degradación lenta.

## Notes
- Mantener scaler original (StandardScaler-fast) para compatibilidad.
- Si se quiere iterar rápido: implementar Fix 1 primero (postproc), medir mejora; luego Fix 2 si la ganancia es limitada.
