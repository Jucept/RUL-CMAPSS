RUL-CMAPSS: subset upload (src, notebooks, scripts, proposals, models CSVs). See full repo for details.

## Avcances
- Construcción del pipeline
- Diseño de un flujo rápido y reproducible para el dataset NASA C‑MAPSS FD001.
- Lectura de datos, cálculo de RUL, imputación causal simple, winsorization ligera y escalado con StandardScaler.
- Features temporales esenciales: medias y desviaciones móviles, pendientes (slopes), deltas desde el inicio y promedios de ventanas largas.
- Artefactos clave: X_train.parquet, y_train.parquet, X_test.parquet, scaler.pkl, features.json, metadata.json.
- No hay RUL negativos.
- Tocó hacer spot‑checks de causalidad en rolling features para asegurar que no se usara información futura.
- Entrenamiento inicial con **XGBoost**
- Validación cruzada por unidad (GroupKFold) → varios folds para medir robustez.
- Métricas iniciales obtenidas (baseline): RMSE ≈ 42, MAE ≈ 27.
- Suavizado exponencial (EWM).
- Esto reduce outliers y hace que las predicciones respeten la lógica física del problema.

## Explicación
- Pipeline: Es una versión ligera que prioriza velocidad sobre exhaustividad.
- Features rolling y slopes: capturan cómo cambian los sensores en ventanas cortas y largas; son indicadores de degradación progresiva.
- Validación por folds: Cada fold representa un conjunto distinto de motores usados como validación. Así evitamos depender de una sola partición y obtuvimos métricas más confiables.
- Resultados iniciales: RMSE y MAE te dicen el error promedio en ciclos de predicción. Aunque nos quedaron altos, sirven como referencia para mejoras.
