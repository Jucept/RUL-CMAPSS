# Resultados del EDA para FD001 (NASA C‑MAPSS)

## Archivos generados en esta carpeta

- **summary_stats.csv** : estadísticas generales incluyendo percentil 99 y porcentaje de valores faltantes  
- **per_unit_stats.csv** : estadísticas agregadas por unidad con valores iniciales y finales de sensores  
- **top_features.csv** : 12 características candidatas con su justificación  
- **rul_per_unit.csv** : RUL por unidad y por ciclo (conjunto de entrenamiento)  
- **pearson_sensors.csv**, **spearman_sensors.csv** : matrices de correlación  
- **eda_metadata.csv** : metadatos básicos (conteos, unidades seleccionadas)  
- **Figuras (PNG)**:  
  - count_units_cycles.png  
  - trajectory_samples.png  
  - correlation_heatmap.png  
  - pca_scree.png  
  - rul_trajectories.png  
  - acf_pacf_samples.png  
  - sensor_vs_op_hexbin.png  
  - summary_stats_table.png  

---

## Cómo reproducir

1. Asegúrate de tener los archivos raw:
  - train_FD001.txt  
  - test_FD001.txt  
  - RUL_FD001.txt  

2. Instala los requisitos:  
```bash
pip install -r requirements.txt
```

3. Ejecuta las celdas del notebook de arriba hacia abajo en Jupyter.
4. Todos los resultados se guardan en:
`./eda_outputs/`



