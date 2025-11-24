
EDA outputs for FD001 (NASA C-MAPSS)

Files generated in this folder:
- summary_stats.csv : overall summary stats including 99th percentile and missing_pct.
- per_unit_stats.csv : aggregated per-unit stats with first/last sensor values.
- top_features.csv : 12 candidate features with rationale.
- rul_per_unit.csv : RUL per unit per cycle (train set).
- pearson_sensors.csv, spearman_sensors.csv : correlation matrices.
- eda_metadata.csv : basic metadata (counts, selected units).
- Figures (png): count_units_cycles.png, trajectory_samples.png, correlation_heatmap.png,
                pca_scree.png, rul_trajectories.png, acf_pacf_samples.png, sensor_vs_op_hexbin.png, summary_stats_table.png

How to reproduce:
1. Ensure raw files are at: C:\Users\jucep\OneDrive\Escritorio\Proyecto CMAPSS\MANTENIMIENTO-PREDICTIVO\CMAPSSData\raw
   - train_FD001.txt
   - test_FD001.txt
   - RUL_FD001.txt
2. Install requirements:
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
3. Run the notebook cells top-to-bottom in Jupyter.
4. All outputs are saved to ./eda_outputs/
