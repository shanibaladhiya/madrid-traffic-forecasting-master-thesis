## Environment

```
conda create -n traffic python=3.11
conda activate traffic
```

| Package | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.13.0+cu126 |
| pytorch-lightning | 2.2.4 |
| pytorch-forecasting | 1.8.0 |
| captum | 0.9.0 
| duckdb | 1.5.3 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| geopandas | 1.1.3 |
| shapely | 2.1.2 |
| osmnx | 2.1.0 |
| holidays | 0.99 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| statsmodels | 0.14.6 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| pyarrow | 23.0.1 |
| tqdm | 4.68.1 |
| requests | 2.34.2 |

Hardware used: NVIDIA RTX A2000, 12 GB VRAM.

## Data

Input file for all 3 models is traffic_features_v3.parquet(Approximately 3.9 GB) which is generated after preprocessing and used further training the models and evaluation. These datset generated from different 3 datasets which can be downloaded from https://datos.madrid.es/dataset/ . For getting idea about the input datast for all 3 models - It is uploaded online.

Link to Dataset: https://drive.google.com/file/d/1WXyk2lS9Thlt6DSljEIHE-ZpKqkD6Kzn/view?usp=sharing

Throught out the Thesis - Date range used: 2023-01-01 to 2026-05-31 (in code - traffic_features_v3.parquet)

Checkpoints from model, can be generated after applying dataset to the models.

## Running the pipeline
It can be used in order from 001 to 006 .ipynb files
Then train_cnn_lstm.py and then 008 file for evaluation of CNN+LSTM --> Same for N-beats(009 evaluation file) and TFT (010 Evaluation file)
Furthermore 011 for comparison of models and later 012 and 013 for explainability.

## Code Structure

### Data Pipeline
Run sequentially before any model training.

| File | Purpose |
|---|---|
| 001_data_download.ipynb | Download raw traffic CSV files and weather data from Madrid Open Data Portal |
| 002_data_ingestion_and_quality.ipynb | Load raw data into Parquet, detect physically impossible readings, flag stuck sensors |
| 003_road_attributes_and_classification.ipynb | Match sensors to OSM road attributes, build four-class road sub-class taxonomy |
| 004_imputation_and_final_dataset.ipynb | Tiered missing value imputation, seasonal fill, 30% fabrication cap, final clean dataset |
| 005_EDA.ipynb | Exploratory analysis of temporal patterns, sub-class distributions, and rush hour definition |
| 006_feature_engineering.ipynb | Build 28 engineered features, produce traffic_features_v3.parquet |

---

### Model Training
Each training script must be run before its evaluation notebook.
Training scripts are run from the terminal, not as notebooks.

**CNN+LSTM**
| File | Purpose |
|---|---|
| train_cnn_lstm.py | Train CNN+LSTM v2 — saves checkpoint and sensor_stats.csv |
| 008_evaluation_cnn_lstm.ipynb | Evaluate CNN+LSTM v2 on test set — produces results JSON and all figures |

**N-BEATS**
| File | Purpose |
|---|---|
| nbeats_v7_train.py | Train N-BEATS v7 — saves checkpoint and sensor_stats.csv |
| 009_evaluation_nbeats_v7.ipynb | Evaluate N-BEATS v7 on test set — produces results JSON and all figures |

**TFT**
| File | Purpose |
|---|---|
| train_tft_v3.py | Train TFT v3 — saves checkpoint and sensor_stats.csv |
| 010_evaluation_tft_v3.ipynb | Evaluate TFT v3 on test set — produces results JSON and all figures |

---

### Model Comparison
Run after all three evaluation notebooks are complete.

| File | Purpose |
|---|---|
| 011_model_comparison.ipynb | Generate cross-model comparison figures — per-horizon MASE curves, training efficiency scatter, combined snapshot forecast |

---

### Explainability
Run after all three evaluation notebooks and model comparison are complete.

| File | Purpose |
|---|---|
| 012_tft_interpretability.ipynb | TFT intrinsic explainability — variable selection weights, attention profile, per sub-class analysis |
| 013_shap_cnn_lstm.ipynb | CNN+LSTM GradientSHAP — overall importance, per sub-class, peak vs valley signed analysis, cross-model comparison |

---

### Environment
| File | Purpose |
|---|---|
| 000_environment.ipynb | Verify all library versions before running any other file |

## Reproducibility

Before running the scripts, configure the required input paths,
install the dependencies, and check any model-specific settings.

Execution instructions will be documented as the codebase is finalised.

## Research Context

Master's thesis:
*Traffic Forecasting in Madrid: An Explainable Deep Learning Approach*

## License

No reuse license has been specified yet. All rights are reserved
by default unless a license is added.
