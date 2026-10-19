# Input:  traffic_features_v3.parquet
# Output: TFT_results_v3/tft_v3_best.ckpt
#         TFT_results_v3/tft_v3_config.json
#         TFT_results_v3/sensor_stats.csv
import os, json, time, warnings
warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

pl.seed_everything(42, workers=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH = "/home/iai-labor/traffic_project/traffic_features_v3.parquet"
RESULTS_DIR   = "/home/iai-labor/traffic_project/TFT_results_v3"
FIGURES_DIR   = "/home/iai-labor/traffic_project/TFT_figures_v3"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

CFG = {
    "sensors_per_class":      50,
    "max_encoder_length":     48,   # ← CHANGED from 24
    "max_prediction_length":   6,   # ← CHANGED from 3
    "hidden_size":            64,
    "attention_head_size":     4,
    "dropout":               0.1,
    "hidden_continuous_size": 16,
    "batch_size":            128,   # unchanged
    "max_epochs":            150,   # ← CHANGED from 100
    "patience":               15,   # ← CHANGED from 10
    "learning_rate":         1e-3,
    "gradient_clip_val":     0.1,
    "train_end":      "2025-06-01",
    "val_end":        "2025-12-01",
    "reduce_on_plateau_patience": 4,
}

SUBCLASS_NAMES = {0:"URB_normal", 1:"URB_ramp", 2:"M30", 3:"C30"}

# ── Feature column definitions — identical to v2 ───────────────────────────────
# All lag/rolling features normalised with same per-sensor μᵢ, σᵢ as target.

STATIC_REALS = [
    "road_subclass_enc", "longitude", "latitude", "is_main_artery",
    "dist_station_km", "dist_stadium_km", "dist_hospital_km",
    "dist_university_km", "dist_mall_km", "dist_bus_station_km",
    "near_major_poi",
]
TIME_VARYING_KNOWN_REALS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday", "is_rush_hour", "season",
]
TIME_VARYING_UNKNOWN_REALS = [
    "lag_1h_norm", "lag_2h_norm", "lag_3h_norm",
    "lag_24h_norm", "lag_168h_norm",
    "roll_mean_3h_norm", "roll_mean_24h_norm",
    "target_norm",
]

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = os.path.join(RESULTS_DIR, "tft_v3_training_log.txt")
def log(msg):
    from datetime import datetime
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")

log(f"Device: {device}")
if device == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("medium")

log("="*60)
log("TFT v3 — Extended Encoder and Horizon")
log(f"  encoder:    {CFG['max_encoder_length']}h  (was 24h in v2)")
log(f"  prediction: {CFG['max_prediction_length']}h   (was 3h in v2)")
log(f"  batch_size: {CFG['batch_size']}   (unchanged)")
log(f"  max_epochs: {CFG['max_epochs']}  (was 100 in v2)")
log(f"  patience:   {CFG['patience']}    (was 10 in v2)")
log("  Everything else identical to v2")
log("="*60)

# ── Step 1: Select sensors ─────────────────────────────────────────────────────
log("Selecting sensors...")
con = duckdb.connect()

sampled = con.sql(f"""
    WITH ranked AS (
        SELECT SENSOR_ID, road_subclass_enc, COUNT(*) AS n_obs,
               ROW_NUMBER() OVER (
                   PARTITION BY road_subclass_enc
                   ORDER BY COUNT(*) DESC, SENSOR_ID ASC
               ) AS rn
        FROM read_parquet('{FEATURES_PATH}')
        WHERE split    = 'train'
          AND target   IS NOT NULL
          AND lag_1h   IS NOT NULL
          AND lag_168h IS NOT NULL
        GROUP BY SENSOR_ID, road_subclass_enc
        HAVING COUNT(*) > 8760
    )
    SELECT SENSOR_ID, road_subclass_enc, n_obs
    FROM ranked
    WHERE rn <= {CFG['sensors_per_class']}
    ORDER BY road_subclass_enc, SENSOR_ID
""").df()

SENSOR_IDS = sampled["SENSOR_ID"].tolist()
id_list    = ",".join(map(str, SENSOR_IDS))
log(f"Sensors selected: {len(SENSOR_IDS)} ({CFG['sensors_per_class']} per sub-class)")

# ── Step 2: Load data ──────────────────────────────────────────────────────────
log("Loading data...")
t0 = time.time()

df = con.sql(f"""
    SELECT SENSOR_ID, DATETIME, target, road_subclass_enc, split,
           longitude, latitude, is_main_artery,
           dist_station_km, dist_stadium_km, dist_hospital_km,
           dist_university_km, dist_mall_km, dist_bus_station_km,
           near_major_poi,
           hour_sin, hour_cos, dow_sin, dow_cos,
           month_sin, month_cos,
           is_weekend, is_holiday, is_rush_hour, season,
           lag_1h, lag_2h, lag_3h, lag_24h, lag_168h,
           roll_mean_3h, roll_mean_24h
    FROM read_parquet('{FEATURES_PATH}')
    WHERE SENSOR_ID IN ({id_list})
      AND target   IS NOT NULL
      AND lag_1h   IS NOT NULL
    ORDER BY SENSOR_ID, DATETIME
""").df()

df["DATETIME"]   = pd.to_datetime(df["DATETIME"])
df               = df.sort_values(["SENSOR_ID", "DATETIME"]).reset_index(drop=True)
df["sensor_str"] = df["SENSOR_ID"].astype(str)
log(f"Loaded: {len(df):,} rows in {time.time()-t0:.1f}s")

# ── Step 3: Time index ─────────────────────────────────────────────────────────
time_origin    = df["DATETIME"].min()
df["time_idx"] = (
    (df["DATETIME"] - time_origin)
    .dt.total_seconds().div(3600).round().astype(int)
)
train_end_idx = int(
    (pd.Timestamp(CFG["train_end"]) - time_origin).total_seconds() / 3600
)
val_end_idx = int(
    (pd.Timestamp(CFG["val_end"]) - time_origin).total_seconds() / 3600
)
BUFFER     = CFG["max_encoder_length"]   # 48h — increased from 24h
train_mask = df["time_idx"] <= train_end_idx

log(f"train_end_idx={train_end_idx}  val_end_idx={val_end_idx}")
log(f"Buffer: {BUFFER}h  (encoder length, increased from 24h)")

# ── Step 4: Per-sensor normalisation ──────────────────────────────────────────
# Identical to v2 — per-sensor z-score, p99.9 cap, training data only.

log("Computing per-sensor normalisation statistics...")

TARGET_COLS = [
    "target", "lag_1h", "lag_2h", "lag_3h",
    "lag_24h", "lag_168h", "roll_mean_3h", "roll_mean_24h"
]
NORM_COLS = [
    "target_norm", "lag_1h_norm", "lag_2h_norm", "lag_3h_norm",
    "lag_24h_norm", "lag_168h_norm", "roll_mean_3h_norm", "roll_mean_24h_norm"
]

sensor_stats_list = []
for sid in SENSOR_IDS:
    sensor_train = df.loc[train_mask & (df["SENSOR_ID"] == sid), "target"]
    p999    = float(sensor_train.quantile(0.999))
    clean   = sensor_train[sensor_train <= p999]
    mu_i    = float(clean.mean())
    sigma_i = float(clean.std())
    if sigma_i < 1e-6 or np.isnan(sigma_i):
        sigma_i = 1.0
    sensor_stats_list.append({
        "SENSOR_ID": sid, "mu": mu_i, "sigma": sigma_i, "p999": p999,
    })

SENSOR_STATS = pd.DataFrame(sensor_stats_list).set_index("SENSOR_ID")
df["mu_i"]    = df["SENSOR_ID"].map(SENSOR_STATS["mu"])
df["sigma_i"] = df["SENSOR_ID"].map(SENSOR_STATS["sigma"])

for raw_col, norm_col in zip(TARGET_COLS, NORM_COLS):
    df[norm_col] = (df[raw_col] - df["mu_i"]) / df["sigma_i"]

# Verify
is_clean   = df.loc[train_mask, "target"] <= df.loc[
    train_mask, "SENSOR_ID"].map(SENSOR_STATS["p999"])
clean_norm = df.loc[train_mask, "target_norm"][is_clean.values]
all_norm   = df.loc[train_mask, "target_norm"]
log(f"  Clean pop: mean={clean_norm.mean():.4f} std={clean_norm.std():.4f} "
    f"(expected 0.000, 1.000)")
log(f"  Full pop:  mean={all_norm.mean():.4f}  max={all_norm.max():.2f}")

# ── Step 5: Split and NaN handling ────────────────────────────────────────────
static_cols = [
    "longitude","latitude","is_main_artery",
    "dist_station_km","dist_stadium_km","dist_hospital_km",
    "dist_university_km","dist_mall_km","dist_bus_station_km",
    "near_major_poi",
]
lag_cols_raw = [
    "lag_1h","lag_2h","lag_3h","lag_24h","lag_168h",
    "roll_mean_3h","roll_mean_24h",
]

# Buffer is now 48h — provides full encoder context for first val/test windows
df_train = df[df["time_idx"] <= train_end_idx].copy()
df_val   = df[
    (df["time_idx"] > train_end_idx - BUFFER) &
    (df["time_idx"] <= val_end_idx)
].copy()
df_test  = df[df["time_idx"] > val_end_idx - BUFFER].copy()

for col in static_cols:
    if df_train[col].isna().any():
        fill_val = float(df_train[col].median())
        for d in [df_train, df_val, df_test]:
            d[col] = d[col].fillna(fill_val)

df_train = df_train.dropna(subset=lag_cols_raw).reset_index(drop=True)
df_val   = df_val.dropna(subset=lag_cols_raw).reset_index(drop=True)
df_test  = df_test.dropna(subset=lag_cols_raw).reset_index(drop=True)

log(f"Split sizes (buffer={BUFFER}h):")
log(f"  Train: {len(df_train):,}")
log(f"  Val:   {len(df_val):,}  (includes {BUFFER}h buffer)")
log(f"  Test:  {len(df_test):,}  (includes {BUFFER}h buffer)")

# ── Step 6: Create TimeSeriesDataSet ──────────────────────────────────────────
# target="target_norm", target_normalizer=None — z-score loss, same as v2
# min_prediction_idx prevents buffer zone contamination — same as v2
# encoder=48h means windows are larger — main change from v2

log("Creating datasets...")
t0 = time.time()

training_dataset = TimeSeriesDataSet(
    df_train,
    time_idx  = "time_idx",
    target    = "target_norm",
    group_ids = ["sensor_str"],
    max_encoder_length    = CFG["max_encoder_length"],    # 48
    min_encoder_length    = CFG["max_encoder_length"],    # 48
    max_prediction_length = CFG["max_prediction_length"], # 6
    min_prediction_length = CFG["max_prediction_length"], # 6
    static_reals               = STATIC_REALS,
    time_varying_known_reals   = TIME_VARYING_KNOWN_REALS,
    time_varying_unknown_reals = TIME_VARYING_UNKNOWN_REALS,
    target_normalizer     = None,
    add_relative_time_idx = True,
    add_target_scales     = True,
    add_encoder_length    = True,
    allow_missing_timesteps = True,
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    df_val,
    min_prediction_idx = train_end_idx + 1,
    predict            = False,
    stop_randomization = True,
)

# Boundary assertions — hard checks prevent data leakage
assert validation_dataset.min_prediction_idx >= train_end_idx + 1, \
    "CONTAMINATION: val predictions start in training period"

train_loader = training_dataset.to_dataloader(
    train=True,
    batch_size  = CFG["batch_size"],       # 128
    num_workers = 4,
    pin_memory  = True,
)
val_loader = validation_dataset.to_dataloader(
    train=False,
    batch_size  = CFG["batch_size"] * 2,  # 256
    num_workers = 4,
    pin_memory  = True,
)

log(f"Datasets created in {time.time()-t0:.1f}s")
log(f"Train windows: {len(training_dataset):,}  "
    f"(fewer than v2 due to longer encoder)")
log(f"Val windows:   {len(validation_dataset):,}")
log(f"Train batches: {len(train_loader):,}  (batch={CFG['batch_size']})")
log(f"Val batches:   {len(val_loader):,}  "
    f"(batch={CFG['batch_size']*2})")
log(f"✓ Boundary assertion passed — "
    f"val starts at time_idx={validation_dataset.min_prediction_idx}")

# Verify z-score space
sample_x, sample_y = training_dataset[0]
enc_vals = sample_x["encoder_target"].numpy()
log(f"Sample encoder shape:  {enc_vals.shape}  (expected (48,))")
log(f"Sample encoder range:  [{enc_vals.min():.3f}, {enc_vals.max():.3f}]")
log(f"Sample prediction shape: {sample_y[0].shape}  (expected (6,))")
if enc_vals.max() > 20:
    log("WARNING: encoder values may be in veh/h — check normalisation")
else:
    log("✓ Encoder values confirmed in z-score space")
if enc_vals.shape[0] == CFG["max_encoder_length"]:
    log(f"✓ Encoder=48h confirmed")
if sample_y[0].shape[0] == CFG["max_prediction_length"]:
    log(f"✓ Prediction=6h confirmed")

# ── Step 7: Build TFT model ───────────────────────────────────────────────────

log("Building TFT v3 model...")
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    hidden_size            = CFG["hidden_size"],
    hidden_continuous_size = CFG["hidden_continuous_size"],
    attention_head_size    = CFG["attention_head_size"],
    dropout                = CFG["dropout"],
    loss                   = MAE(),
    log_interval           = 100,
    log_val_interval       = 1,
    learning_rate          = CFG["learning_rate"],
    reduce_on_plateau_patience = CFG["reduce_on_plateau_patience"],
)

total_params = sum(p.numel() for p in tft.parameters() if p.requires_grad)
log(f"  Parameters:  {total_params:,}")
log(f"  Loss:        MAE on z-score targets (same as v2)")
log(f"  Optimizer:   Adam lr={CFG['learning_rate']}")
log(f"  Scheduler:   ReduceLROnPlateau patience={CFG['reduce_on_plateau_patience']}")

# ── Step 8: Callbacks ─────────────────────────────────────────────────────────

class EpochSummary(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        m     = trainer.callback_metrics
        epoch = trainer.current_epoch
        tl    = float(m.get("train_loss_epoch", float("nan")))
        vl    = float(m.get("val_loss",          float("nan")))
        lr    = trainer.optimizers[0].param_groups[0]["lr"]
        ratio = vl / tl if (not np.isnan(tl) and tl > 1e-8) else float("nan")

        if not np.isnan(ratio):
            if ratio < 0.95:   status = "val<train"
            elif ratio < 1.05: status = "R≈1.0"
            elif ratio < 1.50: status = "val>train"
            else:              status = "⚠ large gap"
        else:
            status = ""

        # Patience counter — useful for monitoring how close to early stopping
        patience_count = 0
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                patience_count = cb.wait_count
                break

        log(f"Ep {epoch:3d} | train={tl:.4f} | val={vl:.4f} | "
            f"lr={lr:.2e} | R={ratio:.3f} | {status} | "
            f"patience={patience_count}/{CFG['patience']}")

        if torch.cuda.is_available():
            mem_a = torch.cuda.memory_allocated()/1e9
            mem_r = torch.cuda.memory_reserved()/1e9
            mem_f = torch.cuda.get_device_properties(0).total_memory/1e9 - mem_r
            log(f"         GPU: {mem_a:.2f}GB alloc | "
                f"{mem_r:.2f}GB reserved | {mem_f:.2f}GB free")

checkpoint_cb = ModelCheckpoint(
    dirpath    = RESULTS_DIR,
    filename   = "tft_v3_best",
    monitor    = "val_loss",
    mode       = "min",
    save_top_k = 1,
    verbose    = False,
)
earlystop_cb = EarlyStopping(
    monitor   = "val_loss",
    patience  = CFG["patience"],   # 15
    mode      = "min",
    min_delta = 1e-4,
    verbose   = True,
)

trainer = pl.Trainer(
    max_epochs          = CFG["max_epochs"],   # 150
    accelerator         = "gpu",
    devices             = 1,
    gradient_clip_val   = CFG["gradient_clip_val"],
    callbacks           = [
        checkpoint_cb,
        earlystop_cb,
        LearningRateMonitor("epoch"),
        EpochSummary(),
    ],
    logger = CSVLogger(RESULTS_DIR, name="tft_training_v3", version=0),
    enable_progress_bar = False,
    log_every_n_steps   = 50,
)

# ── Step 9: Train ─────────────────────────────────────────────────────────────
log("="*60)
log(f"TRAINING: TFT v3")
log(f"  encoder={CFG['max_encoder_length']}h | "
    f"prediction={CFG['max_prediction_length']}h | "
    f"batch={CFG['batch_size']}")
log(f"  max_epochs={CFG['max_epochs']} | patience={CFG['patience']}")
log(f"  Normalisation: per-sensor z-score (target_normalizer=None)")
log(f"  Loss: MAE on z-score targets — same as v2")
log("="*60)

t0 = time.time()
trainer.fit(
    tft,
    train_dataloaders = train_loader,
    val_dataloaders   = val_loader,
)
train_minutes = (time.time() - t0) / 60
best_val_loss = float(checkpoint_cb.best_model_score)
epochs_run    = trainer.current_epoch + 1

log(f"Training complete: {train_minutes:.1f} min ({train_minutes/60:.1f} hours)")
log(f"Epochs run:        {epochs_run}")
log(f"Best val_loss:     {best_val_loss:.4f} (z-score MAE)")
log(f"Checkpoint:        {checkpoint_cb.best_model_path}")

# ── Step 10: Save config and stats ────────────────────────────────────────────
config_out = {
    "model":           "TFT v3",
    "config":          CFG,
    "changes_from_v2": {
        "max_encoder_length":    "24 → 48",
        "max_prediction_length": "3 → 6",
        "max_epochs":            "100 → 150",
        "patience":              "10 → 15",
        "batch_size":            "128 (unchanged)",
    },
    "normalisation":   "per-sensor z-score, target_normalizer=None",
    "loss_function":   "MAE on z-score targets",
    "epochs_run":      epochs_run,
    "best_val_loss":   best_val_loss,
    "train_minutes":   round(train_minutes, 1),
    "checkpoint":      checkpoint_cb.best_model_path,
    "n_params":        total_params,
    "sensor_stats_path": os.path.join(RESULTS_DIR, "sensor_stats.csv"),
}
with open(os.path.join(RESULTS_DIR, "tft_v3_config.json"), "w") as f:
    json.dump(config_out, f, indent=2)

SENSOR_STATS.to_csv(os.path.join(RESULTS_DIR, "sensor_stats.csv"))

log("Config and SENSOR_STATS saved.")
log("Training complete.")
log(f"Next: run evaluation notebook with checkpoint "
    f"{checkpoint_cb.best_model_path}")