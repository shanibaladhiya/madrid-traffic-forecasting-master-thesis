import os, json, time, warnings
warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH = "/home/iai-labor/traffic_project/traffic_features_v3.parquet"
RESULTS_DIR   = "/home/iai-labor/traffic_project/CNN_LSTM_results_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
# Architecture follows Asadi & Regan (2020)
CFG = {
    # Data
    "sensors_per_class":      50,
    "max_encoder_length":     48,  
    "max_prediction_length":   6,   
    "train_end":      "2025-06-01",
    "val_end":        "2025-12-01",
    # Architecture
    "cnn_filters":            64,
    "cnn_kernel":              3,
    "lstm_hidden":           128,
    "lstm_layers":             2,
    "dropout":               0.1,
    # Training
    "batch_size":            128,   
    "max_epochs":            100,  
    "patience":               15, 
    "lr":                   1e-4,  
    "gradient_clip":         0.1,   
    "reduce_on_plateau_patience": 4,
    # Features
    "n_workers":               4,
}

SUBCLASS_NAMES = {0:"URB_normal", 1:"URB_ramp", 2:"M30", 3:"C30"}

# ── Feature columns ──────────────────────────────────────────
# Static sensor properties (do not change over time)
STATIC_COLS = [
    "road_subclass_enc", "longitude", "latitude", "is_main_artery",
    "dist_station_km", "dist_stadium_km", "dist_hospital_km",
    "dist_university_km", "dist_mall_km", "dist_bus_station_km",
    "near_major_poi",
]
# Calendar features (known for any future date)
CALENDAR_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday", "is_rush_hour", "season",
]
# Lag features — normalised with same per-sensor stats as target
LAG_COLS_NORM = [
    "lag_1h_norm", "lag_2h_norm", "lag_3h_norm",
    "lag_24h_norm", "lag_168h_norm",
    "roll_mean_3h_norm", "roll_mean_24h_norm",
]
# Raw lag columns — used for normalisation and NaN dropping
LAG_COLS_RAW = [
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h",
    "roll_mean_3h", "roll_mean_24h",
]

# All feature columns fed to the model encoder
FEATURE_COLS = STATIC_COLS + CALENDAR_COLS + LAG_COLS_NORM + ["target_norm"]
N_FEATURES   = len(FEATURE_COLS)

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = os.path.join(RESULTS_DIR, "cnn_lstm_training_log_v2.txt")
def log(msg):
    from datetime import datetime
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")

log(f"Device: {device}")
if device.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("medium")

log("=" * 60)
log("CNN+LSTM Forecasting — train_cnn_lstm.py")
log(f"  encoder:    {CFG['max_encoder_length']}h")
log(f"  prediction: {CFG['max_prediction_length']}h  (MIMO — all horizons simultaneously)")
log(f"  features:   {N_FEATURES}")
log(f"  batch:      {CFG['batch_size']}")
log(f"  max_epochs: {CFG['max_epochs']} | patience: {CFG['patience']}")
log(f"  loss:       MAE on z-score targets")
log(f"  train_loss: FrozenTrainEval")
log(f"  Normalisation: per-sensor z-score")
log("=" * 60)

# ── Step 1: Sensor selection ───────────────────────────────────────────────────
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
df               = df.sort_values(["SENSOR_ID","DATETIME"]).reset_index(drop=True)
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
BUFFER     = CFG["max_encoder_length"]  # 48h
train_mask = df["time_idx"] <= train_end_idx

log(f"train_end_idx={train_end_idx} | val_end_idx={val_end_idx} | buffer={BUFFER}h")

# ── Step 4: Per-sensor normalisation  ─────
#
# Per-sensor z-score: compute μᵢ, σᵢ from training data only

log("Computing per-sensor normalisation statistics...")

TARGET_COLS = [
    "target","lag_1h","lag_2h","lag_3h",
    "lag_24h","lag_168h","roll_mean_3h","roll_mean_24h"
]
NORM_COLS = [
    "target_norm","lag_1h_norm","lag_2h_norm","lag_3h_norm",
    "lag_24h_norm","lag_168h_norm","roll_mean_3h_norm","roll_mean_24h_norm"
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
        "SENSOR_ID": sid, "mu": mu_i,
        "sigma": sigma_i, "p999": p999,
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

# ── Step 5: Splits and NaN handling ───────────────────────────────────────────
static_cols = [
    "longitude","latitude","is_main_artery",
    "dist_station_km","dist_stadium_km","dist_hospital_km",
    "dist_university_km","dist_mall_km","dist_bus_station_km",
    "near_major_poi",
]

df_train = df[df["time_idx"] <= train_end_idx].copy()
df_val   = df[
    (df["time_idx"] > train_end_idx - BUFFER) &
    (df["time_idx"] <= val_end_idx)
].copy()

for col in static_cols:
    if df_train[col].isna().any():
        fill_val = float(df_train[col].median())
        for d in [df_train, df_val]:
            d[col] = d[col].fillna(fill_val)

df_train = df_train.dropna(subset=LAG_COLS_RAW).reset_index(drop=True)
df_val   = df_val.dropna(subset=LAG_COLS_RAW).reset_index(drop=True)

log(f"Split sizes (buffer={BUFFER}h):")
log(f"  Train: {len(df_train):,}")
log(f"  Val:   {len(df_val):,}")

# ── Step 6: Dataset class ──────────────────────────────────────────────────────
#
# Sliding window MIMO forecasting dataset.
# Each sample:
#   X: (encoder_length, n_features) — 48h of feature history
#   y: (prediction_length,)         — 6 future z-score target values
#
# Boundary enforcement:
#   Validation windows: decoder targets start at train_end_idx + 1
#   This mirrors the min_prediction_idx used in PyTorch Forecasting
#
# No data leakage:
#   - SENSOR_STATS computed from training data only
#   - Validation split uses training SENSOR_STATS for normalisation
#   - Lag features in z-score space (same μᵢ, σᵢ as target)

class ForecastDataset(Dataset):
    """
    Sliding window MIMO forecasting dataset for CNN+LSTM.

    Produces windows of:
        X: (encoder_len, n_features) normalised feature history
        y: (pred_len,) normalised future target values
        sensor_id: str — for inverse transform in evaluation
        t_start: int — decoder start time_idx for naive baseline
    """
    def __init__(self, df_split, split_name,
                 encoder_len, pred_len,
                 min_pred_idx=None):
        self.windows   = []
        self.sensor_ids = []
        self.t_starts  = []
        enc  = encoder_len
        pred = pred_len
        sensors = df_split["sensor_str"].unique()

        for sid in sensors:
            sdf = df_split[
                df_split["sensor_str"] == sid
            ].sort_values("time_idx").reset_index(drop=True)

            if len(sdf) < enc + pred:
                continue

            feats   = sdf[FEATURE_COLS].values.astype(np.float32)
            targets = sdf["target_norm"].values.astype(np.float32)
            t_idxs  = sdf["time_idx"].values

            for i in range(len(sdf) - enc - pred + 1):
                decoder_start = int(t_idxs[i + enc])
                # Boundary check — decoder targets must be
                # strictly after min_pred_idx
                if min_pred_idx is not None:
                    if decoder_start < min_pred_idx:
                        continue
                X = feats[i: i + enc]              # (enc, n_features)
                y = targets[i + enc: i + enc + pred]  # (pred,)
                self.windows.append((X, y))
                self.sensor_ids.append(sid)
                self.t_starts.append(decoder_start)

        log(f"  {split_name}: {len(self.windows):,} windows "
            f"from {len(sensors)} sensors")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        X, y = self.windows[idx]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

log("Building datasets...")
t0 = time.time()

train_ds = ForecastDataset(
    df_train, "train",
    CFG["max_encoder_length"],
    CFG["max_prediction_length"],
    min_pred_idx = None,  # no constraint for training
)
val_ds = ForecastDataset(
    df_val, "val",
    CFG["max_encoder_length"],
    CFG["max_prediction_length"],
    min_pred_idx = train_end_idx + 1,  # boundary enforcement
)
log(f"Datasets built in {time.time()-t0:.1f}s")

# Verify shapes
X_sample, y_sample = train_ds[0]
assert X_sample.shape == (CFG["max_encoder_length"], N_FEATURES), \
    f"X shape mismatch: {X_sample.shape}"
assert y_sample.shape == (CFG["max_prediction_length"],), \
    f"y shape mismatch: {y_sample.shape}"
log(f"✓ X shape: {tuple(X_sample.shape)}  (expected ({CFG['max_encoder_length']}, {N_FEATURES}))")
log(f"✓ y shape: {tuple(y_sample.shape)}  (expected ({CFG['max_prediction_length']},))")

train_loader = DataLoader(
    train_ds,
    batch_size  = CFG["batch_size"],
    shuffle     = True,
    num_workers = CFG["n_workers"],
    pin_memory  = True,
)
val_loader = DataLoader(
    val_ds,
    batch_size  = CFG["batch_size"] * 2,
    shuffle     = False,
    num_workers = CFG["n_workers"],
    pin_memory  = True,
)
# Frozen train eval loader — no shuffle, no augmentation
# Used by FrozenTrainEval for comparable train loss
train_eval_loader = DataLoader(
    train_ds,
    batch_size  = CFG["batch_size"] * 2,
    shuffle     = False,
    num_workers = CFG["n_workers"],
    pin_memory  = True,
)

log(f"Train batches: {len(train_loader):,}  (batch={CFG['batch_size']})")
log(f"Val batches:   {len(val_loader):,}  (batch={CFG['batch_size']*2})")
log(f"Train eval batches: {len(train_eval_loader):,}  (for frozen eval)")

# ── Step 7: CNN+LSTM model ─────────────────────────────────────────────────────
# Input:  (batch, encoder_len, n_features)
# Output: (batch, pred_len) — z-score predictions for t+1h..t+6h

class CNNLSTMForecaster(nn.Module):
    def __init__(self, n_features, pred_len, cfg):
        super().__init__()
        # ── CNN block ─────────────────────────────────────────────
        # Permute to (batch, n_features, seq_len) for Conv1d
        # Extracts local 3h temporal patterns across all features
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels  = n_features,
                out_channels = cfg["cnn_filters"],
                kernel_size  = cfg["cnn_kernel"],
                padding      = cfg["cnn_kernel"] // 2,
            ),
            nn.BatchNorm1d(cfg["cnn_filters"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )
        # ── LSTM block ────────────────────────────────────────────
        # Receives CNN output: (batch, seq_len, cnn_filters)
        # Two stacked layers capture hierarchical temporal patterns
        self.lstm = nn.LSTM(
            input_size  = cfg["cnn_filters"],
            hidden_size = cfg["lstm_hidden"],
            num_layers  = cfg["lstm_layers"],
            batch_first = True,
            dropout     = cfg["dropout"] if cfg["lstm_layers"] > 1 else 0.0,
        )
        # ── MIMO forecast head ────────────────────────────────────
        # Projects final LSTM hidden state to all 6 horizons
        # simultaneously — no autoregressive error accumulation
        self.head = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["lstm_hidden"], 64),
            nn.ReLU(),
            nn.Linear(64, pred_len),  # → (batch, pred_len)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)      # (batch, n_features, seq_len)
        x = self.cnn(x)              # (batch, cnn_filters, seq_len)
        x = x.permute(0, 2, 1)      # (batch, seq_len, cnn_filters)
        _, (h_n, _) = self.lstm(x)  # h_n: (n_layers, batch, hidden)
        x = h_n[-1]                  # (batch, hidden) — last layer
        return self.head(x)          # (batch, pred_len)

model = CNNLSTMForecaster(
    n_features = N_FEATURES,
    pred_len   = CFG["max_prediction_length"],
    cfg        = CFG,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"CNN+LSTM parameters: {total_params:,}")
log(f"  CNN filters: {CFG['cnn_filters']} | kernel: {CFG['cnn_kernel']}")
log(f"  LSTM hidden: {CFG['lstm_hidden']} | layers: {CFG['lstm_layers']}")
log(f"  Dropout: {CFG['dropout']} | Head: {CFG['lstm_hidden']} → 64 → {CFG['max_prediction_length']}")

# ── Step 8: Loss, optimizer, scheduler ────────────────────────────────────────
criterion = nn.L1Loss()  # MAE loss 
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = CFG["lr"],
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode     = "min",
    factor   = 0.5,
    patience = CFG["reduce_on_plateau_patience"],
)

log(f"Loss: MAE (nn.L1Loss) on z-score targets")
log(f"Optimizer: Adam lr={CFG['lr']}")
log(f"Scheduler: ReduceLROnPlateau patience={CFG['reduce_on_plateau_patience']}")

# ── Step 9: Training loop with FrozenTrainEval ────────────────────────────────
#
# FrozenTrainEval: compute train loss with model.eval() after weight updates

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = os.path.join(RESULTS_DIR, "cnn_lstm_v2_best.pt")
history = {
    "train_loss_frozen": [],
    "val_loss":          [],
    "lr":                [],
}

log("=" * 60)
log("TRAINING")
log(f"  Frozen eval: full training pass after each epoch weight update")
log("=" * 60)

t_total = time.time()

for epoch in range(1, CFG["max_epochs"] + 1):
    t_ep = time.time()

    # ── Training pass ──────────────────────────────────────────────
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm = CFG["gradient_clip"]
        )
        optimizer.step()

    # ── Frozen training eval ───────────────────────────────────────
    # Identical computation to validation loss:
    # model.eval(), no dropout, no gradient, full training data
    model.eval()
    frozen_total = 0.0
    frozen_n     = 0
    with torch.no_grad():
        for X_batch, y_batch in train_eval_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds   = model(X_batch)
            loss    = criterion(preds, y_batch)
            frozen_total += loss.item() * X_batch.size(0)
            frozen_n     += X_batch.size(0)
    train_loss_frozen = frozen_total / frozen_n

    # ── Validation pass ────────────────────────────────────────────
    val_total = 0.0
    val_n     = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds   = model(X_batch)
            loss    = criterion(preds, y_batch)
            val_total += loss.item() * X_batch.size(0)
            val_n     += X_batch.size(0)
    val_loss = val_total / val_n

    # ── Scheduler step ─────────────────────────────────────────────
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # ── Generalisation ratio ───────────────────────────────────────
    ratio = val_loss / train_loss_frozen if train_loss_frozen > 1e-8 \
        else float("nan")
    if ratio < 0.95:   status = "val<train"
    elif ratio < 1.05: status = "R≈1.0"
    elif ratio < 1.50: status = "val>train"
    else:              status = "⚠ large gap"

    # ── Early stopping ─────────────────────────────────────────────
    if val_loss < best_val_loss - 1e-4:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1

    # ── GPU memory ─────────────────────────────────────────────────
    mem_a = torch.cuda.memory_allocated() / 1e9  if device.type == "cuda" else 0
    mem_r = torch.cuda.memory_reserved()  / 1e9  if device.type == "cuda" else 0
    mem_f = (torch.cuda.get_device_properties(0).total_memory / 1e9 - mem_r) \
            if device.type == "cuda" else 0

    # ── Log ────────────────────────────────────────────────────────
    ep_min = (time.time() - t_ep) / 60
    log(f"Ep {epoch:3d} | "
        f"train={train_loss_frozen:.4f} | "
        f"val={val_loss:.4f} | "
        f"lr={current_lr:.2e} | "
        f"R={ratio:.3f} | {status} | "
        f"patience={patience_counter}/{CFG['patience']} | "
        f"{ep_min:.1f}min")
    log(f"         GPU: {mem_a:.2f}GB alloc | "
        f"{mem_r:.2f}GB reserved | {mem_f:.2f}GB free")

    history["train_loss_frozen"].append(train_loss_frozen)
    history["val_loss"].append(val_loss)
    history["lr"].append(current_lr)

    # ── Early stopping trigger ─────────────────────────────────────
    if patience_counter >= CFG["patience"]:
        log(f"Early stopping at epoch {epoch} — "
            f"no improvement for {CFG['patience']} epochs")
        break

# ── Step 10: Save everything ───────────────────────────────────────────────────
train_minutes = (time.time() - t_total) / 60
epochs_run    = epoch

log("=" * 60)
log(f"Training complete: {train_minutes:.1f} min ({train_minutes/60:.1f} hours)")
log(f"Epochs run:        {epochs_run}")
log(f"Best val_loss:     {best_val_loss:.4f} z-score MAE")
log(f"Checkpoint:        {best_model_path}")
log("=" * 60)

# Save training history
with open(os.path.join(RESULTS_DIR, "cnn_lstm_v2_history.json"), "w") as f:
    json.dump(history, f, indent=2)

# Save SENSOR_STATS for evaluation inverse transform
SENSOR_STATS.to_csv(os.path.join(RESULTS_DIR, "sensor_stats.csv"))

# Save config
config_out = {
    "model":           "CNN+LSTM Forecaster",
    "config":          CFG,
    "n_features":      N_FEATURES,
    "feature_cols":    FEATURE_COLS,
    "normalisation":   "per-sensor z-score",
    "loss_function":   "MAE (nn.L1Loss) on z-score targets",
    "train_loss_method": "FrozenTrainEval",
    "epochs_run":      epochs_run,
    "best_val_loss":   best_val_loss,
    "train_minutes":   round(train_minutes, 1),
    "checkpoint":      best_model_path,
    "n_params":        total_params,
    "sensor_stats_path": os.path.join(RESULTS_DIR, "sensor_stats.csv"),
}
with open(os.path.join(RESULTS_DIR, "cnn_lstm_v2_config.json"), "w") as f:
    json.dump(config_out, f, indent=2)

log("Config, history, and SENSOR_STATS saved.")
log(f"Next: run evaluation notebook with checkpoint {best_model_path}")