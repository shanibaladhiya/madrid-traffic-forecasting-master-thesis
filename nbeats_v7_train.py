# N-BEATS v7 Training
# Input:  traffic_features_v3.parquet
# Output: nbeats_experiments_v7/exp01_enc72_pred6/best-v1.ckpt
#         nbeats_experiments_v7/exp01_enc72_pred6/results.json
#         nbeats_experiments_v7/exp01_enc72_pred6/sensor_stats.csv

import os
import json
import time
import types
import warnings
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
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE

pl.seed_everything(42, workers=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision("medium")

print(f"Device: {device}")
if device == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print(f"PyTorch:           {torch.__version__}")
print(f"PyTorch Lightning: {pl.__version__}")

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "/home/iai-labor/traffic_project/traffic_features_v3.parquet"
EXP_BASE_DIR  = "/home/iai-labor/traffic_project/nbeats_experiments_v7"
FIG_DIR       = "/home/iai-labor/traffic_project/nbeats_figures_v7"
os.makedirs(EXP_BASE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,      exist_ok=True)

SUBCLASS_NAMES = {0:"URB_normal", 1:"URB_ramp", 2:"M30", 3:"C30"}

# ── Configuration ─────────────────────────────────────────────────────────────
FIXED = {
    # Data
    "sensors_per_class":   50,
    "train_end":    "2025-06-01",
    "val_end":      "2025-12-01",

    # Sequence lengths  ← CHANGED from v6
    "max_encoder_length":    72,   # was 48 — three diurnal cycles
    "max_prediction_length":  6,   # unchanged

    # Architecture — identical to v6
    "stack_types":                   ["trend", "seasonality"],
    "num_block_layers":              [4, 4],
    "sharing":                       [True, True],
    "expansion_coefficient_lengths": [3, 7],
    "dropout":                       0.1,

    # Training  ← batch and epochs changed
    "batch_size":  128,    # was 256 — halved for larger encoder windows
    "patience":    15,
    "min_delta":   1e-5,
    "max_epochs":  150,    # was 100 — more time for scheduler to decay

    # Scheduler — identical to v6
    "reduce_on_plateau_patience": 4,

    # Weight decay — identical to v6
    "weight_decay": 1e-4,
}

EXPERIMENTS = [
    {
        "name":        "exp01_enc72_pred6",
        "description": (
            "Encoder extended 48h→72h. Three complete diurnal cycles. "
            "batch=128 (memory-matched to v6). "
            "blocks=[4,4] widths=[256,2048] lr=1e-4 wd=1e-4. "
            "Purely univariate — no lag features. "
            "Reference: Oreshkin et al. (2020) 2x-7x lookback ratio."
        ),
        "num_blocks":  [4, 4],
        "widths":      [256, 2048],
        "lr":          1e-4,
    },
]

EXP_INDEX = 0
exp = EXPERIMENTS[EXP_INDEX]

print(f"\nN-BEATS v7 — Extended Encoder 72h")
print(f"{'='*60}")
print(f"Encoder:       {FIXED['max_encoder_length']}h  (was 48h in v6)")
print(f"Prediction:    {FIXED['max_prediction_length']}h")
print(f"Batch size:    {FIXED['batch_size']}    (was 256 in v6)")
print(f"Max epochs:    {FIXED['max_epochs']}   (was 100 in v6)")
print(f"Weight decay:  λ={FIXED['weight_decay']}")
print(f"Scheduler:     ReduceLROnPlateau "
      f"patience={FIXED['reduce_on_plateau_patience']} factor=0.5")
print(f"Experiment:    {exp['name']}")
print(f"  blocks={exp['num_blocks']}  widths={exp['widths']}  lr={exp['lr']}")
print(f"  Univariate: target_norm only — no lag features")
print(f"{'='*60}")
print(f"  Experiments: {EXP_BASE_DIR}")
print(f"  Figures:     {FIG_DIR}")

# ── Sensor selection ──────────────────────────────────────────────────────────
print("\nSelecting sensors...")
con = duckdb.connect()
sampled = con.sql(f"""
    WITH ranked AS (
        SELECT SENSOR_ID, road_subclass_enc, COUNT(*) AS n_obs,
               ROW_NUMBER() OVER (
                   PARTITION BY road_subclass_enc
                   ORDER BY COUNT(*) DESC, SENSOR_ID ASC
               ) AS rn
        FROM read_parquet('{FEATURES_PATH}')
        WHERE split='train' AND target IS NOT NULL
          AND lag_168h IS NOT NULL
        GROUP BY SENSOR_ID, road_subclass_enc
        HAVING COUNT(*) > 8760
    )
    SELECT SENSOR_ID, road_subclass_enc, n_obs
    FROM ranked
    WHERE rn <= {FIXED['sensors_per_class']}
    ORDER BY road_subclass_enc, SENSOR_ID
""").df()

SENSOR_IDS = sampled["SENSOR_ID"].tolist()
id_list    = ",".join(map(str, SENSOR_IDS))
print(f"  Sensors selected: {len(SENSOR_IDS)}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
t0 = time.time()
df = con.sql(f"""
    SELECT SENSOR_ID, DATETIME, target, road_subclass_enc, split
    FROM read_parquet('{FEATURES_PATH}')
    WHERE SENSOR_ID IN ({id_list}) AND target IS NOT NULL
    ORDER BY SENSOR_ID, DATETIME
""").df()
df["DATETIME"]   = pd.to_datetime(df["DATETIME"])
df               = df.sort_values(
    ["SENSOR_ID","DATETIME"]).reset_index(drop=True)
df["sensor_str"] = df["SENSOR_ID"].astype(str)
print(f"  Rows: {len(df):,}  ({time.time()-t0:.1f}s)")

# ── Time index ────────────────────────────────────────────────────────────────
time_origin    = df["DATETIME"].min()
df["time_idx"] = (
    (df["DATETIME"] - time_origin)
    .dt.total_seconds().div(3600).round().astype(int)
)
train_end_idx = int(
    (pd.Timestamp(FIXED["train_end"]) - time_origin).total_seconds()/3600
)
val_end_idx = int(
    (pd.Timestamp(FIXED["val_end"]) - time_origin).total_seconds()/3600
)
BUFFER     = FIXED["max_encoder_length"]   # 72h — changed from 48h
train_mask = df["time_idx"] <= train_end_idx
print(f"  train_end_idx={train_end_idx}  val_end_idx={val_end_idx}")
print(f"  Buffer: {BUFFER}h  (encoder length — increased from 48h)")

# ── Per-sensor normalisation ──────────────────────────────────────────────────
# Identical to v6 — per-sensor z-score, p99.9 cap, training data only
print("Computing normalisation statistics...")
sensor_stats_list = []
for sid in SENSOR_IDS:
    st      = df.loc[train_mask & (df["SENSOR_ID"]==sid), "target"]
    p999    = float(st.quantile(0.999))
    clean   = st[st <= p999]
    mu_i    = float(clean.mean())
    sigma_i = float(clean.std())
    if sigma_i < 1e-6 or np.isnan(sigma_i):
        sigma_i = 1.0
    sensor_stats_list.append({
        "SENSOR_ID": sid, "mu": mu_i, "sigma": sigma_i, "p999": p999,
        "n_excluded": int((st > p999).sum())
    })

SENSOR_STATS = pd.DataFrame(sensor_stats_list).set_index("SENSOR_ID")
df["mu_i"]        = df["SENSOR_ID"].map(SENSOR_STATS["mu"])
df["sigma_i"]     = df["SENSOR_ID"].map(SENSOR_STATS["sigma"])
df["target_norm"] = (df["target"] - df["mu_i"]) / df["sigma_i"]

# Verify
is_clean   = df.loc[train_mask,"target"] <= df.loc[
    train_mask,"SENSOR_ID"].map(SENSOR_STATS["p999"])
clean_norm = df.loc[train_mask,"target_norm"][is_clean.values]
all_norm   = df.loc[train_mask,"target_norm"]
print(f"  Clean pop: mean={clean_norm.mean():.4f} "
      f"std={clean_norm.std():.4f}  (expected 0.000, 1.000)")
print(f"  Full pop:  mean={all_norm.mean():.4f}  "
      f"max={all_norm.max():.2f}")

# ── Splits ────────────────────────────────────────────────────────────────────
# Buffer is now 72h — provides full encoder context for first val window
df_train = df[df["time_idx"] <= train_end_idx].copy()
df_val   = df[
    (df["time_idx"] > train_end_idx - BUFFER) &
    (df["time_idx"] <= val_end_idx)
].copy()
df_test  = df[df["time_idx"] > val_end_idx - BUFFER].copy()

print(f"\nSplit sizes (buffer={BUFFER}h):")
print(f"  Train: {len(df_train):>10,}")
print(f"  Val:   {len(df_val):>10,}  (includes {BUFFER}h buffer)")
print(f"  Test:  {len(df_test):>10,}  (includes {BUFFER}h buffer)")

gap_check = (
    df.groupby("SENSOR_ID")["time_idx"]
      .apply(lambda x: (x.diff().dropna() != 1).sum())
)
print(f"  Time continuity: {(gap_check>0).sum()} sensors with gaps — "
      f"{'clean' if (gap_check>0).sum()==0 else 'GAPS DETECTED'}")

# ── TimeSeriesDataSet ─────────────────────────────────────────────────────────
print("\nCreating datasets...")
t0 = time.time()

training_dataset = TimeSeriesDataSet(
    df_train,
    time_idx      = "time_idx",
    target        = "target_norm",
    group_ids     = ["sensor_str"],

    # Extended encoder — key v7 change
    max_encoder_length    = FIXED["max_encoder_length"],    # 72
    min_encoder_length    = FIXED["max_encoder_length"],    # 72
    max_prediction_length = FIXED["max_prediction_length"], # 6
    min_prediction_length = FIXED["max_prediction_length"], # 6

    # Univariate — only target_norm, no lag features
    # Preserves original N-BEATS univariate architecture
    time_varying_unknown_reals = ["target_norm"],
    target_normalizer          = None,

    add_relative_time_idx = False,
    add_target_scales     = False,
    add_encoder_length    = False,
    allow_missing_timesteps = False,
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, df_val,
    min_prediction_idx = train_end_idx + 1,
    predict            = False,
    stop_randomization = True,
)
test_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, df_test,
    min_prediction_idx = val_end_idx + 1,
    predict            = False,
    stop_randomization = True,
)

# Boundary assertions — hard checks prevent data leakage
assert validation_dataset.min_prediction_idx >= train_end_idx + 1, \
    "CONTAMINATION: val predictions start in training period"
assert test_dataset.min_prediction_idx >= val_end_idx + 1, \
    "CONTAMINATION: test predictions start in validation period"

# Window count verification
expected_windows = sum(
    max(0, len(g) - FIXED["max_encoder_length"]
               - FIXED["max_prediction_length"] + 1)
    for _, g in df_train.groupby("SENSOR_ID")
)
assert len(training_dataset) == expected_windows, (
    f"Window count mismatch: {len(training_dataset):,} "
    f"vs expected {expected_windows:,}"
)

# DataLoaders
# batch_size=128 for training (gradient updates)
# batch_size=256 for eval passes (no gradients — double batch safe)
train_loader = training_dataset.to_dataloader(
    train=True,
    batch_size  = FIXED["batch_size"],        # 128
    num_workers = 4,
    pin_memory  = True
)
val_loader = validation_dataset.to_dataloader(
    train=False,
    batch_size  = FIXED["batch_size"] * 2,    # 256
    num_workers = 4,
    pin_memory  = True
)
test_loader = test_dataset.to_dataloader(
    train=False,
    batch_size  = FIXED["batch_size"] * 2,    # 256
    num_workers = 4,
    pin_memory  = True
)

print(f"  Created in {time.time()-t0:.1f}s")
print(f"  Training windows:  {len(training_dataset):,}  "
      f"(fewer than v6 due to longer encoder)")
print(f"  Validation windows:{len(validation_dataset):,}")
print(f"  Test windows:      {len(test_dataset):,}")
print(f"  Train batches:     {len(train_loader):,}  "
      f"(batch=128, doubled from v6)")
print(f"  Val batches:       {len(val_loader):,}  (batch=256)")
print(f"  ✓ Boundary verification passed")

# Sample verification
sample_x, sample_y = training_dataset[0]
enc_vals = sample_x["encoder_target"].numpy()
print(f"\nSample window:")
print(f"  encoder shape:    {enc_vals.shape}  (expected (72,))")
print(f"  prediction shape: {sample_y[0].shape}  (expected (6,))")
print(f"  z-score range:    [{enc_vals.min():.3f}, {enc_vals.max():.3f}]")
assert enc_vals.shape[0] == FIXED["max_encoder_length"], \
    f"Encoder length mismatch: {enc_vals.shape[0]} vs {FIXED['max_encoder_length']}"
assert sample_y[0].shape[0] == FIXED["max_prediction_length"], \
    f"Prediction length mismatch"
print(f"  ✓ Encoder=72h and prediction=6h confirmed")

# ── Metric functions ──────────────────────────────────────────────────────────
def compute_all_metrics(preds, actuals, naive_mae):
    preds   = preds.flatten().astype(float)
    actuals = actuals.flatten().astype(float)
    errors  = preds - actuals
    abs_e   = np.abs(errors)
    mae     = float(np.mean(abs_e))
    rmse    = float(np.sqrt(np.mean(errors**2)))
    mase    = float(mae/naive_mae) if naive_mae > 1e-8 else float("nan")
    denom   = np.abs(preds) + np.abs(actuals)
    smape   = float(np.mean(
        np.where(denom > 1e-8, 2.0*abs_e/denom, 0.0))*100)
    ss_res  = float(np.sum(errors**2))
    ss_tot  = float(np.sum((actuals - actuals.mean())**2))
    r2      = float(1.0-ss_res/ss_tot) if ss_tot > 1e-8 else float("nan")
    return {
        "MAE":   round(mae,   2),
        "RMSE":  round(rmse,  2),
        "MASE":  round(mase,  4),
        "SMAPE": round(smape, 2),
        "R2":    round(r2,    4),
    }

def inverse_transform(pred_norm, actual_norm, sensor_str_array):
    try:
        mu_arr    = np.array([
            SENSOR_STATS.loc[int(s),"mu"] for s in sensor_str_array
        ], dtype=float)
        sigma_arr = np.array([
            SENSOR_STATS.loc[int(s),"sigma"] for s in sensor_str_array
        ], dtype=float)
    except KeyError as e:
        raise KeyError(f"Sensor {e} not in SENSOR_STATS")
    pred_veh   = pred_norm   * sigma_arr[:,None] + mu_arr[:,None]
    actual_veh = actual_norm * sigma_arr[:,None] + mu_arr[:,None]
    return np.clip(pred_veh, 0, None).astype(float), actual_veh.astype(float)

# ── Naive baselines ───────────────────────────────────────────────────────────
SENSOR_FULL = {
    str(sid): (
        df[df["SENSOR_ID"]==sid]
        .sort_values("time_idx").set_index("time_idx")["target"]
    )
    for sid in SENSOR_IDS
}

def compute_exact_naive_mae(timestamps, actuals_veh, sensors):
    errors  = []
    missing = []
    for i in range(len(timestamps)):
        t_idx = int(timestamps[i])
        lag_t = t_idx - 168
        s     = SENSOR_FULL.get(sensors[i])
        if s is None or lag_t not in s.index:
            missing.append((sensors[i], t_idx, lag_t))
            continue
        errors.append(abs(float(actuals_veh[i]) - float(s.loc[lag_t])))
    assert len(missing) == 0, f"Missing lags: {missing[:3]}"
    assert len(errors)  == len(timestamps)
    return float(np.mean(errors))

print("\nComputing naive baselines...")
naive_errors = []
for sid in SENSOR_IDS:
    fs = SENSOR_FULL[str(sid)]
    for t, actual in fs[fs.index > val_end_idx].items():
        lag_t = t - 168
        if lag_t in fs.index:
            naive_errors.append(abs(actual - float(fs.loc[lag_t])))
naive_mae = float(np.mean(naive_errors))

val_naive_errors = []
for sid in SENSOR_IDS:
    fs = SENSOR_FULL[str(sid)]
    for t, actual in fs[
        (fs.index > train_end_idx) & (fs.index <= val_end_idx)
    ].items():
        lag_t = t - 168
        if lag_t in fs.index:
            val_naive_errors.append(abs(actual - float(fs.loc[lag_t])))
val_naive_mae = float(np.mean(val_naive_errors))

print(f"  Test naive MAE: {naive_mae:.2f} veh/h")
print(f"  Val  naive MAE: {val_naive_mae:.2f} veh/h")
print(f"  (identical to v6 — same sensors, same periods)")

# ── Training infrastructure ───────────────────────────────────────────────────
MEAN_SIGMA       = float(SENSOR_STATS["sigma"].mean())
SENSOR_STR_ORDER = df_train["sensor_str"].unique().tolist()

# Train eval loader uses batch=256 (double training batch)
# Same as val_loader — ensures identical computation
train_eval_loader = training_dataset.to_dataloader(
    train       = False,
    batch_size  = FIXED["batch_size"] * 2,  # 256
    num_workers = 4,
    pin_memory  = True,
)
print(f"\nTrain eval loader: {len(train_eval_loader):,} batches  (batch=256)")
print(f"Val loader:        {len(val_loader):,} batches  (batch=256)")
print(f"Both use eval() + no_grad + final model state — identical")

# ── Build model ───────────────────────────────────────────────────────────────
# NBeats.from_dataset reads max_encoder_length=72 from training_dataset
# and sets FC input dimension to 72 automatically
print(f"\nBuilding N-BEATS v7 model...")
model = NBeats.from_dataset(
    training_dataset,
    stack_types                   = FIXED["stack_types"],
    num_blocks                    = exp["num_blocks"],
    num_block_layers              = FIXED["num_block_layers"],
    widths                        = exp["widths"],
    sharing                       = FIXED["sharing"],
    expansion_coefficient_lengths = FIXED["expansion_coefficient_lengths"],
    dropout                       = FIXED["dropout"],
    loss                          = MAE(),
    log_interval                  = 200,
    log_val_interval              = 1,
    learning_rate                 = exp["lr"],
    optimizer                     = "adam",
    reduce_on_plateau_patience    = FIXED["reduce_on_plateau_patience"],
    # optimizer_kwargs not supported in PyTorch Forecasting v1.8.0 NBeats
    # Weight decay applied via configure_optimizers patch below
)

# ── Weight decay patch ────────────────────────────────────────────────────────
# Identical approach to v6 — types.MethodType override of configure_optimizers
# Injects weight_decay=1e-4 into Adam and rebuilds ReduceLROnPlateau
# Reference: Joseph (2022) Ch.5 — weight decay as L2 regularisation
# Reference: Krogh & Hertz (1992) — weight decay for generalisation

def _configure_optimizers_with_wd(self):
    optimizer = torch.optim.Adam(
        self.parameters(),
        lr           = exp["lr"],
        weight_decay = FIXED["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience = FIXED["reduce_on_plateau_patience"],
        factor   = 0.5,
        mode     = "min",
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor":   "val_loss",
            "interval":  "epoch",
            "frequency": 1,
        },
    }

model.configure_optimizers = types.MethodType(
    _configure_optimizers_with_wd, model
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters:   {n_params:,}  (was 51,649,536 in v6)")
print(f"  Increase:     +{n_params - 51_649_536:,}  "
      f"(72-48=24 extra input dims × FC weights)")
print(f"  Optimizer:    Adam(lr={exp['lr']}, "
      f"weight_decay={FIXED['weight_decay']})")
print(f"  Scheduler:    ReduceLROnPlateau("
      f"patience={FIXED['reduce_on_plateau_patience']}, factor=0.5)")

# ── Frozen training evaluation callback ───────────────────────────────────────
# Identical to v6 — complete pass over all training data using
# final frozen model state θ_final, eval() mode, no gradients
# Produces train_loss_frozen comparable with val_loss

class FrozenTrainEval(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        pl_module.eval()
        total_loss    = 0.0
        total_windows = 0

        with torch.no_grad():
            for x_b, y_b in train_eval_loader:
                x_b = {
                    k: v.to(pl_module.device)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in x_b.items()
                }
                out        = pl_module(x_b)
                pred       = out.prediction.squeeze(-1)
                actual     = y_b[0].to(pl_module.device)
                batch_size = pred.shape[0]
                batch_loss = float(
                    torch.sum(torch.mean(torch.abs(pred - actual), dim=1))
                )
                total_loss    += batch_loss
                total_windows += batch_size

        pl_module.train()

        frozen_train_loss = total_loss / total_windows

        pl_module.log(
            "train_loss_frozen", frozen_train_loss,
            on_epoch=True, prog_bar=False,
        )

        val_l      = float(trainer.callback_metrics.get(
            "val_loss", float("nan")))
        epoch      = trainer.current_epoch
        current_lr = trainer.optimizers[0].param_groups[0]["lr"]
        ratio      = (val_l / frozen_train_loss
                      if frozen_train_loss > 1e-8 else float("nan"))

        if not np.isnan(ratio):
            if ratio < 0.95:
                status = "val<train"
            elif ratio < 1.05:
                status = "R≈1.0"
            elif ratio < 1.50:
                status = "val>train"
            else:
                status = "⚠ large gap"
        else:
            status = ""

        print(
            f"  Ep {epoch:3d} | "
            f"train_frozen={frozen_train_loss:.4f} | "
            f"val={val_l:.4f} "
            f"(~{val_l*MEAN_SIGMA:.0f} veh/h) | "
            f"R={ratio:.3f} | {status} | "
            f"lr={current_lr:.2e}"
        )

        if torch.cuda.is_available():
            mem_alloc    = torch.cuda.memory_allocated()/1e9
            mem_reserved = torch.cuda.memory_reserved()/1e9
            mem_total    = torch.cuda.get_device_properties(0).total_memory/1e9
            mem_free     = mem_total - mem_reserved
            print(
                f"         GPU: {mem_alloc:.2f}GB alloc | "
                f"{mem_reserved:.2f}GB reserved | "
                f"{mem_free:.2f}GB free / {mem_total:.2f}GB total"
            )

# ── Callbacks and trainer ─────────────────────────────────────────────────────
exp_dir = os.path.join(EXP_BASE_DIR, exp["name"])
os.makedirs(exp_dir, exist_ok=True)

checkpoint_cb = ModelCheckpoint(
    dirpath    = exp_dir,
    filename   = "best",
    monitor    = "val_loss",
    mode       = "min",
    save_top_k = 1,
    verbose    = False,
)
earlystop_cb = EarlyStopping(
    monitor   = "val_loss",
    patience  = FIXED["patience"],
    mode      = "min",
    min_delta = FIXED["min_delta"],
    verbose   = True,
)
csv_logger = CSVLogger(exp_dir, name="logs", version=0)

trainer = pl.Trainer(
    max_epochs          = FIXED["max_epochs"],   # 150
    accelerator         = "gpu",
    devices             = 1,
    gradient_clip_val   = 1.0,
    callbacks           = [
        checkpoint_cb,
        earlystop_cb,
        LearningRateMonitor("epoch"),
        FrozenTrainEval(),
    ],
    logger              = csv_logger,
    enable_progress_bar = False,
    log_every_n_steps   = 50,
    deterministic       = False,
)

# ── GPU memory check ──────────────────────────────────────────────────────────
print(f"\nGPU memory before training:")
print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  Free:      "
      f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/1e9:.2f} GB")

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"TRAINING: {exp['name']}")
print(f"  encoder={FIXED['max_encoder_length']}h  "
      f"prediction={FIXED['max_prediction_length']}h")
print(f"  blocks={exp['num_blocks']}  widths={exp['widths']}")
print(f"  lr={exp['lr']}  weight_decay={FIXED['weight_decay']}")
print(f"  scheduler=ReduceLROnPlateau"
      f"(patience={FIXED['reduce_on_plateau_patience']})")
print(f"  batch={FIXED['batch_size']}  max_epochs={FIXED['max_epochs']}")
print(f"  train_loss: frozen eval — identical to val loss")
print(f"  Model: purely univariate — target_norm only")
print(f"{'='*65}")

t0 = time.time()
trainer.fit(
    model,
    train_dataloaders = train_loader,
    val_dataloaders   = val_loader,
)
train_minutes = (time.time() - t0) / 60
best_val_loss = float(checkpoint_cb.best_model_score)
epochs_run    = trainer.current_epoch + 1

print(f"\nTraining complete:")
print(f"  Time:         {train_minutes:.1f} min")
print(f"  Epochs:       {epochs_run}")
print(f"  Best val MAE: {best_val_loss:.4f} z-score units")

# ── Collect validation predictions ────────────────────────────────────────────
print("\nCollecting validation predictions...")
best_model = NBeats.load_from_checkpoint(checkpoint_cb.best_model_path)
best_model.eval().to(device)

val_preds_list      = []
val_actuals_list    = []
val_sensors_list    = []
val_timestamps_list = []

with torch.no_grad():
    for x_b, y_b in val_loader:
        x_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in x_b.items()
        }
        out = best_model(x_gpu)
        val_preds_list.append(out.prediction.squeeze(-1).cpu())
        val_actuals_list.append(y_b[0].cpu())
        val_sensors_list.append(np.array([
            SENSOR_STR_ORDER[int(g)]
            for g in x_b["groups"].numpy().flatten()
        ]))
        val_timestamps_list.append(x_b["decoder_time_idx"].numpy())

val_preds_norm    = torch.cat(val_preds_list,   dim=0).numpy()
val_actuals_norm  = torch.cat(val_actuals_list, dim=0).numpy()
val_sensor_arr    = np.concatenate(val_sensors_list,    axis=0)
val_timestamp_arr = np.concatenate(val_timestamps_list, axis=0)

assert val_preds_norm.shape == val_actuals_norm.shape
assert val_preds_norm.shape[1] == FIXED["max_prediction_length"]
print(f"  Shape: {val_preds_norm.shape}  (z-score, 6 horizons)")

val_preds_veh, val_actuals_veh = inverse_transform(
    val_preds_norm, val_actuals_norm, val_sensor_arr
)
print(f"  Range: [{val_preds_veh.min():.1f}, {val_preds_veh.max():.1f}] veh/h")

# ── Per-horizon validation metrics ────────────────────────────────────────────
print("\nComputing per-horizon exact-population naive MAE...")
horizon_metrics    = {}
horizon_naive_maes = {}

for h in range(FIXED["max_prediction_length"]):
    h_label = f"t+{h+1}h"
    h_ts    = val_timestamp_arr[:, h]
    h_act   = val_actuals_veh[:, h]
    h_naive = compute_exact_naive_mae(h_ts, h_act, val_sensor_arr)
    horizon_naive_maes[h_label] = h_naive
    horizon_metrics[h_label]    = compute_all_metrics(
        val_preds_veh[:, h], h_act, h_naive
    )

overall_val = {
    metric: round(float(np.mean([
        horizon_metrics[f"t+{h+1}h"][metric]
        for h in range(FIXED["max_prediction_length"])
    ])), 4 if metric in ["MASE","R2"] else 2)
    for metric in ["MAE","RMSE","MASE","SMAPE","R2"]
}

print(f"\n{'='*65}")
print(f"VALIDATION RESULTS — {exp['name']}")
print(f"  encoder={FIXED['max_encoder_length']}h | "
      f"prediction={FIXED['max_prediction_length']}h | "
      f"batch={FIXED['batch_size']} | wd=λ={FIXED['weight_decay']}")
print(f"{'='*65}")
print(f"{'Horizon':<10} {'MAE':>8} {'RMSE':>8} "
      f"{'MASE':>7} {'SMAPE':>7} {'R²':>7} {'NaiveMAE':>10}")
print(f"{'-'*62}")
for h_label, m in horizon_metrics.items():
    status = "✓" if m["MASE"] < 1.0 else "✗"
    print(f"{h_label:<10} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
          f"{m['MASE']:>7.4f} {m['SMAPE']:>7.2f} "
          f"{m['R2']:>7.4f} "
          f"{horizon_naive_maes[h_label]:>10.2f} {status}")
print(f"{'Overall':<10} {overall_val['MAE']:>8.2f} "
      f"{overall_val['RMSE']:>8.2f} "
      f"{overall_val['MASE']:>7.4f} "
      f"{overall_val['SMAPE']:>7.2f} "
      f"{overall_val['R2']:>7.4f}")
print(f"(Overall = arithmetic mean of 6 per-horizon values)")
print(f"{'='*65}")

# ── Comparison with previous versions ────────────────────────────────────────
print(f"\n  Version comparison (validation):")
print(f"  {'Config':<35} {'best val':>10} {'MAE':>8} {'MASE':>8}")
print(f"  {'-'*63}")
print(f"  {'v4: enc=48h fixed lr':<35} {'0.1923':>10} {'91.69':>8} {'0.8676':>8}")
print(f"  {'v5: enc=48h scheduler':<35} {'0.1884':>10} {'89.74':>8} {'0.8492':>8}")
print(f"  {'v6: enc=48h sched+wd':<35} {'0.1656':>10} {'77.58':>8} {'0.7342':>8}")
print(f"  {'v7: enc=72h sched+wd':<35} "
      f"{best_val_loss:>10.4f} "
      f"{overall_val['MAE']:>8.2f} "
      f"{overall_val['MASE']:>8.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "experiment":    exp["name"],
    "version":       "v7 — encoder=72h",
    "description":   exp["description"],
    "normalisation": "per-sensor z-score, target_normalizer=None",
    "loss_function": "MAE on z-score targets",
    "train_loss_method": (
        "frozen eval on ALL training data — identical to val loss. "
    ),
    "scheduler":     (
        f"ReduceLROnPlateau patience={FIXED['reduce_on_plateau_patience']} "
        f"factor=0.5"
    ),
    "weight_decay":  FIXED["weight_decay"],
    "hyperparameters": {
        "num_blocks":                 exp["num_blocks"],
        "widths":                     exp["widths"],
        "initial_lr":                 exp["lr"],
        "reduce_on_plateau_patience": FIXED["reduce_on_plateau_patience"],
        "weight_decay":               FIXED["weight_decay"],
        "max_epochs":                 FIXED["max_epochs"],
        "max_encoder_length":         FIXED["max_encoder_length"],
        "max_prediction_length":      FIXED["max_prediction_length"],
        "batch_size":                 FIXED["batch_size"],
        "n_params":                   n_params,
        "univariate":                 True,
    },
    "training": {
        "epochs_run":    epochs_run,
        "best_val_loss": best_val_loss,
        "training_min":  round(train_minutes, 1),
        "early_stopped": epochs_run < FIXED["max_epochs"],
        "checkpoint":    checkpoint_cb.best_model_path,
    },
    "val_naive_mae":        val_naive_mae,
    "test_naive_mae":       naive_mae,
    "horizon_naive_maes":   horizon_naive_maes,
    "val_horizon_metrics":  horizon_metrics,
    "val_overall_metrics":  overall_val,
}
out_json = os.path.join(exp_dir, "results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
SENSOR_STATS.to_csv(os.path.join(exp_dir, "sensor_stats.csv"))

print(f"\nResults saved: {out_json}")
print(f"Checkpoint:    {checkpoint_cb.best_model_path}")

del model, best_model
torch.cuda.empty_cache()
print(f"\nGPU memory after training:")
print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print("\nDone.")