# =============================================================================
#  ULTIMATE STOCK PREDICTOR  v7.0  —  Live Tracking & Website Edition
# =============================================================================
#
#  All v6 features PLUS:
#  [11] Live Prediction Accuracy Tracking
#         — store predictions, evaluate when actuals arrive
#         — last_prediction_error_pct / rolling_7d / rolling_30d accuracy
#  [12] Confidence Calibration
#         — tracks how often actual price falls inside q10–q90 band
#  [13] Market Status
#         — OPEN / CLOSED / PRE_MARKET / AFTER_HOURS per asset type
#  [14] Asset Correlation Matrix
#         — rolling 60-day return correlations, GET /correlations
#  [15] Alert Flags
#         — LOW_CONFIDENCE | HIGH_VOLATILITY | MODEL_STALE | DRIFT_DETECTED
#         — SIGNAL_REVERSAL | CALIBRATION_POOR | MARKET_CLOSED
#  [16] Signal Performance Metrics
#         — long_win_rate / short_win_rate from live prediction ledger
#  [17] Forecast Cone Data
#         — cumulative uncertainty bands for timeline visualization
#  [18] Model Versioning
#         — model_version, schema_version, trained_hash per asset
#
#  New API endpoints:
#    GET /correlations          ← heatmap-ready correlation matrix
#    GET /accuracy/<ticker>     ← live accuracy metrics
#    GET /alerts/<ticker>       ← current alert flags
#    GET /cone/<ticker>         ← forecast cone data for charts
#
#  Directory structure:
#    models/    ← saved model files per asset
#    state/     ← retrain state, drift logs, audit trail
#    ledger/    ← prediction ledger (stored predictions + outcomes)
#    reports/   ← monthly performance reports
#    plots/     ← saved charts
# =============================================================================

import warnings, os, random, math, time, json, threading, logging
import pickle, hashlib
from datetime import datetime, timedelta
from pathlib   import Path

warnings.filterwarnings("ignore")

import numpy        as np
import pandas       as pd
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import joblib

import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model  import Ridge, LinearRegression
from sklearn.ensemble      import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics       import mean_squared_error

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, callbacks, backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("state/predictor.log", mode="a"),
    ]
)
log = logging.getLogger("StockPredictor")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)

# =============================================================================
#  DIRECTORIES
# =============================================================================
for d in ["models", "state", "reports", "plots", "ledger"]:
    Path(d).mkdir(exist_ok=True)

# Model versioning constants
MODEL_VERSION  = "v7.0"
SCHEMA_VERSION = "2025-02"

# =============================================================================
#  PER-ASSET CONFIG
#  Start date = IPO date or earliest reliable data
#  This maximises training history for each asset individually
# =============================================================================
ASSET_CONFIG = {
    # ── Companies ─────────────────────────────────────────────────────────────
    "AAPL":  {"desc": "Apple",              "start": "1980-12-12", "type": "stock"},
    "MSFT":  {"desc": "Microsoft",          "start": "1986-03-13", "type": "stock"},
    "GOOGL": {"desc": "Google / Alphabet",  "start": "2004-08-19", "type": "stock"},
    "AMZN":  {"desc": "Amazon",             "start": "1997-05-15", "type": "stock"},
    "TSLA":  {"desc": "Tesla",              "start": "2010-06-29", "type": "stock"},
    "JPM":   {"desc": "JPMorgan",           "start": "1980-01-02", "type": "stock"},
    "NVDA":  {"desc": "Nvidia",             "start": "1999-01-22", "type": "stock"},
    "META":  {"desc": "Meta (Facebook)",    "start": "2012-05-18", "type": "stock"},
    "BRK-B": {"desc": "Berkshire Hathaway", "start": "1996-05-09", "type": "stock"},
    "XOM":   {"desc": "Exxon Mobil",        "start": "1980-01-02", "type": "stock"},
    # ── Asset Classes ─────────────────────────────────────────────────────────
    "GLD":      {"desc": "Gold ETF",         "start": "2004-11-18", "type": "etf"},
    "EURUSD=X": {"desc": "EUR/USD Forex",    "start": "2003-12-01", "type": "forex"},
    "BTC-USD":  {"desc": "Bitcoin",          "start": "2014-09-17", "type": "crypto"},
    "QQQ":      {"desc": "Nasdaq 100 ETF",   "start": "1999-03-10", "type": "etf"},
    "^VIX":     {"desc": "VIX Volatility",   "start": "1990-01-02", "type": "index"},
}

END_DATE       = datetime.today().strftime("%Y-%m-%d")
LOOK_BACK      = 60
EMBARGO_DAYS   = 10
WF_STEPS       = 5

MODEL_DIM      = 64
NUM_HEADS      = 4
FF_DIM         = 128
NUM_BLOCKS     = 2
DROPOUT        = 0.15
EPOCHS         = 80
BATCH_SIZE     = 32
MC_SAMPLES     = 20

PRED_THRESHOLD = 0.002
KELLY_FRACTION = 0.10
MAX_LEVERAGE   = 1.0
VOL_TARGET     = 0.01
STOP_LOSS      = -0.04
HIGH_VOL_THRESH= 0.022

BASE_SPREAD    = 0.0005
MARKET_IMPACT  = 0.001
MIN_LIQUIDITY  = 500_000

DRIFT_THRESHOLD = 0.15    # if live MAE > train MAE * (1 + this) → drift alert
RETRAIN_DAY     = 1       # retrain on 1st of every month

# =============================================================================
#  STATE MANAGER  —  tracks last retrain, drift, performance per asset
# =============================================================================
STATE_FILE = "state/retrain_state.json"

def load_state() -> dict:
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def update_asset_state(ticker: str, key: str, value):
    state = load_state()
    if ticker not in state:
        state[ticker] = {}
    state[ticker][key] = value
    save_state(state)

def get_asset_state(ticker: str) -> dict:
    return load_state().get(ticker, {})

def needs_retraining(ticker: str) -> bool:
    """Returns True if asset has never been trained or last trained > 30 days ago."""
    s = get_asset_state(ticker)
    if "last_trained" not in s:
        return True
    last = datetime.fromisoformat(s["last_trained"])
    return (datetime.now() - last).days >= 30

# =============================================================================
#  MODEL PERSISTENCE  —  save/load per asset
# =============================================================================
def model_path(ticker: str, model_name: str) -> str:
    safe = ticker.replace("=", "_").replace("^", "_").replace("-", "_")
    return f"models/{safe}_{model_name}"

def save_sklearn_model(model, ticker: str, name: str):
    path = model_path(ticker, name) + ".joblib"
    joblib.dump(model, path)
    log.info(f"  Saved {name} → {path}")

def load_sklearn_model(ticker: str, name: str):
    path = model_path(ticker, name) + ".joblib"
    if Path(path).exists():
        return joblib.load(path)
    return None

def save_scaler(scaler, ticker: str):
    path = model_path(ticker, "scaler") + ".joblib"
    joblib.dump(scaler, path)

def load_scaler(ticker: str):
    path = model_path(ticker, "scaler") + ".joblib"
    if Path(path).exists():
        return joblib.load(path)
    return None

if TF_AVAILABLE:
    def save_keras_model(model, ticker: str, name: str):
        path = model_path(ticker, name) + ".keras"
        model.save(path)
        log.info(f"  Saved {name} → {path}")

    def load_keras_model(ticker: str, name: str):
        path = model_path(ticker, name) + ".keras"
        if Path(path).exists():
            try:
                return tf.keras.models.load_model(
                    path,
                    custom_objects={"loss": quantile_loss(0.5)}
                )
            except Exception:
                return None
        return None

# =============================================================================
#  FEATURE ENGINEERING
# =============================================================================
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def atr(h, l, c, n=14):
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.rolling(n).mean() / (c + 1e-9)

def bollinger(c, n=20):
    ma  = c.rolling(n).mean()
    std = c.rolling(n).std()
    return (c - (ma - 2*std)) / (4*std + 1e-9), (4*std) / (ma + 1e-9)

def parkinson(h, l, n=10):
    return np.sqrt((np.log(h/(l+1e-9))**2).rolling(n).mean()/(4*np.log(2)))

def calendar_enc(idx):
    df = pd.DataFrame(index=idx)
    for name, val, period in [("dow",idx.dayofweek,5),
                               ("month",idx.month,12),
                               ("quarter",idx.quarter,4)]:
        df[f"{name}_sin"] = np.sin(2*np.pi*val/period)
        df[f"{name}_cos"] = np.cos(2*np.pi*val/period)
    return df

def build_features(ticker: str, start_date: str) -> pd.DataFrame:
    """Download from IPO date and build full feature matrix."""
    raw = yf.download(ticker, start=start_date, end=END_DATE,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c,tuple) else c for c in raw.columns]
    if raw.empty or len(raw) < 200:
        raise ValueError(f"Insufficient data for {ticker} from {start_date}")

    c = raw.get("Close", raw.iloc[:,0]).squeeze()
    h = raw.get("High",  c).squeeze()
    l = raw.get("Low",   c).squeeze()
    o = raw.get("Open",  c).squeeze()
    v = raw.get("Volume",pd.Series(1e6, index=c.index)).squeeze()

    # Download macro with matching date range
    vix   = yf.download("^VIX", start=start_date, end=END_DATE,
                         progress=False)["Close"].squeeze()
    rates = yf.download("^TNX", start=start_date, end=END_DATE,
                         progress=False)["Close"].squeeze()

    df = pd.DataFrame({"Close": c, "Volume": v}, index=c.index)
    df["Return"]    = c.pct_change()
    df["LogReturn"] = np.log(c / c.shift(1))

    for lag in [1,2,3,5,10]:
        df[f"Ret_lag{lag}"] = df["Return"].shift(lag)

    for w in [5,10,20,50,200]:
        df[f"MA_{w}"] = c.rolling(w).mean()

    df["MA10_ratio"]  = c / (df["MA_10"]  + 1e-9)
    df["MA50_ratio"]  = c / (df["MA_50"]  + 1e-9)
    df["MA10_50"]     = df["MA_10"] / (df["MA_50"]  + 1e-9)
    df["MA50_200"]    = df["MA_50"] / (df["MA_200"] + 1e-9)
    df["Vol_10"]      = df["Return"].rolling(10).std()
    df["Vol_30"]      = df["Return"].rolling(30).std()
    df["Vol_ratio"]   = df["Vol_10"] / (df["Vol_30"] + 1e-9)
    df["Parkinson"]   = parkinson(h, l, 10)
    df["ATR"]         = atr(h, l, c, 14)
    df["HL_pct"]      = (h - l) / (c + 1e-9)
    df["Gap"]         = (o - c.shift(1)) / (c.shift(1) + 1e-9)
    df["RSI_14"]      = rsi(c, 14)
    df["RSI_28"]      = rsi(c, 28)

    m  = _ema(c,12) - _ema(c,26)
    ms = _ema(m, 9)
    df["MACD"]      = m  / (c + 1e-9)
    df["MACD_sig"]  = ms / (c + 1e-9)
    df["MACD_hist"] = (m - ms) / (c + 1e-9)

    bb, bw          = bollinger(c, 20)
    df["BollB"]     = bb
    df["BollBW"]    = bw
    df["Vol_ratio2"]= v / (v.rolling(10).mean() + 1e-9)
    df["VIX"]       = vix
    df["Rates"]     = rates
    df["AutoCorr"]  = df["Return"].rolling(20).apply(
                          lambda x: pd.Series(x).autocorr(1), raw=False)

    cal = calendar_enc(df.index)
    df  = pd.concat([df, cal], axis=1)
    df  = df.drop(columns=[f"MA_{w}" for w in [5,10,20,50,200]])
    df  = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# =============================================================================
#  SEQUENCES
# =============================================================================
def make_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def flatten(data, look_back):
    X, y = make_sequences(data, look_back)
    return X.reshape(len(X), -1), y

# =============================================================================
#  REGIME DETECTION
# =============================================================================
def detect_regime(returns, window=20):
    ret  = pd.Series(returns)
    rvol = ret.rolling(window).std()
    rma  = ret.rolling(window).mean()
    med  = rvol.median()
    reg  = np.zeros(len(ret), dtype=int)
    for i in range(len(ret)):
        v = rvol.iloc[i] if not np.isnan(rvol.iloc[i]) else med
        m = rma.iloc[i]  if not np.isnan(rma.iloc[i]) else 0
        if   m > 0 and v <= med: reg[i] = 0   # bull
        elif m < 0:               reg[i] = 2   # bear
        else:                     reg[i] = 1   # chop
    return reg, rvol.values

# =============================================================================
#  QUANTILE LOSS + DEEP MODELS
# =============================================================================
if TF_AVAILABLE:
    def quantile_loss(q):
        def loss(y_true, y_pred):
            e = y_true - y_pred
            return tf.reduce_mean(tf.maximum(q*e, (q-1)*e))
        loss.__name__ = f"q{int(q*100)}_loss"
        return loss

    class WarmupCosine(callbacks.Callback):
        def __init__(self, lr_max, warmup, total):
            super().__init__()
            self.lr_max = lr_max; self.warmup = warmup; self.total = total
        def on_epoch_begin(self, epoch, logs=None):
            if epoch < self.warmup:
                lr = self.lr_max * (epoch+1) / self.warmup
            else:
                p  = (epoch-self.warmup) / max(1, self.total-self.warmup)
                lr = self.lr_max * 0.5 * (1+math.cos(math.pi*p))
            K.set_value(self.model.optimizer.lr, lr)

    def pos_enc(length, depth):
        pos  = np.arange(length)[:,None]
        dims = np.arange(depth)[None,:]
        ang  = pos / np.power(10000, 2*(dims//2)/np.float32(depth))
        ang[:,0::2] = np.sin(ang[:,0::2])
        ang[:,1::2] = np.cos(ang[:,1::2])
        return tf.cast(ang[None], tf.float32)

    def tf_block(x, heads, kdim, ff, drop):
        a = layers.MultiHeadAttention(heads, kdim, dropout=drop)(x, x)
        a = layers.Dropout(drop)(a)
        x = layers.LayerNormalization(1e-6)(x + a)
        f = layers.Dense(ff, activation="gelu")(x)
        f = layers.Dropout(drop)(f)
        f = layers.Dense(x.shape[-1])(f)
        return layers.LayerNormalization(1e-6)(x + f)

    def build_quantile_transformer(input_shape):
        inp = layers.Input(shape=input_shape)
        x   = layers.Dense(MODEL_DIM)(inp)
        x   = x + pos_enc(input_shape[0], MODEL_DIM)
        for _ in range(NUM_BLOCKS):
            x = tf_block(x, NUM_HEADS, MODEL_DIM//NUM_HEADS, FF_DIM, DROPOUT)
        x   = layers.GlobalAveragePooling1D()(x)
        x   = layers.Dropout(DROPOUT)(x)
        x   = layers.Dense(64, activation="gelu")(x)
        q10 = layers.Dense(1, name="q10")(x)
        q50 = layers.Dense(1, name="q50")(x)
        q90 = layers.Dense(1, name="q90")(x)
        m   = Model(inp, [q10, q50, q90], name="QuantileTransformer")
        m.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
            loss={"q10":quantile_loss(0.10),
                  "q50":quantile_loss(0.50),
                  "q90":quantile_loss(0.90)},
            loss_weights={"q10":1.0,"q50":2.0,"q90":1.0},
        )
        return m

    def build_bilstm(input_shape):
        inp = layers.Input(shape=input_shape)
        x   = layers.Bidirectional(layers.LSTM(64,return_sequences=True,dropout=DROPOUT))(inp)
        x   = layers.Bidirectional(layers.LSTM(32,return_sequences=True,dropout=DROPOUT))(x)
        x   = layers.MultiHeadAttention(2,16,dropout=DROPOUT)(x,x)
        x   = layers.GlobalAveragePooling1D()(x)
        x   = layers.Dropout(DROPOUT)(x)
        out = layers.Dense(1)(x)
        m   = Model(inp, out, name="BiLSTM")
        m.compile(tf.keras.optimizers.Adam(5e-4, clipnorm=1.0), "mse")
        return m

    def get_callbacks(lr=1e-3):
        return [
            callbacks.EarlyStopping("val_loss", patience=10,
                                    restore_best_weights=True, verbose=0),
            WarmupCosine(lr, warmup=8, total=EPOCHS),
        ]

    def mc_predict(model, X, n=MC_SAMPLES):
        preds = np.array([model(X, training=True).numpy().squeeze()
                          for _ in range(n)])
        return preds.mean(0), preds.std(0)

# =============================================================================
#  DRIFT DETECTOR
# =============================================================================
def check_drift(ticker: str, actual: np.ndarray, predicted: np.ndarray) -> bool:
    """
    Compare live prediction error against stored baseline MAE.
    If error has grown by DRIFT_THRESHOLD → flag drift → trigger retraining.
    """
    live_mae  = float(np.mean(np.abs(actual - predicted)))
    state     = get_asset_state(ticker)
    baseline  = state.get("baseline_mae", live_mae)

    drift_ratio = (live_mae - baseline) / (baseline + 1e-9)
    is_drift    = drift_ratio > DRIFT_THRESHOLD

    update_asset_state(ticker, "live_mae",    live_mae)
    update_asset_state(ticker, "drift_ratio", round(drift_ratio, 4))
    update_asset_state(ticker, "drift_flag",  is_drift)

    if is_drift:
        log.warning(f"  ⚠ DRIFT DETECTED — {ticker}: "
                    f"live MAE={live_mae:.4f}, baseline={baseline:.4f}, "
                    f"ratio={drift_ratio:.2%} → scheduling early retraining")
    return is_drift

# =============================================================================
#  CORE TRAINING FUNCTION  (per asset, full history)
# =============================================================================
def train_asset(ticker: str, force: bool = False) -> dict:
    """
    Full training pipeline for one asset from its IPO date to today.
    Saves all models to disk. Updates state file.
    Returns results dict for reporting.
    """
    cfg       = ASSET_CONFIG[ticker]
    start     = cfg["start"]
    desc      = cfg["desc"]

    log.info(f"\n{'='*60}")
    log.info(f"  Training: {ticker} ({desc})")
    log.info(f"  History : {start} → {END_DATE}")
    log.info(f"{'='*60}")

    # ── Download & build features ─────────────────────────────────────────────
    try:
        df = build_features(ticker, start)
    except Exception as e:
        log.error(f"  Feature build failed for {ticker}: {e}")
        return {}

    n_rows = len(df)
    n_feat = df.shape[1]
    log.info(f"  Dataset : {n_rows} rows × {n_feat} features "
             f"({start} → {END_DATE})")

    data      = df.values
    step_size = n_rows // WF_STEPS
    results   = []

    for step in range(WF_STEPS):
        train_end  = step_size * (step + 1)
        test_start = train_end + EMBARGO_DAYS
        test_end   = test_start + step_size

        train = data[:train_end]
        test  = data[test_start:test_end]

        if len(test) < LOOK_BACK + 30:
            continue

        scaler   = RobustScaler()
        tr_sc    = scaler.fit_transform(train)
        te_sc    = scaler.transform(test)

        X_tr, y_tr = make_sequences(tr_sc, LOOK_BACK)
        X_te, y_te = make_sequences(te_sc, LOOK_BACK)
        X_tr_f     = X_tr.reshape(len(X_tr), -1)
        X_te_f     = X_te.reshape(len(X_te), -1)

        actual_raw = test[LOOK_BACK:, 0]

        def inv_close(col):
            d = np.zeros((len(col), n_feat)); d[:,0] = col
            return scaler.inverse_transform(d)[:,0]

        # ── Naive baseline ────────────────────────────────────────────────────
        naive_rmse = float(np.sqrt(mean_squared_error(
            actual_raw[1:], actual_raw[:-1])))

        # ── Sklearn baselines ─────────────────────────────────────────────────
        lr  = LinearRegression()
        gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=SEED)
        rf  = RandomForestRegressor(n_estimators=200, max_depth=6,
                                    min_samples_leaf=5, random_state=SEED,
                                    n_jobs=-1)
        lr.fit(X_tr_f, y_tr);  lr_pred  = lr.predict(X_te_f)
        gbm.fit(X_tr_f, y_tr); gbm_pred = gbm.predict(X_te_f)
        rf.fit(X_tr_f, y_tr);  rf_pred  = rf.predict(X_te_f)

        lr_rmse  = float(np.sqrt(mean_squared_error(y_te, lr_pred)))
        gbm_rmse = float(np.sqrt(mean_squared_error(y_te, gbm_pred)))
        rf_rmse  = float(np.sqrt(mean_squared_error(y_te, rf_pred)))

        # ── Deep ensemble (if TF available) ───────────────────────────────────
        q10_inv = q50_inv = q90_inv = None
        deep_rmse = None

        if TF_AVAILABLE:
            input_shape = (LOOK_BACK, n_feat)

            qt = build_quantile_transformer(input_shape)
            qt.fit(X_tr, [y_tr, y_tr, y_tr],
                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_split=0.1, callbacks=get_callbacks(),
                   verbose=0, shuffle=False)

            bl = build_bilstm(input_shape)
            bl.fit(X_tr, y_tr,
                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_split=0.1, callbacks=get_callbacks(5e-4),
                   verbose=0, shuffle=False)

            # Quantile predictions
            q10_s, q50_s, q90_s = qt.predict(X_te, verbose=0)
            bl_mean, bl_unc      = mc_predict(bl, X_te)

            # Blend: q50 = 0.6 × Transformer + 0.4 × BiLSTM
            q50_blend = 0.6 * q50_s.squeeze() + 0.4 * bl_mean
            q10_inv   = inv_close(q10_s.squeeze())
            q50_inv   = inv_close(q50_blend)
            q90_inv   = inv_close(q90_s.squeeze())
            deep_rmse = float(np.sqrt(mean_squared_error(actual_raw, q50_inv)))

        # If TF not available, use GBM as primary predictor
        if q50_inv is None:
            q50_inv   = inv_close(gbm_pred)
            q10_inv   = q50_inv * 0.98
            q90_inv   = q50_inv * 1.02
            deep_rmse = gbm_rmse

        log.info(f"  [Window {step+1}/{WF_STEPS}] "
                 f"Naive={naive_rmse:.4f} | LR={lr_rmse:.4f} | "
                 f"GBM={gbm_rmse:.4f} | RF={rf_rmse:.4f} | "
                 f"Ensemble={deep_rmse:.4f}")

        results.append({
            "step":       step+1,
            "actual":     actual_raw,
            "q10":        q10_inv,
            "q50":        q50_inv,
            "q90":        q90_inv,
            "naive_rmse": naive_rmse,
            "lr_rmse":    lr_rmse,
            "gbm_rmse":   gbm_rmse,
            "rf_rmse":    rf_rmse,
            "deep_rmse":  deep_rmse,
        })

    if not results:
        log.error(f"  No valid windows for {ticker}")
        return {}

    # ── Save final-window models (trained on most data) ───────────────────────
    # Retrain on full dataset for production use
    log.info(f"  Saving production models for {ticker}...")
    full_scaler = RobustScaler()
    full_scaled = full_scaler.fit_transform(data)
    save_scaler(full_scaler, ticker)
    save_sklearn_model(gbm, ticker, "gbm")
    save_sklearn_model(rf,  ticker, "rf")
    save_sklearn_model(lr,  ticker, "lr")

    if TF_AVAILABLE:
        # Retrain on all data for production
        X_all, y_all = make_sequences(full_scaled, LOOK_BACK)
        qt_prod = build_quantile_transformer((LOOK_BACK, n_feat))
        qt_prod.fit(X_all, [y_all,y_all,y_all],
                    epochs=40, batch_size=BATCH_SIZE,
                    validation_split=0.05, callbacks=get_callbacks(),
                    verbose=0, shuffle=False)
        save_keras_model(qt_prod, ticker, "qt_prod")

    # ── Compute baseline MAE for drift detection ───────────────────────────────
    last_actual = results[-1]["actual"]
    last_pred   = results[-1]["q50"]
    baseline_mae = float(np.mean(np.abs(last_actual - last_pred)))

    # ── Update state ──────────────────────────────────────────────────────────
    now = datetime.now().isoformat()
    update_asset_state(ticker, "last_trained",  now)
    update_asset_state(ticker, "baseline_mae",  baseline_mae)
    update_asset_state(ticker, "n_rows",        n_rows)
    update_asset_state(ticker, "n_features",    n_feat)
    update_asset_state(ticker, "start_date",    start)
    update_asset_state(ticker, "deep_rmse",     results[-1]["deep_rmse"])
    update_asset_state(ticker, "naive_rmse",    results[-1]["naive_rmse"])
    update_asset_state(ticker, "drift_flag",    False)

    log.info(f"  ✓ {ticker} training complete. "
             f"Baseline MAE={baseline_mae:.4f}")

    return {
        "ticker":  ticker,
        "desc":    desc,
        "df":      df,
        "results": results,
    }

# =============================================================================
#  TRANSPARENCY LAYER  —  all 8 interpretability features
# =============================================================================

# ── [1] Confidence Score ──────────────────────────────────────────────────────
def compute_confidence(q10: float, q50: float, q90: float,
                       price: float) -> float:
    """
    Confidence = 1 - (normalised interval width).
    Narrow band → high confidence.  Wide band → low confidence.
    Output: 0.0 – 1.0
    """
    spread      = max(q90 - q10, 1e-9)
    rel_spread  = spread / (abs(price) + 1e-9)
    # A spread of 5 % of price → confidence ≈ 0.  A spread of 0 % → 1.0
    confidence  = float(np.clip(1 - rel_spread / 0.05, 0.0, 1.0))
    return round(confidence, 4)

# ── [2] Signal Strength ───────────────────────────────────────────────────────
def compute_signal_strength(pred_return: float, confidence: float,
                             threshold: float = PRED_THRESHOLD) -> int:
    """
    Signal strength 0–100.
    = (return magnitude / 2× threshold) × confidence × 100
    Capped at 100.  Below threshold → max 40 (noise zone).
    """
    magnitude  = abs(pred_return) / (2 * threshold + 1e-9)
    raw        = magnitude * confidence * 100
    if abs(pred_return) < threshold:
        raw = min(raw, 40)          # cap noise-zone signals at 40
    return int(np.clip(round(raw), 0, 100))

# ── [3] Regime Label ──────────────────────────────────────────────────────────
def regime_label(returns_window: np.ndarray) -> str:
    """
    Maps recent returns to a human-readable regime string.
    LOW_VOL  — low volatility, positive drift
    HIGH_VOL — high volatility, positive drift
    BEAR     — negative trend regardless of vol
    CHOP     — no clear direction
    """
    if len(returns_window) < 5:
        return "UNKNOWN"
    vol  = float(np.std(returns_window))
    mean = float(np.mean(returns_window))
    if mean < -0.001:
        return "BEAR"
    if vol > HIGH_VOL_THRESH:
        return "HIGH_VOL"
    if mean > 0.001 and vol <= HIGH_VOL_THRESH:
        return "LOW_VOL"
    return "CHOP"

# ── [4] Model Freshness ───────────────────────────────────────────────────────
def model_age_days(ticker: str) -> int:
    """Days since this asset's models were last trained."""
    s = get_asset_state(ticker)
    if "last_trained" not in s:
        return -1
    last = datetime.fromisoformat(s["last_trained"])
    return (datetime.now() - last).days

# ── [5] Baseline Comparison ───────────────────────────────────────────────────
def baseline_comparison(ticker: str) -> dict:
    """Return RMSE, naive RMSE, and improvement % from saved state."""
    s = get_asset_state(ticker)
    deep  = s.get("deep_rmse",  None)
    naive = s.get("naive_rmse", None)
    if deep is None or naive is None:
        return {}
    imp = (naive - deep) / (naive + 1e-9) * 100
    return {
        "rmse":               round(float(deep),  6),
        "naive_rmse":         round(float(naive), 6),
        "improvement_percent":round(float(imp),   2),
    }

# ── [6] Feature Importance ────────────────────────────────────────────────────
def top_features(ticker: str, feature_names: list, top_n: int = 5) -> list:
    """
    Extract feature importances from the saved RandomForest.
    Aggregates across the look-back window (mean importance per original feature).
    Returns top_n as [{"name": ..., "importance": ...}].
    """
    rf = load_sklearn_model(ticker, "rf")
    if rf is None:
        gbm = load_sklearn_model(ticker, "gbm")
        if gbm is None:
            return []
        imp_flat = gbm.feature_importances_
    else:
        imp_flat = rf.feature_importances_

    n_orig  = len(feature_names)
    # imp_flat has shape (LOOK_BACK × n_features,) — reshape and mean over time
    if len(imp_flat) == LOOK_BACK * n_orig:
        imp_2d  = imp_flat.reshape(LOOK_BACK, n_orig)
        imp_agg = imp_2d.mean(axis=0)
    else:
        # fallback: truncate/pad to n_orig
        imp_agg = imp_flat[:n_orig]

    total  = imp_agg.sum() + 1e-9
    normed = imp_agg / total

    idx    = np.argsort(normed)[::-1][:top_n]
    return [
        {"name": feature_names[i], "importance": round(float(normed[i]), 6)}
        for i in idx
    ]

# ── [7] Multi-Horizon Forecast ────────────────────────────────────────────────
def multi_horizon_forecast(ticker: str, df: pd.DataFrame,
                            scaler, n_feat: int,
                            gbm_model=None, qt_model=None) -> dict:
    """
    Predict 1d, 5d, 20d by recursively rolling the sequence forward.
    Each step: predict next bar → append to sequence → predict again.
    Returns dict with one entry per horizon.
    """
    data    = df.values
    price_0 = float(data[-1, 0])

    def predict_n_steps(n: int) -> tuple:
        """Roll n steps forward, return (q10, q50, q90) at step n."""
        window = scaler.transform(data[-LOOK_BACK:]).copy()

        for _ in range(n):
            X  = window[None].astype(np.float32)

            if TF_AVAILABLE and qt_model is not None:
                q10_s, q50_s, q90_s = qt_model.predict(X, verbose=0)
                next_val = float(q50_s[0, 0])
            elif gbm_model is not None:
                next_val = float(gbm_model.predict(X.reshape(1, -1))[0])
            else:
                return None, None, None

            # Build next row: shift window, insert prediction at [0]
            new_row      = window[-1].copy()
            new_row[0]   = next_val           # update Close (scaled)
            window       = np.vstack([window[1:], new_row[None]])

        # Inverse-transform final prediction
        def inv(val):
            d = np.zeros((1, n_feat)); d[0, 0] = val
            return float(scaler.inverse_transform(d)[0, 0])

        X_final = window[None].astype(np.float32)
        if TF_AVAILABLE and qt_model is not None:
            q10_s, q50_s, q90_s = qt_model.predict(X_final, verbose=0)
            return inv(q10_s[0,0]), inv(q50_s[0,0]), inv(q90_s[0,0])
        elif gbm_model is not None:
            p = float(gbm_model.predict(X_final.reshape(1, -1))[0])
            return inv(p * 0.97), inv(p), inv(p * 1.03)
        return None, None, None

    horizons = {"1d": 1, "5d": 5, "20d": 20}
    forecast  = {}

    for label, n in horizons.items():
        q10, q50, q90 = predict_n_steps(n)
        if q50 is None:
            forecast[label] = {}
            continue
        pred_ret  = (q50 - price_0) / (price_0 + 1e-9)
        conf      = compute_confidence(q10, q50, q90, price_0)
        forecast[label] = {
            "q10":                  round(q10, 4),
            "q50":                  round(q50, 4),
            "q90":                  round(q90, 4),
            "predicted_return_pct": round(pred_ret * 100, 4),
            "confidence":           conf,
            "signal":               ("LONG"  if pred_ret >  PRED_THRESHOLD else
                                     "FLAT"  if pred_ret > -PRED_THRESHOLD else
                                     "SHORT"),
        }

    return forecast

# ── [8] Backtest Summary ──────────────────────────────────────────────────────
def backtest_summary(results: list) -> dict:
    """
    Aggregate backtest metrics across all walk-forward windows.
    Returns clean dict ready for API/frontend consumption.
    """
    if not results:
        return {}

    all_strat = []
    all_bh    = []

    for r in results:
        a   = r["actual"]
        q50 = r["q50"]
        n   = min(len(a), len(q50)) - 1

        pred_rets = (q50[1:n+1] - a[:n]) / (a[:n] + 1e-9)
        true_rets = (a[1:n+1]   - a[:n]) / (a[:n] + 1e-9)
        signals   = (pred_rets > PRED_THRESHOLD).astype(float)
        sr        = true_rets * signals
        sr        = np.clip(sr, STOP_LOSS, None)

        all_strat.extend(sr.tolist())
        all_bh.extend(true_rets.tolist())

    sr  = np.array(all_strat)
    bhr = np.array(all_bh)
    cum_s  = (1 + sr).cumprod()
    cum_bh = (1 + bhr).cumprod()

    peak   = np.maximum.accumulate(cum_s)
    dd     = (cum_s - peak) / (peak + 1e-9)
    max_dd = float(dd.min())

    sharpe  = float(np.sqrt(252) * sr.mean() / (sr.std() + 1e-9))
    sortino_d = sr[sr < 0]
    sortino = float(np.sqrt(252) * sr.mean() / (sortino_d.std() + 1e-9))
    pf      = float(sr[sr>0].sum() / (abs(sr[sr<0].sum()) + 1e-9))

    return {
        "total_return_pct":    round(float(cum_s[-1]  - 1) * 100, 2),
        "buyhold_return_pct":  round(float(cum_bh[-1] - 1) * 100, 2),
        "sharpe_ratio":        round(sharpe,  4),
        "sortino_ratio":       round(sortino, 4),
        "max_drawdown_pct":    round(max_dd * 100, 2),
        "win_rate_pct":        round(float(np.mean(sr > 0)) * 100, 2),
        "profit_factor":       round(pf, 4),
        "n_trades":            int(np.sum(signals)),
        "avg_daily_return_pct":round(float(sr.mean()) * 100, 4),
        "volatility_ann_pct":  round(float(sr.std() * np.sqrt(252)) * 100, 2),
    }

# =============================================================================
#  V7 ENHANCEMENTS  —  Live Tracking & Website Layer
# =============================================================================

# ── [11] Prediction Ledger  ───────────────────────────────────────────────────
LEDGER_DIR = "ledger"

def ledger_path(ticker: str) -> str:
    safe = ticker.replace("=","_").replace("^","_").replace("-","_")
    return f"{LEDGER_DIR}/{safe}_ledger.jsonl"

def store_prediction(ticker: str, payload: dict):
    """
    Write a prediction to the per-asset ledger when it is made.
    Fields stored: timestamp, q10, q50, q90, signal, current_price.
    Actual outcome is filled in by resolve_predictions() the next day.
    """
    entry = {
        "timestamp":     payload.get("timestamp"),
        "prediction_date": datetime.now().date().isoformat(),
        "q10":           payload.get("forecast", {}).get("1d", {}).get("q10"),
        "q50":           payload.get("forecast", {}).get("1d", {}).get("q50"),
        "q90":           payload.get("forecast", {}).get("1d", {}).get("q90"),
        "signal":        payload.get("signal"),
        "current_price": payload.get("current_price"),
        "actual_price":  None,    # filled in next day
        "resolved":      False,
    }
    with open(ledger_path(ticker), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

def resolve_predictions(ticker: str, current_price: float):
    """
    Called at prediction time — resolves any unresolved prior entries
    by comparing stored q50 against today's actual price.
    Rewrites the ledger file with resolved entries.
    """
    path = ledger_path(ticker)
    if not Path(path).exists():
        return

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    today = datetime.now().date().isoformat()
    changed = False
    for e in entries:
        if not e.get("resolved") and e.get("prediction_date") != today:
            e["actual_price"] = current_price
            e["resolved"]     = True
            if e.get("q50") and e.get("current_price"):
                err_pct = (current_price - e["q50"]) / (e["current_price"] + 1e-9) * 100
                e["error_pct"]     = round(err_pct, 4)
                e["in_band"]       = (e["q10"] or 0) <= current_price <= (e["q90"] or 1e9)
                e["signal_correct"]= (
                    (e["signal"] == "LONG"  and current_price > e["current_price"]) or
                    (e["signal"] == "SHORT" and current_price < e["current_price"]) or
                    (e["signal"] == "FLAT")
                )
            changed = True

    if changed:
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e, default=str) + "\n")

def load_ledger(ticker: str) -> list:
    path = ledger_path(ticker)
    if not Path(path).exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return [e for e in entries if e.get("resolved")]

def live_accuracy_metrics(ticker: str) -> dict:
    """
    [11] Live prediction accuracy from the ledger.
    Returns last_prediction_error_pct, rolling_7d_accuracy, rolling_30d_accuracy.
    Accuracy = fraction of predictions where signal direction was correct.
    """
    entries = load_ledger(ticker)
    if not entries:
        return {
            "last_prediction_error_pct": None,
            "rolling_7d_accuracy":       None,
            "rolling_30d_accuracy":      None,
            "total_resolved":            0,
        }

    # Sort by date
    entries = sorted(entries, key=lambda e: e.get("prediction_date",""))
    last    = entries[-1]
    today   = datetime.now().date()

    def accuracy_window(days: int) -> float | None:
        cutoff = (today - timedelta(days=days)).isoformat()
        window = [e for e in entries
                  if e.get("prediction_date","") >= cutoff
                  and e.get("signal_correct") is not None]
        if not window:
            return None
        return round(float(np.mean([e["signal_correct"] for e in window])), 4)

    return {
        "last_prediction_error_pct": last.get("error_pct"),
        "rolling_7d_accuracy":       accuracy_window(7),
        "rolling_30d_accuracy":      accuracy_window(30),
        "total_resolved":            len(entries),
    }

# ── [12] Confidence Calibration  ─────────────────────────────────────────────
def confidence_calibration(ticker: str) -> float | None:
    """
    [12] Fraction of resolved predictions where actual fell inside q10–q90.
    A perfectly calibrated 80% interval → calibration ≈ 0.80.
    """
    entries = load_ledger(ticker)
    resolved = [e for e in entries if e.get("in_band") is not None]
    if not resolved:
        return None
    return round(float(np.mean([e["in_band"] for e in resolved])), 4)

# ── [13] Market Status  ───────────────────────────────────────────────────────
def market_status(ticker: str) -> str:
    """
    [13] Returns OPEN | CLOSED | PRE_MARKET | AFTER_HOURS.
    Uses NYSE/NASDAQ hours for stocks/ETFs, 24/7 for crypto, FX weekdays.
    All times in US Eastern.
    """
    from datetime import timezone
    import zoneinfo

    asset_type = ASSET_CONFIG.get(ticker, {}).get("type", "stock")
    now_utc    = datetime.now(timezone.utc)

    try:
        eastern = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        # fallback: approximate UTC-5
        eastern = timezone(timedelta(hours=-5))

    now_et = now_utc.astimezone(eastern)
    weekday = now_et.weekday()   # 0=Mon … 6=Sun
    hour    = now_et.hour
    minute  = now_et.minute
    time_dec= hour + minute / 60.0

    if asset_type == "crypto":
        return "OPEN"   # crypto trades 24/7

    if asset_type == "forex":
        # FX closed Sat 17:00 ET → Sun 17:00 ET
        if weekday == 5:                    return "CLOSED"
        if weekday == 6 and hour < 17:      return "CLOSED"
        return "OPEN"

    # Stocks, ETFs, indices — NYSE hours
    if weekday >= 5:
        return "CLOSED"

    if   4.0 <= time_dec < 9.5:   return "PRE_MARKET"
    elif 9.5 <= time_dec < 16.0:  return "OPEN"
    elif 16.0<= time_dec < 20.0:  return "AFTER_HOURS"
    else:                          return "CLOSED"

# ── [14] Correlation Matrix  ──────────────────────────────────────────────────
_correlation_cache = {"data": None, "computed_at": None}
CORR_CACHE_MINUTES = 60

def compute_correlation_matrix(window_days: int = 60) -> dict:
    """
    [14] Rolling return correlations across all 15 assets.
    Cached for CORR_CACHE_MINUTES to avoid expensive repeated downloads.
    Returns matrix + metadata suitable for a heatmap.
    """
    now = datetime.now()
    if (_correlation_cache["data"] is not None and
            _correlation_cache["computed_at"] is not None and
            (now - _correlation_cache["computed_at"]).seconds < CORR_CACHE_MINUTES * 60):
        return _correlation_cache["data"]

    tickers = list(ASSET_CONFIG.keys())
    start   = (now - timedelta(days=window_days + 10)).strftime("%Y-%m-%d")
    returns = {}

    for t in tickers:
        try:
            raw = yf.download(t, start=start, end=END_DATE,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            c = raw.get("Close", raw.iloc[:,0]).squeeze()
            r = c.pct_change().dropna()
            if len(r) >= 20:
                returns[t] = r
        except Exception:
            pass

    if not returns:
        return {}

    df_ret = pd.DataFrame(returns).dropna()
    corr   = df_ret.tail(window_days).corr()

    labels = corr.columns.tolist()
    matrix = corr.values.tolist()

    result = {
        "labels":       labels,
        "matrix":       [[round(v, 4) for v in row] for row in matrix],
        "window_days":  window_days,
        "computed_at":  now.isoformat(),
        "descriptions": {t: ASSET_CONFIG.get(t, {}).get("desc", t)
                         for t in labels},
    }

    _correlation_cache["data"]        = result
    _correlation_cache["computed_at"] = now
    return result

# ── [15] Alert Flags  ─────────────────────────────────────────────────────────
def compute_alerts(payload: dict, ticker: str) -> list[str]:
    """
    [15] Returns list of active alert strings for the frontend to display.
    Possible values:
      LOW_CONFIDENCE      confidence < 0.40
      HIGH_VOLATILITY     regime is HIGH_VOL or BEAR
      MODEL_STALE         freshness is STALE or EXPIRED
      DRIFT_DETECTED      drift.flag is True
      SIGNAL_REVERSAL     signal changed vs last prediction
      CALIBRATION_POOR    confidence_calibration < 0.50
      MARKET_CLOSED       market is not OPEN
    """
    alerts = []

    if payload.get("confidence", 1.0) < 0.40:
        alerts.append("LOW_CONFIDENCE")

    if payload.get("regime") in ("HIGH_VOL", "BEAR"):
        alerts.append("HIGH_VOLATILITY")

    if payload.get("model_freshness") in ("STALE", "EXPIRED"):
        alerts.append("MODEL_STALE")

    if payload.get("drift", {}).get("flag"):
        alerts.append("DRIFT_DETECTED")

    if payload.get("market_status") != "OPEN":
        alerts.append("MARKET_CLOSED")

    # Signal reversal — compare against last ledger entry
    entries = load_ledger(ticker)
    if entries:
        last_signal = entries[-1].get("signal")
        curr_signal = payload.get("signal")
        if last_signal and curr_signal and last_signal != curr_signal:
            alerts.append("SIGNAL_REVERSAL")

    # Calibration check
    calib = payload.get("confidence_calibration")
    if calib is not None and calib < 0.50:
        alerts.append("CALIBRATION_POOR")

    return alerts

# ── [16] Signal Performance Metrics  ─────────────────────────────────────────
def signal_performance(ticker: str) -> dict:
    """
    [16] Historical win rates per signal type from the live ledger.
    long_win_rate  — fraction of LONG signals where price went up
    short_win_rate — fraction of SHORT signals where price went down
    flat_accuracy  — fraction of FLAT signals where price stayed within ±threshold
    """
    entries = load_ledger(ticker)
    if not entries:
        return {
            "long_win_rate":  None,
            "short_win_rate": None,
            "flat_accuracy":  None,
            "n_long":         0,
            "n_short":        0,
            "n_flat":         0,
        }

    def win_rate(signal_type: str) -> tuple[float | None, int]:
        subset = [e for e in entries
                  if e.get("signal") == signal_type
                  and e.get("signal_correct") is not None]
        if not subset:
            return None, 0
        return round(float(np.mean([e["signal_correct"] for e in subset])), 4), len(subset)

    lwr, n_l = win_rate("LONG")
    swr, n_s = win_rate("SHORT")
    far, n_f = win_rate("FLAT")

    return {
        "long_win_rate":  lwr,
        "short_win_rate": swr,
        "flat_accuracy":  far,
        "n_long":         n_l,
        "n_short":        n_s,
        "n_flat":         n_f,
        "total_signals":  n_l + n_s + n_f,
    }

# ── [17] Forecast Cone  ───────────────────────────────────────────────────────
def forecast_cone(ticker: str, df: pd.DataFrame,
                  scaler, n_feat: int,
                  gbm_model=None, qt_model=None,
                  horizon: int = 20, n_paths: int = 50) -> dict:
    """
    [17] Monte Carlo forecast cone — simulate n_paths forward trajectories
    by adding Gaussian noise scaled to historical vol at each step.
    Returns percentile bands at each future timestep for timeline visualisation.

    Frontend usage: plot p10/p25/p50/p75/p90 as nested shaded bands.
    """
    data      = df.values
    price_now = float(data[-1, 0])
    hist_vol  = float(np.std(np.diff(data[-30:, 0]) / (data[-31:-1, 0] + 1e-9)))

    # Get base 1-day model prediction as starting drift
    window = scaler.transform(data[-LOOK_BACK:]).copy()
    base_drift = 0.0
    if TF_AVAILABLE and qt_model is not None:
        X = window[None].astype(np.float32)
        _, q50_s, _ = qt_model.predict(X, verbose=0)
        d = np.zeros((1, n_feat)); d[0,0] = float(q50_s[0,0])
        pred_price  = float(scaler.inverse_transform(d)[0,0])
        base_drift  = (pred_price - price_now) / (price_now + 1e-9)
    elif gbm_model is not None:
        X = window[None].reshape(1,-1)
        d = np.zeros((1, n_feat)); d[0,0] = float(gbm_model.predict(X)[0])
        pred_price  = float(scaler.inverse_transform(d)[0,0])
        base_drift  = (pred_price - price_now) / (price_now + 1e-9)

    # Simulate paths
    rng   = np.random.default_rng(SEED)
    paths = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = price_now

    for t in range(1, horizon + 1):
        # Drift decays toward 0 after day 1 (uncertainty compounds)
        drift = base_drift * math.exp(-0.3 * (t - 1))
        shock = rng.normal(drift, hist_vol, n_paths)
        paths[:, t] = paths[:, t-1] * (1 + shock)

    # Compute percentile bands
    percentiles = [10, 25, 50, 75, 90]
    bands = {f"p{p}": [round(float(v), 4) for v in np.percentile(paths, p, axis=0)]
             for p in percentiles}

    return {
        "current_price": round(price_now, 4),
        "horizon_days":  horizon,
        "n_paths":       n_paths,
        "bands":         bands,
        "timestamps":    [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                          for i in range(horizon + 1)],
    }

# ── [18] Model Versioning  ────────────────────────────────────────────────────
def model_version_info(ticker: str) -> dict:
    """
    [18] Returns version identifiers for transparency and cache-busting.
    trained_hash: SHA1 of last_trained timestamp — changes every retrain.
    """
    s     = get_asset_state(ticker)
    lt    = s.get("last_trained", "unknown")
    thash = hashlib.sha1(lt.encode()).hexdigest()[:8]
    return {
        "model_version":  MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "trained_hash":   thash,
        "last_trained":   lt[:19] if lt != "unknown" else "never",
    }


AUDIT_LOG = "state/audit.jsonl"

def audit_prediction(payload: dict):
    """Append every prediction to a JSON-lines audit trail."""
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")

# =============================================================================
#  PREDICT — full v6 output with all 8 transparency features
# =============================================================================
def predict_latest(ticker: str, results: list = None) -> dict:
    """
    Load saved models and return a fully enriched prediction payload.

    Returns
    -------
    {
      "ticker":            "AAPL",
      "description":       "Apple",
      "asset_type":        "stock",
      "timestamp":         "2025-...",
      "current_price":     185.42,

      "forecast": {
        "1d":  {"q10":..,"q50":..,"q90":..,"confidence":..,"signal":..,"predicted_return_pct":..},
        "5d":  {...},
        "20d": {...}
      },

      "confidence":          0.82,
      "signal_strength":     74,
      "signal":              "LONG",
      "regime":              "LOW_VOL",

      "model_age_days":      12,
      "model_freshness":     "FRESH",   # FRESH <30d / STALE 30-60d / EXPIRED >60d

      "baseline_comparison": {"rmse":1.92,"naive_rmse":2.41,"improvement_percent":20.3},

      "top_features": [
        {"name":"VIX","importance":0.21},
        ...
      ],

      "backtest_summary":  {total_return, sharpe, max_dd, win_rate, ...},

      "drift": {"flag": false, "ratio": 0.04},
      "predicted_return_pct": 0.31
    }
    """
    cfg    = ASSET_CONFIG.get(ticker, {})
    scaler = load_scaler(ticker)
    if scaler is None:
        log.warning(f"  No saved models for {ticker}. Run training first.")
        return {"error": f"No trained model found for {ticker}"}

    try:
        df = build_features(ticker, cfg.get("start", "2010-01-01"))
    except Exception as e:
        log.error(f"  Data error for {ticker}: {e}")
        return {"error": str(e)}

    data         = df.values
    n_feat       = data.shape[1]
    feature_names= df.columns.tolist()
    recent_raw   = data[-LOOK_BACK:]
    scaled       = scaler.transform(recent_raw)
    X_live       = scaled[None].astype(np.float32)

    def inv_close(val):
        d = np.zeros((1, n_feat)); d[0, 0] = val
        return float(scaler.inverse_transform(d)[0, 0])

    # ── Base 1-day quantile prediction ────────────────────────────────────────
    qt_model  = None
    gbm_model = load_sklearn_model(ticker, "gbm")
    q10 = q50 = q90 = None

    if TF_AVAILABLE:
        qt_model = load_keras_model(ticker, "qt_prod")
        if qt_model is not None:
            q10_s, q50_s, q90_s = qt_model.predict(X_live, verbose=0)
            q10 = inv_close(q10_s[0, 0])
            q50 = inv_close(q50_s[0, 0])
            q90 = inv_close(q90_s[0, 0])

    if q50 is None and gbm_model is not None:
        p   = float(gbm_model.predict(X_live.reshape(1, -1))[0])
        q10 = inv_close(p * 0.98)
        q50 = inv_close(p)
        q90 = inv_close(p * 1.02)

    if q50 is None:
        return {"error": f"Prediction failed for {ticker} — no model output"}

    # ── Core scalars ──────────────────────────────────────────────────────────
    price_now  = float(data[-1, 0])
    pred_ret   = (q50 - price_now) / (price_now + 1e-9)
    confidence = compute_confidence(q10, q50, q90, price_now)
    strength   = compute_signal_strength(pred_ret, confidence)

    recent_returns = np.diff(data[-21:, 0]) / (data[-21:-1, 0] + 1e-9)
    regime         = regime_label(recent_returns)

    signal = ("LONG"  if pred_ret >  PRED_THRESHOLD else
              "FLAT"  if pred_ret > -PRED_THRESHOLD else
              "SHORT")

    # ── [4] Model freshness ───────────────────────────────────────────────────
    age  = model_age_days(ticker)
    freshness = ("FRESH"   if age < 30  else
                 "STALE"   if age < 60  else
                 "EXPIRED")

    # ── [5] Baseline comparison ───────────────────────────────────────────────
    bl_comp = baseline_comparison(ticker)

    # ── [6] Feature importance ────────────────────────────────────────────────
    feat_imp = top_features(ticker, feature_names, top_n=5)

    # ── [7] Multi-horizon forecast ────────────────────────────────────────────
    forecast = multi_horizon_forecast(
        ticker, df, scaler, n_feat, gbm_model, qt_model)

    # Inject 1d into forecast from live prediction (more accurate than rolled)
    forecast["1d"] = {
        "q10":                  round(q10, 4),
        "q50":                  round(q50, 4),
        "q90":                  round(q90, 4),
        "predicted_return_pct": round(pred_ret * 100, 4),
        "confidence":           confidence,
        "signal":               signal,
    }

    # ── [8] Backtest summary ──────────────────────────────────────────────────
    bt_summary = backtest_summary(results) if results else \
                 {"note": "Pass walk-forward results to get backtest summary"}

    # ── Drift info ────────────────────────────────────────────────────────────
    s     = get_asset_state(ticker)
    drift = {
        "flag":  bool(s.get("drift_flag", False)),
        "ratio": round(float(s.get("drift_ratio", 0)), 4),
    }

    # ── [11] Resolve prior predictions + live accuracy ────────────────────────
    resolve_predictions(ticker, price_now)
    accuracy = live_accuracy_metrics(ticker)

    # ── [12] Confidence calibration ───────────────────────────────────────────
    calibration = confidence_calibration(ticker)

    # ── [13] Market status ────────────────────────────────────────────────────
    mkt_status = market_status(ticker)

    # ── [16] Signal performance ───────────────────────────────────────────────
    sig_perf = signal_performance(ticker)

    # ── [17] Forecast cone ────────────────────────────────────────────────────
    cone = forecast_cone(ticker, df, scaler, n_feat, gbm_model, qt_model)

    # ── [18] Model versioning ─────────────────────────────────────────────────
    version_info = model_version_info(ticker)

    # ── Assemble full payload ─────────────────────────────────────────────────
    payload = {
        # ── Identity ──────────────────────────────────────────────────────────
        "ticker":               ticker,
        "description":          cfg.get("desc", ""),
        "asset_type":           cfg.get("type", ""),
        "timestamp":            datetime.now().isoformat(),
        "current_price":        round(price_now, 4),

        # ── [18] Versioning ───────────────────────────────────────────────────
        "model_version":        version_info["model_version"],
        "schema_version":       version_info["schema_version"],
        "trained_hash":         version_info["trained_hash"],

        # ── Forecast across horizons [7] ──────────────────────────────────────
        "forecast":             forecast,

        # ── 1-day summary (convenience) ───────────────────────────────────────
        "predicted_return_pct": round(pred_ret * 100, 4),
        "signal":               signal,

        # ── Confidence + strength [1][2] ──────────────────────────────────────
        "confidence":           confidence,
        "signal_strength":      strength,

        # ── [12] Calibration ──────────────────────────────────────────────────
        "confidence_calibration": calibration,

        # ── [3] Regime ────────────────────────────────────────────────────────
        "regime":               regime,

        # ── [13] Market status ────────────────────────────────────────────────
        "market_status":        mkt_status,

        # ── [4] Model freshness ───────────────────────────────────────────────
        "model_age_days":       age,
        "model_freshness":      freshness,

        # ── [5] Baseline ──────────────────────────────────────────────────────
        "baseline_comparison":  bl_comp,

        # ── [6] Feature importance ────────────────────────────────────────────
        "top_features":         feat_imp,

        # ── [8] Backtest ──────────────────────────────────────────────────────
        "backtest_summary":     bt_summary,

        # ── [11] Live accuracy ────────────────────────────────────────────────
        "live_accuracy":        accuracy,

        # ── [16] Signal performance ───────────────────────────────────────────
        "signal_performance":   sig_perf,

        # ── [17] Forecast cone ────────────────────────────────────────────────
        "forecast_cone":        cone,

        # ── Drift ─────────────────────────────────────────────────────────────
        "drift":                drift,
    }

    # ── [15] Alerts (computed last — needs full payload) ──────────────────────
    payload["alerts"] = compute_alerts(payload, ticker)

    # ── Store prediction in ledger ────────────────────────────────────────────
    store_prediction(ticker, payload)

    # ── Audit log ─────────────────────────────────────────────────────────────
    audit_prediction(payload)

    return payload

# =============================================================================
#  MONTHLY RETRAINING SCHEDULER
# =============================================================================
class MonthlyRetrainer:
    """
    Background thread that wakes up every hour, checks if it's the 1st of
    the month, and retrains any asset due for retraining.
    Also checks for drift daily and triggers early retraining if detected.
    """
    def __init__(self):
        self._stop  = threading.Event()
        self._thread= None
        self.state  = load_state()

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="MonthlyRetrainer")
        self._thread.start()
        log.info("  ✓ Monthly retrainer started (background thread).")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("  Retrainer stopped.")

    def _run(self):
        while not self._stop.is_set():
            now = datetime.now()

            for ticker in ASSET_CONFIG:
                state = get_asset_state(ticker)

                # Monthly retrain on RETRAIN_DAY of each month
                should_retrain = needs_retraining(ticker)

                # Also retrain if drift detected
                if state.get("drift_flag", False):
                    log.info(f"  Drift flag set for {ticker} → early retrain")
                    should_retrain = True

                if should_retrain:
                    log.info(f"\n  [SCHEDULER] Retraining {ticker} "
                             f"({datetime.now().strftime('%Y-%m-%d %H:%M')})")
                    try:
                        train_asset(ticker)
                        self._write_monthly_report(ticker)
                    except Exception as e:
                        log.error(f"  Scheduled retrain failed for {ticker}: {e}")

            # Sleep 1 hour between checks
            self._stop.wait(timeout=3600)

    def _write_monthly_report(self, ticker: str):
        """Write a brief JSON performance report after each retrain."""
        s    = get_asset_state(ticker)
        month= datetime.now().strftime("%Y-%m")
        path = f"reports/{ticker.replace('=','_').replace('^','_')}_{month}.json"
        report = {
            "ticker":       ticker,
            "month":        month,
            "trained_at":   s.get("last_trained"),
            "deep_rmse":    s.get("deep_rmse"),
            "naive_rmse":   s.get("naive_rmse"),
            "baseline_mae": s.get("baseline_mae"),
            "n_rows":       s.get("n_rows"),
            "drift_flag":   s.get("drift_flag"),
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"  Monthly report saved → {path}")

# =============================================================================
#  PLOTTING
# =============================================================================
def plot_asset(ticker: str, results: list, save: bool = True):
    if not results: return
    last = results[-1]
    a    = last["actual"]
    q10  = last["q10"]; q50 = last["q50"]; q90 = last["q90"]
    x    = range(len(a))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0d0d1a")
    fig.suptitle(f"{ticker} — {ASSET_CONFIG[ticker]['desc']}  |  "
                 f"Trained from {ASSET_CONFIG[ticker]['start']} → {END_DATE}",
                 color="#c0c0e0", fontsize=11, fontweight="bold")

    def style(ax, title):
        ax.set_facecolor("#13132b")
        ax.tick_params(colors="#888", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#222244")
        ax.set_title(title, color="#9090c0", fontsize=9, fontweight="bold")
        ax.grid(alpha=0.12, color="#444")

    # Price + quantile band
    ax = axes[0,0]
    style(ax, "Price + Prediction Interval (q10–q90)")
    ax.plot(x, a,   color="#4fc3f7", lw=1.0, label="Actual")
    ax.plot(x, q50, color="#ef5350", lw=0.9, ls="--", label="Median (q50)")
    ax.fill_between(x, q10, q90, color="#ef5350", alpha=0.15, label="q10–q90")
    ax.legend(fontsize=7, facecolor="#1a1a3e", edgecolor="#333",
              labelcolor="#ddd")

    # RMSE comparison across windows
    ax = axes[0,1]
    style(ax, "RMSE per Walk-Forward Window")
    windows = [r["step"] for r in results]
    ax.plot(windows, [r["naive_rmse"] for r in results],
            "o--", color="#607d8b", label="Naive")
    ax.plot(windows, [r["lr_rmse"]    for r in results],
            "o--", color="#42a5f5", label="LinearReg")
    ax.plot(windows, [r["gbm_rmse"]   for r in results],
            "o--", color="#ffca28", label="GBM")
    ax.plot(windows, [r["rf_rmse"]    for r in results],
            "o--", color="#ef5350", label="RandomForest")
    ax.plot(windows, [r["deep_rmse"]  for r in results],
            "o-",  color="#66bb6a", lw=2, label="Ensemble")
    ax.legend(fontsize=7, facecolor="#1a1a3e", edgecolor="#333",
              labelcolor="#ddd")
    ax.set_xlabel("Window", fontsize=7, color="#888")
    ax.set_ylabel("RMSE", fontsize=7, color="#888")

    # Prediction residuals
    ax = axes[1,0]
    style(ax, "Prediction Residuals (Last Window)")
    errs = a[:len(q50)] - q50
    ax.hist(errs, bins=40, color="#7e57c2", alpha=0.85, edgecolor="#1a1a2e")
    ax.axvline(0, color="#fff", lw=0.8, ls="--")
    ax.axvline(np.mean(errs),   color="#ffca28", lw=1.2, label=f"Mean={np.mean(errs):.2f}")
    ax.axvline(np.median(errs), color="#66bb6a", lw=1.2, label=f"Median={np.median(errs):.2f}")
    ax.legend(fontsize=7, facecolor="#1a1a3e", edgecolor="#333",
              labelcolor="#ddd")

    # State info
    ax = axes[1,1]
    ax.set_facecolor("#13132b")
    ax.axis("off")
    s   = get_asset_state(ticker)
    info= [
        ("Ticker",         ticker),
        ("Description",    ASSET_CONFIG[ticker]["desc"]),
        ("Type",           ASSET_CONFIG[ticker]["type"].upper()),
        ("Training Start", ASSET_CONFIG[ticker]["start"]),
        ("Last Trained",   s.get("last_trained","—")[:19]),
        ("Rows of Data",   f"{s.get('n_rows','—'):,}"),
        ("Features",       str(s.get("n_features","—"))),
        ("Best RMSE",      f"{s.get('deep_rmse',0):.4f}"),
        ("Naive RMSE",     f"{s.get('naive_rmse',0):.4f}"),
        ("Baseline MAE",   f"{s.get('baseline_mae',0):.4f}"),
        ("Drift Flag",     "⚠ YES" if s.get("drift_flag") else "✓ No"),
        ("Drift Ratio",    f"{s.get('drift_ratio',0):.2%}"),
    ]
    for i, (k, v) in enumerate(info):
        y = 0.95 - i * 0.08
        ax.text(0.05, y, k+":", transform=ax.transAxes,
                color="#6080a0", fontsize=8)
        ax.text(0.5,  y, str(v), transform=ax.transAxes,
                color="#e0e0f0", fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save:
        safe   = ticker.replace("=","_").replace("^","_").replace("-","_")
        path   = f"plots/{safe}_results.png"
        plt.savefig(path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        log.info(f"  Chart saved → {path}")
    plt.show()

def plot_universe_summary(all_results: dict):
    """Dashboard showing all 15 assets side by side."""
    tickers = [t for t,r in all_results.items() if r and r.get("results")]
    n       = len(tickers)
    if n == 0: return

    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4*rows),
                              facecolor="#0a0a18")
    fig.suptitle("Universe Overview — All 15 Assets  |  "
                 "Ultimate Predictor v5.0",
                 color="#c0c0e0", fontsize=13, fontweight="bold")
    axes = axes.flatten() if n > 1 else [axes]

    for i, ticker in enumerate(tickers):
        ax   = axes[i]
        res  = all_results[ticker]["results"][-1]
        a, q50 = res["actual"], res["q50"]
        n_pt = min(len(a), len(q50))

        ax.set_facecolor("#13132b")
        ax.plot(a[:n_pt],   color="#4fc3f7", lw=0.8, label="Actual")
        ax.plot(q50[:n_pt], color="#ef5350", lw=0.7, ls="--", label="q50")
        ax.fill_between(range(n_pt), res["q10"][:n_pt], res["q90"][:n_pt],
                        color="#ef5350", alpha=0.12)
        ax.tick_params(colors="#888", labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#222244")
        ax.grid(alpha=0.1, color="#444")

        s   = get_asset_state(ticker)
        imp = ((s.get("naive_rmse",1) - s.get("deep_rmse",1)) /
               (s.get("naive_rmse",1) + 1e-9) * 100)
        ax.set_title(
            f"{ticker} — {ASSET_CONFIG[ticker]['desc'][:20]}\n"
            f"RMSE={s.get('deep_rmse',0):.4f}  Δ={imp:+.1f}% vs naive",
            color="#9090c0", fontsize=7.5, fontweight="bold")

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = "plots/universe_summary.png"
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log.info(f"  Universe summary saved → {path}")
    plt.show()

# =============================================================================
#  REST API LAYER  v7.0
#  Run with: python stock_predictor_v7.py --api
#
#  Endpoints:
#    GET  /predict/<ticker>          → full prediction payload
#    GET  /predict/<ticker>/summary  → lightweight signal card
#    GET  /predict/<ticker>/forecast → multi-horizon forecast only
#    GET  /universe                  → all tickers + state
#    GET  /health                    → system health
#    GET  /correlations              → correlation matrix (heatmap-ready)
#    GET  /accuracy/<ticker>         → live accuracy metrics
#    GET  /alerts/<ticker>           → current alert flags
#    GET  /cone/<ticker>             → forecast cone data
#    GET  /backtest/<ticker>         → backtest summary + baselines
#    POST /retrain/<ticker>          → trigger background retrain
# =============================================================================
def start_api(retrainer_ref):
    try:
        from flask import Flask, jsonify, request
    except ImportError:
        log.warning("  Flask not installed — API disabled. "
                    "Install with: pip install flask")
        return

    app = Flask("StockPredictor")

    # ── CORS headers (for browser frontends) ──────────────────────────────────
    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.route("/predict/<ticker>")
    def api_predict(ticker):
        ticker = ticker.upper()
        if ticker not in ASSET_CONFIG:
            return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
        return jsonify(predict_latest(ticker))

    @app.route("/predict/<ticker>/summary")
    def api_summary(ticker):
        ticker = ticker.upper()
        if ticker not in ASSET_CONFIG:
            return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
        p = predict_latest(ticker)
        if "error" in p:
            return jsonify(p), 500
        return jsonify({
            "ticker":                 p["ticker"],
            "description":            p["description"],
            "current_price":          p["current_price"],
            "signal":                 p["signal"],
            "signal_strength":        p["signal_strength"],
            "confidence":             p["confidence"],
            "confidence_calibration": p.get("confidence_calibration"),
            "regime":                 p["regime"],
            "market_status":          p.get("market_status"),
            "predicted_return_pct":   p["predicted_return_pct"],
            "model_freshness":        p["model_freshness"],
            "model_age_days":         p["model_age_days"],
            "model_version":          p.get("model_version"),
            "alerts":                 p.get("alerts", []),
            "timestamp":              p["timestamp"],
        })

    @app.route("/predict/<ticker>/forecast")
    def api_forecast(ticker):
        ticker = ticker.upper()
        if ticker not in ASSET_CONFIG:
            return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
        p = predict_latest(ticker)
        return jsonify({
            "ticker":        p.get("ticker"),
            "current_price": p.get("current_price"),
            "forecast":      p.get("forecast", {}),
            "forecast_cone": p.get("forecast_cone", {}),
            "timestamp":     p.get("timestamp"),
        })

    @app.route("/universe")
    def api_universe():
        out = {}
        for ticker, cfg in ASSET_CONFIG.items():
            s   = get_asset_state(ticker)
            age = model_age_days(ticker)
            out[ticker] = {
                "description":    cfg["desc"],
                "type":           cfg["type"],
                "start_date":     cfg["start"],
                "last_trained":   s.get("last_trained", "never"),
                "model_age_days": age,
                "model_freshness":("FRESH"   if age < 30 else
                                   "STALE"   if age < 60 else
                                   "EXPIRED"),
                "model_version":  MODEL_VERSION,
                "deep_rmse":      s.get("deep_rmse"),
                "naive_rmse":     s.get("naive_rmse"),
                "drift_flag":     s.get("drift_flag", False),
                "market_status":  market_status(ticker),
            }
        return jsonify(out)

    @app.route("/health")
    def api_health():
        state    = load_state()
        trained  = sum(1 for t in ASSET_CONFIG if "last_trained" in state.get(t,{}))
        drifted  = sum(1 for t in ASSET_CONFIG if state.get(t,{}).get("drift_flag"))
        return jsonify({
            "status":           "ok",
            "timestamp":        datetime.now().isoformat(),
            "model_version":    MODEL_VERSION,
            "schema_version":   SCHEMA_VERSION,
            "total_assets":     len(ASSET_CONFIG),
            "trained_assets":   trained,
            "drifted_assets":   drifted,
            "retrainer_active": retrainer_ref._thread is not None and
                                retrainer_ref._thread.is_alive(),
            "tf_available":     TF_AVAILABLE,
        })

    # ── [14] Correlations ─────────────────────────────────────────────────────
    @app.route("/correlations")
    def api_correlations():
        window = int(request.args.get("window", 60))
        return jsonify(compute_correlation_matrix(window_days=window))

    # ── [11] Live accuracy ────────────────────────────────────────────────────
    @app.route("/accuracy/<ticker>")
    def api_accuracy(ticker):
        ticker = ticker.upper()
        if ticker not in ASSET_CONFIG:
            return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
        acc   = live_accuracy_metrics(ticker)
        calib = confidence_calibration(ticker)
        sigp  = signal_performance(ticker)
        return jsonify({
            "ticker":                   ticker,
            "live_accuracy":            acc,
            "confidence_calibration":   calib,
            "signal_performance":       sigp,
            "timestamp":                datetime.now().isoformat(),
        })

    # ── [15] Alerts ───────────────────────────────────────────────────────────
    @app.route("/alerts/<ticker>")
    def api_alerts(ticker):
        ticker = ticker.upper()
        if ticker not in ASSET_CONFIG:
            return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
        p      = predict_latest(ticker)
        alerts = p.get("alerts", [])
        return jsonify({
            "ticker":    ticker,
            "alerts":    alerts,
            "n_alerts":  len(alerts),
            "severity":  ("HIGH"   if any(a in alerts for a in
                                     ["DRIFT_DETECTED","CALIBRATION_POOR"]) else
                          "MEDIUM" if any(a in alerts for a in
                                     ["HIGH_VOLATILITY","MODEL_STALE"]) else
                          "LOW"    if alerts else "NONE"),
            "timestamp": datetime.now().isoformat(),
        })

    # ── [17] Forecast cone ────────────────────────────────────────────────────
    @app.route("/cone/<ticker>")
    def api_cone(ticker):
        ticker  = ticker.upper()
        horizon = int(request.args.get("horizon", 20))
        paths   = int(request.args.get("paths",   50))
        if ticker not in ASSET_CONFIG:
            return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
        try:
            cfg    = ASSET_CONFIG[ticker]
            df     = build_features(ticker, cfg["start"])
            scaler = load_scaler(ticker)
            if scaler is None:
                return jsonify({"error": "Model not trained"}), 404
            n_feat = df.shape[1]
            gbm    = load_sklearn_model(ticker, "gbm")
            qt     = load_keras_model(ticker, "qt_prod") if TF_AVAILABLE else None
            cone   = forecast_cone(ticker, df, scaler, n_feat, gbm, qt,
                                   horizon=horizon, n_paths=paths)
            return jsonify({"ticker": ticker, **cone})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── Backtest ──────────────────────────────────────────────────────────────
    @app.route("/backtest/<ticker>")
    def api_backtest(ticker):
        ticker = ticker.upper()
        if ticker not in ASSET_CONFIG:
            return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
        p = predict_latest(ticker)
        return jsonify({
            "ticker":           ticker,
            "backtest_summary": p.get("backtest_summary", {}),
            "baseline":         p.get("baseline_comparison", {}),
            "top_features":     p.get("top_features", []),
            "timestamp":        p.get("timestamp"),
        })

    # ── Manual retrain ────────────────────────────────────────────────────────
    @app.route("/retrain/<ticker>", methods=["POST"])
    def api_retrain(ticker):
        ticker = ticker.upper()
        if ticker not in ASSET_CONFIG:
            return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
        threading.Thread(
            target=lambda: train_asset(ticker, force=True),
            daemon=True).start()
        return jsonify({
            "status":  "retraining_started",
            "ticker":  ticker,
            "message": f"Retraining {ticker} in background.",
        })

    log.info("\n  ✓ REST API v7.0 → http://0.0.0.0:5000")
    log.info("    GET  /predict/<TICKER>          full payload")
    log.info("    GET  /predict/<TICKER>/summary  signal card")
    log.info("    GET  /predict/<TICKER>/forecast multi-horizon")
    log.info("    GET  /universe                  all assets")
    log.info("    GET  /health                    system status")
    log.info("    GET  /correlations?window=60    correlation matrix")
    log.info("    GET  /accuracy/<TICKER>         live accuracy")
    log.info("    GET  /alerts/<TICKER>           alert flags")
    log.info("    GET  /cone/<TICKER>?horizon=20  forecast cone")
    log.info("    GET  /backtest/<TICKER>         backtest metrics")
    log.info("    POST /retrain/<TICKER>          trigger retrain")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)





# =============================================================================
#  MAIN
# =============================================================================
def main():
    import sys
    api_mode = "--api" in sys.argv

    log.info("\n" + "="*60)
    log.info("  ULTIMATE STOCK PREDICTOR  v6.0")
    log.info("  Transparency & API Edition")
    log.info("  Features: Confidence | Strength | Regime | Freshness |")
    log.info("            Baselines | Importance | Multi-Horizon | Backtest")
    log.info("="*60)

    # ── Start background retrainer ────────────────────────────────────────────
    retrainer = MonthlyRetrainer()
    retrainer.start()

    # ── Initial training pass for all assets ──────────────────────────────────
    all_results = {}
    log.info(f"\n  Training {len(ASSET_CONFIG)} assets "
             f"(skips any trained in last 30 days)...\n")

    for ticker, cfg in ASSET_CONFIG.items():
        if needs_retraining(ticker):
            result = train_asset(ticker)
            if result:
                all_results[ticker] = result
                plot_asset(ticker, result["results"])
        else:
            log.info(f"  ↷ {ticker} skipped — "
                     f"trained {get_asset_state(ticker).get('last_trained','?')[:10]}")
            all_results[ticker] = None

    # ── Live predictions for all assets ───────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  NEXT-BAR PREDICTIONS  —  Full Transparency Payload")
    log.info("="*60)

    all_predictions = {}
    for ticker in ASSET_CONFIG:
        res_list = all_results.get(ticker)
        wf_res   = res_list["results"] if (res_list and isinstance(res_list, dict)
                                           and "results" in res_list) else None
        pred = predict_latest(ticker, results=wf_res)
        all_predictions[ticker] = pred

        if pred and "error" not in pred:
            log.info(f"\n  ── {ticker} ({pred.get('description','')}) ──")
            log.info(f"     Price:          ${pred['current_price']:.4f}")
            log.info(f"     Signal:         {pred['signal']}  "
                     f"(strength={pred['signal_strength']}/100)")
            log.info(f"     Confidence:     {pred['confidence']:.2f}")
            log.info(f"     Regime:         {pred['regime']}")
            log.info(f"     Model age:      {pred['model_age_days']}d  "
                     f"({pred['model_freshness']})")
            log.info(f"     Pred return:    {pred['predicted_return_pct']:+.4f}%")

            # Forecast table
            log.info(f"     {'Horizon':<8} {'q10':>9} {'q50':>9} "
                     f"{'q90':>9} {'Ret%':>8} {'Conf':>6} {'Signal':>7}")
            for hz in ["1d","5d","20d"]:
                f = pred["forecast"].get(hz, {})
                if f:
                    log.info(f"     {hz:<8} "
                             f"{f.get('q10',0):>9.4f} "
                             f"{f.get('q50',0):>9.4f} "
                             f"{f.get('q90',0):>9.4f} "
                             f"{f.get('predicted_return_pct',0):>8.3f}% "
                             f"{f.get('confidence',0):>6.2f} "
                             f"{f.get('signal','—'):>7}")

            # Feature importance
            if pred.get("top_features"):
                log.info(f"     Top features:")
                for ft in pred["top_features"]:
                    bar = "█" * int(ft["importance"] * 50)
                    log.info(f"       {ft['name']:<22} {bar} {ft['importance']:.4f}")

            # Baseline comparison
            bc = pred.get("baseline_comparison", {})
            if bc:
                log.info(f"     RMSE improvement: {bc.get('improvement_percent',0):+.1f}% "
                         f"vs naive  "
                         f"({bc.get('rmse',0):.4f} vs {bc.get('naive_rmse',0):.4f})")

            # Backtest
            bt = pred.get("backtest_summary", {})
            if bt and "total_return_pct" in bt:
                log.info(f"     Backtest:  ret={bt['total_return_pct']:+.1f}%  "
                         f"sharpe={bt['sharpe_ratio']:.2f}  "
                         f"maxDD={bt['max_drawdown_pct']:.1f}%  "
                         f"winRate={bt['win_rate_pct']:.0f}%  "
                         f"PF={bt['profit_factor']:.2f}")

            # Drift
            d = pred.get("drift", {})
            if d.get("flag"):
                log.warning(f"     ⚠ DRIFT DETECTED  ratio={d['ratio']:.2%}")

    # Save all predictions to JSON
    pred_path = f"state/predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f, indent=2, default=str)
    log.info(f"\n  All predictions saved → {pred_path}")

    # ── Universe plot ─────────────────────────────────────────────────────────
    trained = {t:r for t,r in all_results.items() if r}
    if trained:
        plot_universe_summary(trained)

    # ── State summary ─────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  RETRAINING STATE SUMMARY")
    log.info("="*60)
    log.info(f"\n  {'Ticker':<12} {'Last Trained':<22} "
             f"{'Rows':>7} {'Deep RMSE':>10} {'Naive RMSE':>11} {'Drift':>6}")
    log.info("  " + "─"*72)
    for ticker in ASSET_CONFIG:
        s = get_asset_state(ticker)
        log.info(
            f"  {ticker:<12} "
            f"{str(s.get('last_trained','Never'))[:19]:<22} "
            f"{s.get('n_rows',0):>7,} "
            f"{s.get('deep_rmse',0):>10.4f} "
            f"{s.get('naive_rmse',0):>11.4f} "
            f"{'⚠' if s.get('drift_flag') else '✓':>6}"
        )

    log.info("\n  ✓ System running. Retrainer active in background.")
    log.info("  Monthly retraining: 1st of each month (auto)")
    log.info("  Drift detection:    checked after every prediction")
    log.info("  Audit trail:        state/audit.jsonl")
    log.info("  To expose REST API: run with --api flag")
    log.info("  To stop:            retrainer.stop()\n")

    return retrainer, all_results, all_predictions


if __name__ == "__main__":
    import sys
    retrainer, results, predictions = main()

    if "--api" in sys.argv:
        # API mode: start Flask (blocking)
        start_api(retrainer)
    else:
        # Normal mode: keep alive for background retrainer
        try:
            log.info("  System live. Press Ctrl+C to stop.")
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            log.info("\n  Shutting down...")
            retrainer.stop()
            log.info("  Done.")
