"""
wsgi.py — Render/gunicorn entry point for stock_predictor_v7.py
Usage:  gunicorn wsgi:app
"""

import os
import sys
import threading

# ── 1. Create all directories FIRST ──────────────────────────────────────────
# The main module sets up logging.FileHandler("state/predictor.log") at import
# time, before it creates the directories. We must create them first or the
# import crashes on a fresh Render instance.
from pathlib import Path
for _d in ["models", "state", "reports", "plots", "ledger"]:
    Path(_d).mkdir(exist_ok=True)

# ── 2. Force a headless matplotlib backend ────────────────────────────────────
# Must happen before any import of matplotlib.pyplot (including inside the
# predictor module).  Render has no display; 'Agg' writes to files only.
import matplotlib
matplotlib.use("Agg")

# ── 3. Silence plt.show() calls ───────────────────────────────────────────────
# The predictor calls plt.show() after every plot.  That's fine locally but
# raises an error (or hangs) on a headless server.
import matplotlib.pyplot as _plt
_plt.show = lambda *_a, **_kw: None

# ── 4. Intercept Flask.run() so we can capture the app object ─────────────────
# start_api() creates the Flask app, wires all the routes, then ends with
# app.run(...) — which would block forever and prevent gunicorn from ever
# getting hold of the app.  We temporarily replace Flask.run with a no-op
# that stores the instance; then restore the real run afterwards.
from flask import Flask as _Flask

_captured_apps: list = []
_real_run = _Flask.run

def _capture_run(self, *args, **kwargs):          # called instead of app.run()
    _captured_apps.append(self)

_Flask.run = _capture_run                         # monkey-patch

# ── 5. Import the predictor (safe now that dirs + backend are ready) ──────────
import stock_predictor_v7 as sp                   # noqa: E402  (intentional late import)

# ── 6. Restore real Flask.run (so manual app.run() still works if needed) ─────
_Flask.run = _real_run

# ── 7. Build the Flask app via start_api() ────────────────────────────────────
_retrainer = sp.MonthlyRetrainer()
_retrainer.start()

sp.start_api(_retrainer)                          # registers all routes; run() is now a no-op

if not _captured_apps:
    raise RuntimeError(
        "start_api() did not call app.run() — could not capture the Flask app. "
        "Check that flask is installed and start_api() is wired correctly."
    )

app = _captured_apps[0]                           # ← gunicorn binds to this

# ── 8. Background initial training ───────────────────────────────────────────
# Training 15 assets sequentially takes many minutes and would cause Render's
# health-check to time out if done synchronously.  We fire it off in a daemon
# thread so the API is immediately available (returning 503 / empty predictions
# for untrained assets) while training completes in the background.
def _background_train():
    for ticker in sp.ASSET_CONFIG:
        try:
            if sp.needs_retraining(ticker):
                sp.train_asset(ticker)
        except Exception as exc:                  # never let one failure kill the thread
            sp.log.error(f"Background training failed for {ticker}: {exc}")

threading.Thread(target=_background_train, daemon=True, name="InitialTrainer").start()

# ── 9. For local testing ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
