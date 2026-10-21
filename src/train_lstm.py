"""Phase 8. LSTM direct multi-output forecaster (56 past days -> next 28 days).

Input per time step: [sales / series_scale, sin(dow), cos(dow), snap, event]. Series scale is the
TRAIN-period mean of each series (no leakage). Run:  python -m src.train_lstm
"""
import json
import numpy as np
import pandas as pd
from backend import config as C
from src.common import load_long, split_dates, holdout_rows, frame_from_rows, save_predictions

MODEL = "lstm"


def main():
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C.ensure_dirs()
    keras.utils.set_random_seed(C.SEED)
    raw = load_long()
    s = split_dates(raw)
    ids = sorted(raw.id.unique())
    wide = lambda col: raw.pivot(index="id", columns="date", values=col).loc[ids]
    sales, snap, event = wide("sales"), wide("snap"), wide("event_flag")
    dates = sales.columns
    W, H = C.LSTM_WINDOW, C.HORIZON
    i_train, i_val, i_test = (dates.get_loc(s[k]) for k in ("train_end", "val_end", "test_end"))

    scale = np.maximum(sales.loc[:, :s["train_end"]].mean(axis=1).values, 0.1)   # TRAIN only
    x_sales = np.nan_to_num(sales.values / scale[:, None])
    dow = dates.dayofweek.values
    tile = lambda v: np.tile(v, (len(ids), 1))
    feats = np.stack([x_sales, tile(np.sin(2 * np.pi * dow / 7)), tile(np.cos(2 * np.pi * dow / 7)),
                      snap.values, event.values], axis=-1).astype("float32")      # (series, days, 5)
    y_all = np.nan_to_num(sales.values / scale[:, None]).astype("float32")

    def make(origins, with_y=True):
        X = np.concatenate([feats[:, t - W + 1:t + 1, :] for t in origins])
        y = np.concatenate([y_all[:, t + 1:t + 1 + H] for t in origins]) if with_y else None
        return X, y

    Xtr, ytr = make(range(W - 1, i_train - H + 1, C.LSTM_STRIDE))   # all 28 target days inside TRAIN
    Xva, yva = make([i_train])                                       # targets = validation period
    Xte, _ = make([i_val], False)                                    # origin = end of validation
    Xfu, _ = make([i_test], False)                                   # origin = last observed day

    model = keras.Sequential([keras.layers.Input((W, 5)), keras.layers.LSTM(64), keras.layers.Dropout(0.2),
                              keras.layers.Dense(64, activation="relu"), keras.layers.Dense(H)])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=C.LSTM_EPOCHS, batch_size=256, verbose=2,
                     callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    back = lambda p: (p * scale[:, None])                            # inverse transform
    te, fut = holdout_rows(raw, s), raw[raw.is_future]
    p_te, p_fu = back(model.predict(Xte, verbose=0)), back(model.predict(Xfu, verbose=0))
    assert te.id.is_monotonic_increasing and list(te.id.unique()) == ids      # row order == matrix order
    save_predictions([frame_from_rows(te, p_te.reshape(-1), MODEL, "test"),
                      frame_from_rows(fut, p_fu.reshape(-1), MODEL, "future")], MODEL)

    model.save(C.MODELS / "lstm_model.keras")
    (C.PREPROCESSING / "lstm_scaler.json").write_text(json.dumps({
        "window": W, "horizon": H, "inputs": ["sales_scaled", "dow_sin", "dow_cos", "snap", "event_flag"],
        "series_scale": dict(zip(ids, scale.round(5).tolist()))}))
    (C.METRICS / "lstm_training.json").write_text(json.dumps({
        "epochs_run": len(hist.history["loss"]), "train_loss": hist.history["loss"], "val_loss": hist.history["val_loss"],
        "train_samples": int(len(Xtr))}, indent=2))
    plt.figure(figsize=(7, 4)); plt.plot(hist.history["loss"], label="train"); plt.plot(hist.history["val_loss"], label="validation")
    plt.xlabel("epoch"); plt.ylabel("MSE (scaled)"); plt.title("LSTM loss"); plt.legend(); plt.tight_layout()
    plt.savefig(C.REPORTS / "lstm_loss.png", dpi=120)
    print("LSTM done. Epochs run:", len(hist.history["loss"]))


if __name__ == "__main__":
    main()
