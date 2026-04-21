# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pymupdf>=1.27.2.2",
#     "python-pptx>=1.0.2",
# ]
# ///
"""Generate static plot+leaderboard PNGs for the slide deck.

For each dataset, we render one figure per context window size, containing:
  - a time series plot (full history + training window highlight + forecasts)
  - a leaderboard table underneath, ranked by MAE, with the best row highlighted

Output: assets/{dataset}_ctx{n}.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "assets"
OUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "history": "#c0c5ce",
    "context": "#2c3e50",
    "actual": "#1a1a2e",
    "AutoARIMA": "#e74c3c",
    "Chronos": "#3498db",
    "TimesFM": "#2ecc71",
}

DATASETS = {
    "AirPassengers": {
        "freq": "MS",
        "season_length": 12,
        "log_transform": True,
        "has_negatives": False,
        "ylabel": "Passengers (thousands)",
        "total_len": 144,
        "horizon": 24,
        "context_sizes": [120, 48, 24],
    },
    "ETTh1": {
        "freq": "h",
        "season_length": 24,
        "log_transform": False,
        "has_negatives": True,
        "ylabel": "Oil temperature",
        "total_len": 2168,
        "horizon": 168,
        "context_sizes": [2000, 500, 96],
    },
}


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def load_airpassengers() -> pd.DataFrame:
    import statsmodels.api as sm

    raw = sm.datasets.get_rdataset("AirPassengers").data
    return pd.DataFrame(
        {
            "unique_id": "AirPassengers",
            "ds": pd.date_range(start="1949-01", periods=len(raw), freq="MS"),
            "y": raw["value"].values.astype(float),
        }
    )


def load_etth1() -> pd.DataFrame:
    from datasetsforecast.long_horizon import LongHorizon

    df_ett, *_ = LongHorizon.load(directory="./data", group="ETTh1")
    df_ett["ds"] = pd.to_datetime(df_ett["ds"])
    return df_ett


def forecast_arima(train: pd.DataFrame, freq, season_length, h, log_transform):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    df = train.copy()
    df["unique_id"] = "series"
    if log_transform:
        df["y"] = np.log(df["y"])
    sf = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq=freq)
    fc = sf.forecast(df=df, h=h)
    mean = fc["AutoARIMA"].values
    if log_transform:
        mean = np.exp(mean)
    return mean


def forecast_chronos(pipeline, y_context, h):
    import torch

    context = torch.tensor(y_context, dtype=torch.float32)
    quantiles, _ = pipeline.predict_quantiles(
        context.unsqueeze(0), prediction_length=h, quantile_levels=[0.5]
    )
    return quantiles.squeeze(0).squeeze(-1).numpy()


def forecast_timesfm(tfm, y_context, h):
    from timesfm import ForecastConfig

    max_h = max(128, ((h - 1) // 128 + 1) * 128)
    tfm.compile(ForecastConfig(max_context=512, max_horizon=max_h))
    mean, _ = tfm.forecast(horizon=h, inputs=[np.asarray(y_context, dtype=np.float32)])
    return mean.squeeze()[:h]


def render(dataset_name: str, df_full: pd.DataFrame, ctx_size: int, cfg: dict,
           chronos_pipe, tfm_model):
    total = cfg["total_len"]
    h = cfg["horizon"]
    df = df_full.tail(total).reset_index(drop=True) if dataset_name == "ETTh1" else df_full.copy()

    test_start = total - h
    ctx_start = test_start - ctx_size

    ds_full = df["ds"]
    y_full = df["y"].values
    ds_test = df["ds"].iloc[test_start : test_start + h].reset_index(drop=True)
    y_test = y_full[test_start : test_start + h]
    train = df.iloc[ctx_start:test_start][["ds", "y"]].reset_index(drop=True)
    y_ctx = train["y"].values

    # --- forecasts ---
    results: dict[str, np.ndarray] = {}
    print(f"[{dataset_name} ctx={ctx_size}] AutoARIMA...")
    try:
        results["AutoARIMA"] = forecast_arima(
            train, cfg["freq"], cfg["season_length"], h, cfg["log_transform"]
        )
    except Exception as e:
        print(f"  AutoARIMA FAILED: {e}")

    print(f"[{dataset_name} ctx={ctx_size}] Chronos...")
    try:
        results["Chronos"] = forecast_chronos(chronos_pipe, y_ctx, h)
    except Exception as e:
        print(f"  Chronos FAILED: {e}")

    print(f"[{dataset_name} ctx={ctx_size}] TimesFM...")
    try:
        results["TimesFM"] = forecast_timesfm(tfm_model, y_ctx, h)
    except Exception as e:
        print(f"  TimesFM FAILED: {e}")

    # --- metrics table ---
    rows = []
    for name, mean in results.items():
        row = {"Model": name, "MAE": mae(y_test, mean), "RMSE": rmse(y_test, mean)}
        if not cfg["has_negatives"]:
            row["MAPE"] = mape(y_test, mean)
        rows.append(row)
    df_m = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    # --- figure ---
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1], hspace=0.28)

    ax = fig.add_subplot(gs[0])
    # Full history (light grey)
    ax.plot(ds_full, y_full, color=COLORS["history"], linewidth=1.0,
            label="Full history", zorder=1)
    # Training window highlighted
    ax.plot(train["ds"], train["y"], color=COLORS["context"], linewidth=2.2,
            label=f"Training window ({ctx_size} pts)", zorder=3)
    # Shaded region for training window
    ax.axvspan(train["ds"].iloc[0], train["ds"].iloc[-1],
               color=COLORS["context"], alpha=0.06, zorder=0)
    # Cutoff line
    ax.axvline(ds_test.iloc[0], color="#888", linestyle="--", linewidth=1,
               zorder=2)
    ax.text(ds_test.iloc[0], ax.get_ylim()[1] if False else y_full.max(),
            " forecast start", color="#666", fontsize=10, va="top", ha="left")
    # Actual test
    ax.plot(ds_test, y_test, color=COLORS["actual"], linewidth=2.2,
            label="Actual", zorder=4)
    # Model forecasts
    for name, mean in results.items():
        ax.plot(ds_test, mean, color=COLORS[name], linewidth=2.2,
                label=name, zorder=5)

    ax.set_title(f"{dataset_name} — context = {ctx_size} points, horizon = {h}",
                 fontsize=16, pad=10, loc="left")
    ax.set_ylabel(cfg["ylabel"], fontsize=12)
    ax.grid(True, color="#f0f0f0", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=11, frameon=False, ncols=3)

    # --- leaderboard as matplotlib table ---
    ax_t = fig.add_subplot(gs[1])
    ax_t.axis("off")

    cols = ["Rank", "Model"] + [c for c in ("MAE", "RMSE", "MAPE") if c in df_m.columns]
    table_data = []
    fmt_mae = lambda v: f"{v:.2f}"
    fmt_rmse = lambda v: f"{v:.2f}"
    fmt_mape = lambda v: f"{v:.1f}%"
    for i, row in df_m.iterrows():
        r = [f"#{i+1}", row["Model"]]
        r.append(fmt_mae(row["MAE"]))
        r.append(fmt_rmse(row["RMSE"]))
        if "MAPE" in df_m.columns:
            r.append(fmt_mape(row["MAPE"]))
        table_data.append(r)

    tbl = ax_t.table(cellText=table_data, colLabels=cols,
                     loc="center", cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.0)

    # Style header + best cells
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ddd")
        if r == 0:
            cell.set_facecolor("#1a1a2e")
            cell.set_text_props(color="white", weight="bold")
        else:
            if cols[c] == "Model":
                model = table_data[r - 1][c]
                cell.set_text_props(color=COLORS.get(model, "#333"), weight="bold")
            if cols[c] == "Rank" and r == 1:
                cell.set_text_props(color="#b7791f", weight="bold")
            if r == 1 and cols[c] in ("MAE", "RMSE", "MAPE"):
                cell.set_facecolor("#d5f5e3")
                cell.set_text_props(color="#1e8449", weight="bold")

    out = OUT_DIR / f"{dataset_name}_ctx{ctx_size}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    print("Loading datasets...")
    ap = load_airpassengers()
    ett = load_etth1()

    print("Loading Chronos...")
    from chronos import ChronosPipeline
    chronos_pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu"
    )

    print("Loading TimesFM (may download ~800MB)...")
    from timesfm import TimesFM_2p5_200M_torch
    tfm_model = TimesFM_2p5_200M_torch(torch_compile=False)

    for name, cfg in DATASETS.items():
        df = ap if name == "AirPassengers" else ett
        for ctx in cfg["context_sizes"]:
            render(name, df, ctx, cfg, chronos_pipe, tfm_model)


if __name__ == "__main__":
    main()
