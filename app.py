import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLORS = {
    "actual": "#2c3e50",
    "history": "#c0c5ce",
    "context": "#636e78",
    "arima": "#e74c3c",
    "chronos": "#3498db",
    "timesfm": "#2ecc71",
}

DATASETS = {
    "AirPassengers": {
        "freq": "MS",
        "season_length": 12,
        "log_transform": True,
        "has_negatives": False,
        "ylabel": "Passengers (thousands)",
        "total_len": 144,
        "default_context": 120,
        "default_horizon": 24,
        "max_horizon": 48,
        "min_context": 12,
    },
    "ETTh1": {
        "freq": "h",
        "season_length": 24,
        "log_transform": False,
        "has_negatives": True,
        "ylabel": "Temperature",
        "total_len": 2168,
        "default_context": 2000,
        "default_horizon": 168,
        "max_horizon": 336,
        "min_context": 48,
    },
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_airpassengers():
    import statsmodels.api as sm

    raw = sm.datasets.get_rdataset("AirPassengers").data
    return pd.DataFrame(
        {
            "unique_id": "AirPassengers",
            "ds": pd.date_range(start="1949-01", periods=len(raw), freq="MS"),
            "y": raw["value"].values.astype(float),
        }
    )


@st.cache_data
def load_etth1():
    from datasetsforecast.long_horizon import LongHorizon

    df_ett, *_ = LongHorizon.load(directory="./data", group="ETTh1")
    df_ett["ds"] = pd.to_datetime(df_ett["ds"])
    return df_ett


def load_dataset(name: str) -> pd.DataFrame:
    if name == "AirPassengers":
        return load_airpassengers()
    return load_etth1()


# ---------------------------------------------------------------------------
# Model loaders (cached as resources — loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_chronos():
    from chronos import ChronosPipeline

    return ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu"
    )


@st.cache_resource
def load_timesfm():
    from timesfm import TimesFM_2p5_200M_torch

    return TimesFM_2p5_200M_torch(torch_compile=False)


# ---------------------------------------------------------------------------
# Forecast functions (cached by inputs)
# ---------------------------------------------------------------------------

@st.cache_data
def forecast_arima(train_y: tuple, train_ds: tuple, freq: str, season_length: int, h: int, log_transform: bool):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    df_train = pd.DataFrame(
        {"unique_id": "series", "ds": list(train_ds), "y": list(train_y)}
    )
    if log_transform:
        df_train["y"] = np.log(df_train["y"])

    sf = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq=freq)
    fc = sf.forecast(df=df_train, h=h, level=[90])

    if log_transform:
        fc["AutoARIMA"] = np.exp(fc["AutoARIMA"])
        fc["AutoARIMA-lo-90"] = np.exp(fc["AutoARIMA-lo-90"])
        fc["AutoARIMA-hi-90"] = np.exp(fc["AutoARIMA-hi-90"])

    return (
        fc["AutoARIMA"].values,
        fc["AutoARIMA-lo-90"].values,
        fc["AutoARIMA-hi-90"].values,
    )


@st.cache_data
def forecast_chronos(_pipeline, train_y: tuple, h: int):
    import torch

    context = torch.tensor(list(train_y), dtype=torch.float32)
    quantiles, mean = _pipeline.predict_quantiles(
        context.unsqueeze(0),
        prediction_length=h,
        quantile_levels=[0.05, 0.5, 0.95],
    )
    return (
        mean.squeeze().numpy(),
        quantiles.squeeze()[:, 0].numpy(),
        quantiles.squeeze()[:, 2].numpy(),
    )


@st.cache_data
def forecast_timesfm(_tfm, train_y: tuple, h: int):
    from timesfm import ForecastConfig

    max_h = max(128, ((h - 1) // 128 + 1) * 128)
    _tfm.compile(ForecastConfig(max_context=512, max_horizon=max_h))

    mean_fc, quantile_fc = _tfm.forecast(
        horizon=h, inputs=[np.array(train_y, dtype=np.float32)]
    )
    return (
        mean_fc.squeeze()[:h],
        quantile_fc.squeeze()[:h, 0],
        quantile_fc.squeeze()[:h, -1],
    )


# ---------------------------------------------------------------------------
# Plotly chart builder
# ---------------------------------------------------------------------------

def build_figure(ds_full, y_full, context_start, context_end, ds_test, y_test, results, ylabel):
    fig = go.Figure()

    # Full history (light grey, thin)
    fig.add_trace(go.Scatter(
        x=ds_full, y=y_full,
        mode="lines",
        line=dict(color=COLORS["history"], width=1),
        name="Full history",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>History</extra>",
    ))

    # Context window (darker, thicker)
    fig.add_trace(go.Scatter(
        x=ds_full[context_start:context_end],
        y=y_full[context_start:context_end],
        mode="lines",
        line=dict(color=COLORS["context"], width=2.5),
        name=f"Context ({context_end - context_start} pts)",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>Context</extra>",
    ))

    # Context window shading
    ctx_ds = ds_full[context_start:context_end]
    if len(ctx_ds) > 0:
        fig.add_vrect(
            x0=ctx_ds.iloc[0], x1=ctx_ds.iloc[-1],
            fillcolor="rgba(99, 110, 120, 0.07)",
            line_width=0,
            layer="below",
        )

    # Forecast cutoff line
    cutoff = ds_test.iloc[0]
    fig.add_shape(
        type="line", x0=cutoff, x1=cutoff, y0=0, y1=1,
        yref="paper", line=dict(color="#aaa", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=cutoff, y=1.03, yref="paper",
        text="forecast start", showarrow=False,
        font=dict(color="#999", size=11),
    )

    # Actual test values
    fig.add_trace(go.Scatter(
        x=ds_test, y=y_test,
        mode="lines",
        line=dict(color=COLORS["actual"], width=2.5),
        name="Actual",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>Actual</extra>",
    ))

    # Model forecasts
    for name, (mean, lo, hi, color) in results.items():
        # Confidence interval (filled area)
        fig.add_trace(go.Scatter(
            x=pd.concat([ds_test, ds_test[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor=color.replace(")", ", 0.12)").replace("rgb", "rgba") if "rgb" in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Mean forecast line
        fig.add_trace(go.Scatter(
            x=ds_test, y=mean,
            mode="lines",
            line=dict(color=color, width=2.5),
            name=name,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>" + name + "</extra>",
        ))

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="",
        yaxis_title=ylabel,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font_size=13,
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0", showgrid=True),
        yaxis=dict(gridcolor="#f0f0f0", showgrid=True),
    )

    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Time Series Foundation Models", layout="wide")
st.title("Time Series Foundation Models")
st.caption("Interactive comparison — drag the sliders to explore")

# Warm both dataset caches up-front so switching datasets is instant.
with st.spinner("Loading datasets..."):
    load_airpassengers()
    load_etth1()

# ---- Sidebar ----
with st.sidebar:
    st.header("Dataset")
    dataset_name = st.radio("Choose dataset", list(DATASETS.keys()), label_visibility="collapsed")
    cfg = DATASETS[dataset_name]

    df_full = load_dataset(dataset_name)
    total = cfg["total_len"]
    if dataset_name == "ETTh1":
        df_full = df_full.tail(total).copy().reset_index(drop=True)

    st.markdown("---")
    st.header("Forecast settings")

    max_h = cfg["max_horizon"]
    h = st.slider("Forecast horizon", min_value=1, max_value=max_h, value=cfg["default_horizon"])

    max_context = total - h
    default_context = min(cfg["default_context"], max_context)
    context_size = st.slider(
        "Context window (data points the model sees)",
        min_value=cfg["min_context"],
        max_value=max_context,
        value=default_context,
    )

    st.markdown("---")
    st.header("Models")
    run_arima = st.toggle("AutoARIMA", value=True)
    run_chronos = st.toggle("Chronos (zero-shot)", value=True)
    run_timesfm = st.toggle("TimesFM (zero-shot)", value=True)

    if run_chronos and h > 64:
        st.warning("Chronos recommends horizon <= 64.")

# ---- Prepare data ----
# The test set is always the last h points
df_test = df_full.tail(h).copy().reset_index(drop=True)
# The context window is the `context_size` points right before the test set
context_end_idx = total - h
context_start_idx = context_end_idx - context_size

df_context = df_full.iloc[context_start_idx:context_end_idx]

ds_full = df_full["ds"]
y_full = df_full["y"].values

ds_test = df_test["ds"]
y_test = df_test["y"].values

y_context = df_context["y"].values
ds_context = df_context["ds"]

# ---- Run forecasts (auto-runs on any change) ----
results = {}
train_y_tuple = tuple(y_context.tolist())
train_ds_tuple = tuple(ds_context.tolist())

if run_arima:
    with st.spinner("AutoARIMA..."):
        try:
            arima_mean, arima_lo, arima_hi = forecast_arima(
                train_y_tuple, train_ds_tuple, cfg["freq"],
                cfg["season_length"], h, cfg["log_transform"],
            )
            results["AutoARIMA"] = (arima_mean, arima_lo, arima_hi, COLORS["arima"])
        except Exception as e:
            st.error(f"AutoARIMA failed: {e}")

if run_chronos:
    with st.spinner("Chronos..."):
        try:
            pipeline = load_chronos()
            c_mean, c_lo, c_hi = forecast_chronos(pipeline, train_y_tuple, h)
            results["Chronos"] = (c_mean, c_lo, c_hi, COLORS["chronos"])
        except Exception as e:
            st.error(f"Chronos failed: {e}")

if run_timesfm:
    with st.spinner("TimesFM..."):
        try:
            tfm = load_timesfm()
            t_mean, t_lo, t_hi = forecast_timesfm(tfm, train_y_tuple, h)
            results["TimesFM"] = (t_mean, t_lo, t_hi, COLORS["timesfm"])
        except Exception as e:
            st.error(f"TimesFM failed: {e}")

# ---- Chart ----
fig = build_figure(
    ds_full, y_full,
    context_start_idx, context_end_idx,
    ds_test, y_test,
    results, cfg["ylabel"],
)
st.plotly_chart(fig, width="stretch")

# ---- Metrics ----
if results:
    st.markdown("### Leaderboard")
    st.caption(
        "**Lower is better** for every metric. "
        "Models are ranked by MAE (the average forecast error). "
        "🥇 is the overall winner; green cells mark the best value in each column."
    )

    metric_keys = ["MAE", "RMSE"]
    if not cfg["has_negatives"]:
        metric_keys.append("MAPE")

    rows = []
    for name, (mean, _lo, _hi, _color) in results.items():
        row = {"Model": name, "MAE": mae(y_test, mean), "RMSE": rmse(y_test, mean)}
        if "MAPE" in metric_keys:
            row["MAPE"] = mape(y_test, mean)
        rows.append(row)

    df_m = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    medals = ["🥇", "🥈", "🥉"]
    df_m.insert(0, "Rank", [medals[i] if i < 3 else f"#{i + 1}" for i in range(len(df_m))])

    def highlight_best(col):
        best = col.min()
        return [
            "background-color: #d5f5e3; font-weight: 700; color: #1e8449;" if v == best else ""
            for v in col
        ]

    fmt = {"MAE": "{:.2f}", "RMSE": "{:.2f}"}
    if "MAPE" in metric_keys:
        fmt["MAPE"] = "{:.1f}%"

    styled = df_m.style.apply(highlight_best, subset=metric_keys).format(fmt)
    st.dataframe(styled, width="stretch", hide_index=True)

    # One-line winner summary
    if len(df_m) > 1:
        winner = df_m.iloc[0]
        worst = df_m.iloc[-1]
        gap_pct = (worst["MAE"] - winner["MAE"]) / worst["MAE"] * 100
        st.success(
            f"🥇 **{winner['Model']}** wins — "
            f"{gap_pct:.0f}% lower MAE than {worst['Model']} "
            f"({winner['MAE']:.2f} vs {worst['MAE']:.2f})."
        )

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **MAE** (Mean Absolute Error): average distance between prediction and truth, "
            "in the same units as the data. Easy to read: *\"off by 10 passengers on average\"*.\n"
            "- **RMSE** (Root Mean Squared Error): same units, but penalises big misses more. "
            "When RMSE is much higher than MAE, the model had a few large errors.\n"
            "- **MAPE** (Mean Absolute Percentage Error): error as a % of the actual value. "
            "Scale-free, but undefined near zero (hidden for datasets that can be negative)."
        )
