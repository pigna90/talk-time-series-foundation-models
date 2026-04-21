# Time Series Foundation Models — Talk

Materials for a ~30-minute talk comparing classical forecasting (AutoARIMA) with time series foundation models (**Chronos**, **TimesFM**) in zero-shot settings.

**Slides:** https://pigna90.github.io/talk-time-series-foundation-models/

## What's in here

- `index.html` — reveal.js slide deck (also served via GitHub Pages)
- `demo.ipynb` — live-demo notebook: AutoARIMA vs. Chronos vs. TimesFM on AirPassengers and ETTh1
- `app.py` — Streamlit dashboard for interactively running the same comparison
- `time series foundational model survey paper.pdf` — reference survey
- `architecture.mp4`, `TimeSeriesFM-2-Separators.width-1250.png` — visual references

## Setup

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
make setup    # uv sync — install dependencies
```

`timesfm` is installed from git (see `[tool.uv.sources]` in `pyproject.toml`).

## Run the notebook

```bash
make lab      # uv run jupyter lab
```

Open `demo.ipynb`. Everything runs on CPU. First run downloads model weights from HuggingFace (~185MB for Chronos, ~800MB for TimesFM).

## Run the Streamlit app

```bash
make app      # uv run streamlit run app.py
```

Pick a dataset (AirPassengers or ETTh1), adjust context/horizon, toggle which models to run, and compare forecasts side by side.

## View the slides locally

Just open `index.html` in a browser — no build step.

## Clean

```bash
make clean    # rm -rf .venv
```
