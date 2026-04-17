# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research and presentation repository for a ~30-minute talk on **time series foundation models**. It contains a presentation outline, a live-demo Jupyter notebook comparing traditional vs. foundation model forecasting, and reference materials (survey paper PDF, architecture video, diagrams).

## Commands

Uses **uv** for Python package management (Python 3.11).

```bash
make setup   # uv sync — install all dependencies
make lab     # uv run jupyter lab — launch Jupyter
make app     # uv run streamlit run app.py — launch Streamlit dashboard
make clean   # rm -rf .venv
```

To run a single script or command: `uv run python <script.py>` or `uv run <command>`.

Note: `timesfm` is installed from git source (not PyPI) — see `[tool.uv.sources]` in `pyproject.toml`.

## Repository Structure

- `notes.md` — Full presentation outline (~10 sections) and notebook plan with key talking points
- `demo.ipynb` — Live-demo notebook: compares **AutoARIMA** (statsforecast) vs. **Chronos** (Amazon, zero-shot) vs. **TimesFM** (Google, zero-shot) on AirPassengers and ETTh1 datasets
- `data/` — Downloaded datasets cached by `datasetsforecast` (ETTh1); AirPassengers comes from `statsmodels`
- `time series foundational model survey paper.pdf` — Reference survey paper
- `architecture.mp4`, `TimeSeriesFM-2-Separators.width-1250.png` — Visual reference materials

## Notebook Details

The notebook (`demo.ipynb`) runs entirely on CPU. Chronos and TimesFM download model weights from HuggingFace on first run (~185MB and ~800MB respectively).

Key libraries used in the notebook: `statsforecast` (AutoARIMA baseline), `chronos-forecasting` (Amazon's tokenization-based FM), `timesfm` (Google's patch-based FM), `datasetsforecast` (ETTh1 dataset loader).
