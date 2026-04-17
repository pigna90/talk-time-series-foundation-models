# Notes

## Raw Ideas
- Use example of IBM NASA geospatial foundation model as a hook
- What people say: https://www.reddit.com/r/MachineLearning/comments/1hb7ur1/d_whats_stopping_you_from_using_foundation_models/

---

## Presentation Outline

### 1. Time Series Is Everywhere (2-3 min)
- Quick examples the audience knows: stock prices, weather, IoT sensors, patient vitals, energy demand, retail sales
- Common tasks: **forecasting**, **anomaly detection**, **classification**, **imputation**
- Visual: show a time series (e.g. airline passengers) decomposed into trend + seasonality + residuals
- **Running example introduced here**: energy demand forecasting — we'll use this throughout to show "before vs. after" with foundation models

### 2. How We Do Time Series Today (3-4 min)
- **Timeline visual**: `ARIMA (1970s) → RNNs (1986) → LSTMs (1997) → Transformers (2017) → GPT/BERT (2018-19) → Time Series FMs (2023-24)`
- **Classical methods**: Moving Averages, ARIMA/SARIMA — work well for simple, stationary data
  - Example: SARIMA forecasting airline passengers
  - Limitation: assume stationarity, struggle with non-linear patterns, need manual feature engineering
- **ML methods**: SVMs, Gradient Boosting (XGBoost/LightGBM) — better with non-linearity but still need hand-crafted features
- **Deep Learning**: RNNs → LSTMs/GRUs — capture sequences, but sequential processing is slow, vanishing gradients limit long-range dependencies
- Key pain point: **every new dataset = train a new model from scratch**
- Running example: show the traditional workflow for energy demand — data collection, feature engineering, model selection, training, evaluation... repeat for every new site/region

### 3. The Transformer Revolution (3-4 min)
- 2017: "Attention Is All You Need" — originally for NLP (machine translation)
- Core idea: **self-attention** lets the model look at all positions at once (no sequential bottleneck)
  - Visual: Transformer architecture diagram
  - Intuition: "instead of reading a sentence word-by-word, you see all words simultaneously and learn which relate to each other"
- Why it matters: parallelizable, captures long-range dependencies, scales with data
- This powered GPT, BERT, LLaMA → the LLM revolution in text

### 4. What Is a Foundation Model? (2-3 min)
- **Definition**: a large model pre-trained on massive diverse data, then fine-tuned for specific tasks
- Two-stage process: **pre-training** (learn general patterns) → **fine-tuning** (adapt to your task)
- Analogy: like a medical student who first learns general medicine (pre-training), then specializes in cardiology (fine-tuning)
- "Foundation-al" = one model, many downstream tasks (forecasting, classification, anomaly detection...)
- Real-world example: IBM-NASA geospatial foundation model — pre-trained on satellite imagery, fine-tuned for flood detection, crop monitoring, etc.

### 5. From Text to Time Series: Bridging LLMs and Time Series (4-5 min)
- The key insight: **both text and time series are sequential data** — transformers are good at sequences
- Three strategies to bridge the gap:
  1. **Build from scratch for time series** — design transformer architectures specifically for temporal data
     - Example: TimesFM (Google) patches time series like Vision Transformers patch images
  2. **Adapt an existing LLM** — repurpose GPT/LLaMA for time series with minimal changes
     - Example: Chronos (Amazon) tokenizes continuous values into bins, trains a T5 model on them
     - Example: TIME-LLM reprograms time series into text-like representations
  3. **Freeze the LLM, just change the input/output layers**
     - Example: FPT (Frozen Pretrained Transformer) — freeze GPT-2, only fine-tune embeddings
- The tokenization problem: text has words, time series has continuous values → solutions include patching, binning, digit-by-digit encoding
- **Patching explained**: just like Vision Transformers split an image into patches, we split a time series into fixed-length windows — each patch becomes a "token" the transformer can process. This captures local patterns within each patch while attention captures global dependencies across patches.
  - Visual: side-by-side — image patches (ViT) vs. time series patches (TimesFM)
- Visual: TimeSeriesFM-2 separators diagram (existing PNG)

### 6. One Model, Many Tasks (2-3 min)
- The power of foundation models: **pre-train once, fine-tune for any task** — this is the paradigm shift vs. Section 2 where every task needed a separate model
- Show how the same pre-trained backbone serves different heads:
  - **Forecasting**: predict future values (energy demand, stock prices)
  - **Anomaly detection**: spot outliers in sensor data, fraud detection
  - **Classification**: activity recognition from wearables, disease diagnosis
  - **Imputation**: fill gaps in incomplete datasets (IoT sensor failures)
  - **Generation**: simulation, anonymization, data augmentation
  - **Change point detection**: detect regime shifts (market crashes, equipment failures)
- Visual: diagram showing one pre-trained model with arrows to multiple task-specific heads
- Zero-shot & few-shot capability: some models work on unseen datasets without any fine-tuning
- Running example: same Chronos/TimesFM model used for energy demand forecasting can also detect anomalies in the same data — no retraining needed

### 7. The Landscape: Main Models & Packages (4-5 min)

| Model | By | Architecture | Key Feature | Open Source |
|---|---|---|---|---|
| **Chronos** | Amazon | Adapted T5/GPT-2 | Tokenizes values into bins; probabilistic; ~2M downloads/month | Yes (HuggingFace) |
| **TimesFM** | Google | Decoder-only | Patch-based; strong zero-shot forecasting | Yes (HuggingFace) |
| **MOMENT** | CMU | Encoder-only | "Time Series Pile" pre-training; lightweight | Yes (HuggingFace) |
| **Moirai** | Salesforce | Encoder (Any-variate) | Multivariate; handles covariates; probabilistic | Yes (HuggingFace) |
| **Timer** | Tsinghua/PKU | Decoder-only | Unified S3 format; generative pre-training; up to 1B time points | Yes |
| **TimeGPT** | Nixtla | Encoder-Decoder | First commercial TSFM; simple API; no ML expertise needed | No (API only) |
| **Tiny Time Mixers** | IBM | Non-Transformer | Ultra-lightweight (1M params); fast inference | Yes (HuggingFace) |
| **Lag-Llama** | ServiceNow | Decoder (LLaMA) | Probabilistic; lagged features; univariate | Yes |
| **Granite TTM** | IBM | Non-Transformer | Enterprise-focused; part of IBM Granite family | Yes |

- Practical note: most models available via HuggingFace; Chronos and TimesFM have largest adoption
- TimeGPT (Nixtla) is the main API/SaaS option — easiest to use, but closed-source
- For multivariate needs: Moirai, Timer-XL, Time-MOE, Tiny Time Mixers

### 8. The Honest Take: Challenges & When to Use (3-4 min)
From practitioners (Reddit r/MachineLearning + real-world experience):
- **"My hand-crafted model still wins"** — well-tuned XGBoost or domain-specific models often beat foundation models on specific datasets. Traditional models leverage domain knowledge that FMs lack.
- **"My data isn't in their training distribution"** — proprietary/niche data leads to poor zero-shot performance. No "Common Crawl" equivalent exists for time series.
- **"No universal inductive bias"** — unlike images (all share visual structure) or text (all share language), time series from different domains (audio, finance, weather) may share nothing in common.
- **Explainability is a dealbreaker** — many business contexts need interpretable forecasts; FMs are black boxes. ARIMA is simpler to explain to stakeholders.
- **Multivariate support is limited** — many models are univariate only (Chronos, MOMENT, TimesFM, Lag-Llama). Multi-rate and covariate handling is still immature.
- **Uncertainty quantification** — some models don't even provide confidence intervals; hard to trace uncertainty back to data or assumptions.
- **Data leakage concerns** — hard to verify if the FM was trained on the very dataset you're testing on.
- **Computational cost** — largest models (Time-MOE: 2.4B params) require significant resources vs. a lightweight ARIMA.
- Counter-argument (Nixtla): for the majority of users still running ARIMA or Prophet, a foundation model is a massive, easy upgrade.

**So when should you use them?**
- **Good fit**: zero-shot on new datasets, rapid prototyping, no ML team, diverse forecasting needs
- **Bad fit**: highly domain-specific data with known physics, need for interpretability, multivariate with complex interactions, resource-constrained environments
- **Best strategy today**: use as a strong baseline, compare against your tuned models, fine-tune if promising

### 9. Live Demo / Notebook (5-7 min)
- Show a notebook comparing traditional approach vs. foundation model on same dataset
- (To be built separately)

### 10. Takeaways (1 min)
- Foundation models bring the "pre-train once, fine-tune everywhere" paradigm to time series
- The transformer architecture is the backbone — same revolution that powered ChatGPT
- We're at an inflection point: Chronos, TimesFM, Moirai all released in 2024, tooling is maturing fast
- They won't replace domain expertise, but they dramatically lower the bar for getting good-enough forecasts quickly
- Worth experimenting with — especially as a strong zero-shot baseline

---

**Estimated total: ~30-35 min presentation (including notebook demo)**

---

## Notebook Plan (TODO)

### Packages
- **statsforecast** (Nixtla) — for ARIMA/ETS baseline, fast and clean API
- **chronos-forecasting** (Amazon) — zero-shot FM, pip-installable, runs on CPU, probabilistic output
- **timesfm** (Google) — zero-shot FM, patch-based approach, good contrast with Chronos
- Optional: **uni2ts** (Salesforce Moirai) — only if showing multivariate angle

### Dataset
- Energy demand (from `datasetsforecast`) or ETTh1 (electricity transformer temperature) — classic, relevant, easy to load

### Notebook Flow
1. Load dataset (energy demand), quick EDA + plot
2. Train/test split
3. **ARIMA baseline** with `statsforecast` → show the setup effort + result
4. **Chronos zero-shot** → ~3 lines of code, no training, show result with confidence intervals
5. **TimesFM zero-shot** → same, different approach (patching vs. binning)
6. Compare metrics side by side (MAE, RMSE) in a table + overlay plot
7. Punchline: "the FM took 3 lines and no training, and it's competitive"

### Key Insight to Highlight in Notebook
**"Why are these models so small compared to ChatGPT?"**

| Model | Parameters |
|---|---|
| Chronos-Tiny | 8M |
| Chronos-Base | 200M |
| TimesFM | 200M |
| GPT-4 (estimated) | ~1,700,000M |

- **Language** models need to encode all human knowledge — vocabulary, grammar, facts, reasoning across millions of topics. That requires billions/trillions of parameters.
- **Time series** has a much narrower structure — trends, seasonality, cycles, noise. The patterns are mathematical, not semantic. You don't need to "know" that Paris is in France to forecast energy demand.
- So a 200M model can be state-of-the-art for time series. The complexity of the task determines the model size, not the other way around.
- Nuance: the field is also young. Larger models are appearing (Time-MOE: 2.4B params) and it's an open question whether scaling up will help time series as much as it did for language.
- This is a great talking point during the demo: "these run on your laptop because time series is a fundamentally simpler problem than language"

### Install
```bash
pip install statsforecast chronos-forecasting timesfm datasetsforecast
```

