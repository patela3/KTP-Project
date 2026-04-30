# ⚡ Semantic Stock Analysis Tool

A production-grade pipeline that ingests global financial news, runs LLM-powered
sentiment analysis via Gemini, maps signals to S&P 500 sectors, predicts price
impact with an ML ensemble, and renders an interactive dark-mode dashboard.

```
┌─────────────────────────────────────────────────────────────────┐
│  RSS Feeds + HTML Scraping  →  BeautifulSoup News Scraper       │
│           ↓                                                      │
│  Keyword/Entity Matching    →  S&P 500 Sector Mapper            │
│           ↓                                                      │
│  Gemini API (+ Lexicon FB)  →  Sentiment & Impact Analyzer      │
│           ↓                                                      │
│  AdaBoost + GBM + PyTorch   →  Ensemble Price Predictor         │
│           ↓                                                      │
│  Seaborn + Matplotlib       →  Interactive Dashboard            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

```bash
# Clone / unzip the project
cd stock_analyzer

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# PyTorch (choose your version):
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Quick Start

### Demo mode (synthetic data, no API key needed)
```bash
python main.py --demo
```
Generates a full dashboard instantly using simulated news data.

### Live mode (real news scraping)
```bash
python main.py
```

### With Gemini LLM analysis
```bash
# Get a free Gemini API key at https://aistudio.google.com/
export GEMINI_API_KEY="your_key_here"
python main.py
# Or pass directly:
python main.py --gemini-key YOUR_KEY
```

### All options
```bash
python main.py --help

Options:
  --gemini-key KEY      Gemini API key for LLM sentiment analysis
  --max-articles N      Max articles per source (default: 80)
  --days-back N         Days of news history to fetch (default: 2)
  --full-text           Fetch full article bodies (slower but better signals)
  --retrain             Force retrain ML models from scratch
  --use-cached          Skip scraping, use data/raw_news.csv
  --demo                Quick demo with synthetic data
  --output-dir PATH     Where to save outputs (default: outputs/)
```

---

## 📁 Project Structure

```
stock_analyzer/
├── main.py                  # Orchestrator — run this
├── news_scraper.py          # BeautifulSoup + RSS multi-source scraper
├── sector_mapper.py         # Keyword → S&P 500 GICS sector mapping
├── sentiment_analyzer.py    # Gemini LLM + lexicon sentiment analysis
├── ml_models.py             # AdaBoost + GBM + PyTorch ensemble
├── dashboard.py             # Seaborn/Matplotlib visualization
├── requirements.txt
│
├── data/                    # Scraped articles cached here
│   └── raw_news.csv
├── models/                  # Trained ML model files
│   ├── adaboost.pkl
│   ├── gbm.pkl
│   ├── scaler.pkl
│   └── nn_model.pt
└── outputs/                 # Generated dashboards & reports
    ├── dashboard_YYYYMMDD_HHMM.png
    ├── sector_report_YYYYMMDD_HHMM.csv
    └── full_analysis_YYYYMMDD_HHMM.csv
```

---

## 🏗️ Architecture Deep Dive

### 1. News Scraper (`news_scraper.py`)
- **RSS feeds**: Reuters, Bloomberg, CNBC, MarketWatch, Yahoo Finance, Seeking Alpha, Financial Times, Investing.com
- **Direct HTML scraping**: Yahoo Finance top stories, MarketWatch latest
- Rate-limiting and polite delays built in
- Deduplication by URL hash
- Optional full-text extraction with site-aware CSS selectors

### 2. Sector Mapper (`sector_mapper.py`)
- Maps all 11 S&P 500 GICS sectors (IT, Healthcare, Financials, Energy, etc.)
- Keyword matching on titles + body text (title double-weighted)
- Macro signal detection: rate hikes/cuts, recession, inflation, trade war, earnings
- Returns top-5 sectors with relevance scores per article

### 3. Sentiment Analyzer (`sentiment_analyzer.py`)
- **Primary**: Google Gemini 1.5 Flash API
  - Returns: sentiment (-1→+1), confidence, impact magnitude, price impact %, time horizon, key drivers
  - Rate-limited to stay within free tier (15 RPM)
- **Fallback**: Financial lexicon (1000+ bullish/bearish terms)
  - Instant, no API required
  - Confidence capped at 75% (lower than LLM)

### 4. ML Ensemble (`ml_models.py`)
- **AdaBoost**: 100 estimators, DecisionTree base, adaptive weight allocation
- **Gradient Boosting**: 100 estimators, learning rate 0.05
- **PyTorch Neural Net**: 3-layer feedforward with LayerNorm + Dropout
  - Architecture: 10 → 64 → 32 → 16 → 1
- **Ensemble weights**: Dynamically set based on test-set MAE (lower MAE = more weight)
- Trained on synthetic data by default; swap `generate_synthetic_training_data()` for real historical data
- Outputs 95% confidence intervals per prediction

**Feature vector (10 dimensions)**:
```
sentiment_score, confidence, sentiment_encoded, magnitude_encoded,
horizon_encoded, sector_relevance, sector_encoded,
sentiment_score², confidence×|sentiment|, relevance×confidence
```

### 5. Dashboard (`dashboard.py`)
Six-panel dark-mode visualization:
1. **Sector Impact Bar** — Predicted % price change per sector with CI error bars
2. **Source Donut** — Article distribution by news source
3. **Sentiment Heatmap** — Bullish ratio, confidence, and score per sector
4. **Signal Scatter** — Sentiment score vs predicted impact (sized by confidence)
5. **Top Drivers Lollipop** — Top N articles by signal strength with source attribution
6. **Volume Timeline** — News flow by sentiment over time

---

## 🔑 API Keys

| Service | Required? | Where to get |
|---------|-----------|--------------|
| Gemini API | Optional (has lexicon fallback) | [aistudio.google.com](https://aistudio.google.com) — free tier: 15 RPM |

---

## 📊 Running on Real Historical Data

To train the ML models on real data instead of synthetic:

```python
import yfinance as yf
import pandas as pd

# 1. Collect historical articles with your scraper
# 2. Manually label or auto-match to next-day price changes
# 3. Build a DataFrame with 'actual_price_change_pct' column
# 4. Train:

from ml_models import EnsemblePredictor
predictor = EnsemblePredictor()
predictor.train(df=your_historical_df)
```

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. Predictions are
statistical estimates, not financial advice. Past correlations between sentiment
and price movement do not guarantee future results.
