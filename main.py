"""
main.py
Orchestrator: scrape → map → analyze → predict → visualize.
Run with: python main.py
"""

import os
import argparse
import logging
import pandas as pd
from datetime import datetime

# ── Configure logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/run.log"),
    ],
)
logger = logging.getLogger(__name__)

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


def run_pipeline(
    gemini_api_key: str = "",
    max_articles: int = 100,
    days_back: int = 2,
    include_full_text: bool = False,
    retrain: bool = False,
    use_cached: bool = False,
    output_dir: str = "outputs",
):
    """
    Full end-to-end pipeline:
    1. Scrape financial news
    2. Map to S&P 500 sectors
    3. Analyze sentiment (Gemini LLM or lexicon fallback)
    4. Run ML ensemble price prediction
    5. Generate dashboard
    """

    logger.info("=" * 60)
    logger.info("  SEMANTIC STOCK ANALYSIS PIPELINE")
    logger.info(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ── Step 1: Scrape ─────────────────────────────────────────────────────────
    cache_path = "data/raw_news.csv"

    if use_cached and os.path.exists(cache_path):
        logger.info("Loading cached articles...")
        df = pd.read_csv(cache_path)
        logger.info(f"Loaded {len(df)} cached articles.")
    else:
        from news_scraper import FinancialNewsScraper
        scraper = FinancialNewsScraper(
            max_articles_per_source=max_articles // 8,
            days_back=days_back,
        )
        df = scraper.fetch_all(include_full_text=include_full_text)
        df.to_csv(cache_path, index=False)
        logger.info(f"Scraped {len(df)} articles → saved to {cache_path}")

    if df.empty:
        logger.error("No articles fetched. Exiting.")
        return None

    # ── Step 2: Sector Mapping ─────────────────────────────────────────────────
    logger.info("\n[2/5] Mapping articles to S&P 500 sectors...")
    from sector_mapper import SectorMapper
    mapper = SectorMapper()
    df = mapper.map_dataframe(df)
    logger.info(f"Sectors detected: {df['primary_sector'].value_counts().to_dict()}")

    # ── Step 3: Sentiment Analysis ─────────────────────────────────────────────
    logger.info("\n[3/5] Running sentiment analysis...")
    from sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer(gemini_api_key=gemini_api_key)
    df = analyzer.analyze_dataframe(df)

    sentiment_dist = df["sentiment"].value_counts()
    logger.info(f"Sentiment distribution: {sentiment_dist.to_dict()}")
    method_dist = df["analysis_method"].value_counts()
    logger.info(f"Analysis methods used: {method_dist.to_dict()}")

    # ── Step 4: ML Prediction ──────────────────────────────────────────────────
    logger.info("\n[4/5] Running ML ensemble predictions...")
    from ml_models import EnsemblePredictor
    predictor = EnsemblePredictor()

    loaded = predictor.load()
    if not loaded or retrain:
        logger.info("Training models from scratch...")
        metrics = predictor.train()
        logger.info(f"Training metrics: {metrics}")
    else:
        logger.info("Using pre-trained models.")

    df = predictor.predict_dataframe(df)

    avg_impact = df["predicted_price_impact_pct"].mean()
    logger.info(f"Avg predicted impact: {avg_impact:+.3f}%")
    top_sector = df.groupby("primary_sector")["predicted_price_impact_pct"].mean().idxmax()
    logger.info(f"Most bullish sector: {top_sector}")

    # ── Step 5: Dashboard ──────────────────────────────────────────────────────
    logger.info("\n[5/5] Generating dashboard...")
    from dashboard import generate_dashboard, generate_sector_report_csv

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dashboard_path = os.path.join(output_dir, f"dashboard_{timestamp}.png")
    report_path = os.path.join(output_dir, f"sector_report_{timestamp}.csv")
    full_data_path = os.path.join(output_dir, f"full_analysis_{timestamp}.csv")

    generate_dashboard(df, dashboard_path)
    generate_sector_report_csv(df, report_path)
    df.to_csv(full_data_path, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Articles analyzed:  {len(df)}")
    logger.info(f"  Dashboard:          {dashboard_path}")
    logger.info(f"  Sector report:      {report_path}")
    logger.info(f"  Full data:          {full_data_path}")
    logger.info("=" * 60)

    # Print sector summary to console
    print("\n📊 SECTOR SIGNAL SUMMARY")
    print("─" * 55)
    sector_summary = (
        df.groupby("primary_sector")
        .agg(
            articles=("title", "count"),
            avg_impact=("predicted_price_impact_pct", "mean"),
            sentiment=("sentiment_score", "mean"),
        )
        .sort_values("avg_impact", ascending=False)
    )
    for sector, row in sector_summary.iterrows():
        arrow = "▲" if row["avg_impact"] >= 0 else "▼"
        bar = "█" * int(abs(row["avg_impact"]) * 5)
        print(f"  {sector:<30} {arrow} {row['avg_impact']:+.2f}%  {bar}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Stock Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run (lexicon sentiment, no Gemini):
  python main.py

  # With Gemini API key for LLM analysis:
  python main.py --gemini-key YOUR_GEMINI_KEY

  # More articles, fetch full text:
  python main.py --max-articles 200 --full-text

  # Use cached news, retrain models:
  python main.py --use-cached --retrain

  # Quick demo (fewer articles):
  python main.py --demo
        """,
    )
    parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY", ""),
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--max-articles", type=int, default=80,
                        help="Max articles to fetch (default: 80)")
    parser.add_argument("--days-back", type=int, default=2,
                        help="Days of news to fetch (default: 2)")
    parser.add_argument("--full-text", action="store_true",
                        help="Fetch full article text (slower)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain ML models")
    parser.add_argument("--use-cached", action="store_true",
                        help="Use cached news data (skip scraping)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with synthetic data")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory")

    args = parser.parse_args()

    if args.demo:
        logger.info("Running DEMO mode with synthetic data...")
        _run_demo(args.output_dir)
        return

    run_pipeline(
        gemini_api_key=args.gemini_key,
        max_articles=args.max_articles,
        days_back=args.days_back,
        include_full_text=args.full_text,
        retrain=args.retrain,
        use_cached=args.use_cached,
        output_dir=args.output_dir,
    )


def _run_demo(output_dir: str):
    """Generate a demo dashboard with synthetic data."""
    import numpy as np
    from dashboard import generate_dashboard, generate_sector_report_csv
    from ml_models import EnsemblePredictor

    logger.info("Generating synthetic demo data...")
    np.random.seed(42)
    n = 120

    sectors = [
        "Information Technology", "Health Care", "Financials", "Energy",
        "Consumer Discretionary", "Communication Services", "Industrials",
        "Consumer Staples", "Utilities", "Materials", "Real Estate",
    ]
    sentiments = ["bullish", "neutral", "bearish"]
    sources = ["Reuters", "Bloomberg", "CNBC", "MarketWatch", "Yahoo Finance",
               "Seeking Alpha", "Financial Times"]

    demo_titles = [
        "NVIDIA forecasts record AI chip sales amid surging demand",
        "Fed signals rate pause as inflation shows signs of cooling",
        "Energy stocks surge after OPEC+ extends production cuts",
        "JPMorgan beats Q3 earnings, raises full-year guidance",
        "Tesla delivers record EVs but faces margin compression",
        "Healthcare sector weathers regulatory headwinds",
        "Retail sales data beats expectations, consumer resilient",
        "Industrial output slows as global trade tensions rise",
        "Utility stocks attractive as bond yields stabilize",
        "Real estate market shows green shoots amid rate plateau",
        "Semiconductor shortage eases, chip stocks rally broadly",
        "Bank of Japan policy shift rattles financial markets",
        "Amazon Web Services revenue accelerates on AI workloads",
        "Oil prices retreat on weaker-than-expected China data",
        "Biotech index surges after FDA approves key drug",
    ] * (n // 15 + 1)

    df = pd.DataFrame({
        "article_id": [f"demo_{i}" for i in range(n)],
        "title": demo_titles[:n],
        "source": np.random.choice(sources, n),
        "primary_sector": np.random.choice(sectors, n),
        "sentiment": np.random.choice(sentiments, n, p=[0.45, 0.25, 0.30]),
        "sentiment_score": np.random.uniform(-0.9, 0.9, n),
        "confidence": np.random.uniform(0.45, 0.95, n),
        "impact_magnitude": np.random.choice(["high", "medium", "low"], n, p=[0.2, 0.5, 0.3]),
        "sector_relevance": np.random.uniform(0.3, 1.0, n),
        "key_drivers": ["AI demand | cloud | earnings"] * n,
        "analysis_method": np.random.choice(["gemini", "lexicon"], n, p=[0.3, 0.7]),
        "published": pd.date_range("2025-04-28", periods=n, freq="30min"),
        "url": [f"https://example.com/article/{i}" for i in range(n)],
    })

    # Run ML predictions
    predictor = EnsemblePredictor()
    logger.info("Training models for demo...")
    predictor.train(save=False)
    df = predictor.predict_dataframe(df)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dashboard_path = os.path.join(output_dir, f"demo_dashboard_{timestamp}.png")
    report_path = os.path.join(output_dir, f"demo_sector_report_{timestamp}.csv")

    generate_dashboard(df, dashboard_path)
    generate_sector_report_csv(df, report_path)

    print(f"\n✅ Demo complete!")
    print(f"   Dashboard:     {dashboard_path}")
    print(f"   Sector report: {report_path}")


if __name__ == "__main__":
    main()
