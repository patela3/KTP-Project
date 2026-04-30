"""
sentiment_analyzer.py
Uses Google Gemini API for LLM-powered sentiment + market impact analysis.
Falls back to a lexicon-based analyzer if Gemini is unavailable.
"""

import json
import time
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

ANALYSIS_PROMPT_TEMPLATE = """You are a senior equity research analyst. Analyze this financial news article and return ONLY valid JSON.

Title: {title}
Source: {source}
Content: {content}

Return this exact JSON structure (no markdown, no explanation):
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "sentiment_score": <float -1.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "impact_magnitude": "high" | "medium" | "low",
  "price_impact_pct": <estimated % price change, e.g. 2.5 or -1.8>,
  "time_horizon": "intraday" | "short_term" | "medium_term" | "long_term",
  "key_drivers": [<list of 3 key factors driving this analysis>],
  "risk_factors": [<list of up to 2 risk factors>],
  "affected_sectors": [<list of S&P 500 GICS sector names>],
  "reasoning": "<one sentence summary of your analysis>"
}}"""

# ── Lexicon fallback ───────────────────────────────────────────────────────────

BULLISH_WORDS = {
    "surge": 0.8, "soar": 0.8, "rally": 0.7, "beat": 0.7, "exceed": 0.7,
    "record": 0.6, "profit": 0.5, "growth": 0.6, "strong": 0.5, "gain": 0.6,
    "positive": 0.5, "increase": 0.4, "rise": 0.5, "up": 0.3, "boost": 0.6,
    "outperform": 0.7, "upgrade": 0.7, "bullish": 0.9, "optimistic": 0.6,
    "breakthrough": 0.7, "expand": 0.5, "accelerate": 0.6, "momentum": 0.5,
    "robust": 0.6, "recovery": 0.5, "rebound": 0.6, "innovation": 0.4,
}

BEARISH_WORDS = {
    "decline": -0.6, "drop": -0.6, "fall": -0.5, "miss": -0.7, "below": -0.4,
    "loss": -0.7, "warning": -0.7, "risk": -0.4, "concern": -0.5, "weak": -0.6,
    "cut": -0.5, "reduce": -0.4, "slowdown": -0.6, "recession": -0.8,
    "bankruptcy": -0.9, "default": -0.8, "layoff": -0.7, "downgrade": -0.7,
    "bearish": -0.9, "pessimistic": -0.6, "uncertain": -0.4, "volatile": -0.3,
    "tariff": -0.5, "sanction": -0.6, "inflation": -0.4, "shortage": -0.5,
    "crisis": -0.8, "selloff": -0.7, "plunge": -0.8,
}


@dataclass
class SentimentResult:
    article_id: str
    sentiment: str            # bullish / bearish / neutral
    sentiment_score: float    # -1 to +1
    confidence: float         # 0 to 1
    impact_magnitude: str     # high / medium / low
    price_impact_pct: float   # estimated %
    time_horizon: str
    key_drivers: list[str]
    risk_factors: list[str]
    affected_sectors: list[str]
    reasoning: str
    analysis_method: str      # "gemini" or "lexicon"


class LexiconAnalyzer:
    """Fast fallback sentiment analysis using financial lexicon."""

    def analyze(self, text: str, title: str = "") -> dict:
        combined = f"{title} {title} {text}".lower()
        words = re.findall(r"\b\w+\b", combined)

        score = 0.0
        hits_bull = []
        hits_bear = []

        for word in words:
            if word in BULLISH_WORDS:
                score += BULLISH_WORDS[word]
                hits_bull.append(word)
            elif word in BEARISH_WORDS:
                score += BEARISH_WORDS[word]
                hits_bear.append(word)

        # Normalize
        total_hits = len(hits_bull) + len(hits_bear)
        if total_hits > 0:
            score = score / max(total_hits, 5)  # dampen extremes

        score = max(-1.0, min(1.0, score))

        if score > 0.15:
            sentiment = "bullish"
        elif score < -0.15:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        confidence = min(0.3 + total_hits * 0.05, 0.75)  # max 75% confidence for lexicon

        magnitude = "high" if abs(score) > 0.6 else "medium" if abs(score) > 0.3 else "low"
        price_impact = score * 3.0  # rough mapping: max ±3%

        drivers = (hits_bull[:2] if sentiment == "bullish" else hits_bear[:2]) or ["general market sentiment"]
        drivers = list(set(drivers))[:3]

        return {
            "sentiment": sentiment,
            "sentiment_score": round(score, 4),
            "confidence": round(confidence, 3),
            "impact_magnitude": magnitude,
            "price_impact_pct": round(price_impact, 2),
            "time_horizon": "short_term",
            "key_drivers": drivers,
            "risk_factors": [],
            "affected_sectors": [],
            "reasoning": f"Lexicon analysis: {len(hits_bull)} bullish / {len(hits_bear)} bearish signals detected.",
        }


class GeminiAnalyzer:
    """Gemini-powered deep sentiment and market impact analysis."""

    def __init__(self, api_key: str, requests_per_minute: int = 15):
        self.api_key = api_key
        self.delay = 60.0 / requests_per_minute
        self._last_call = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_call = time.time()

    def analyze(self, text: str, title: str = "", source: str = "") -> Optional[dict]:
        if not self.api_key:
            return None

        self._rate_limit()

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            title=title[:200],
            source=source,
            content=text[:1500],
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 512,
                "topP": 0.8,
            },
        }

        try:
            resp = requests.post(
                f"{GEMINI_URL}?key={self.api_key}",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            # Strip markdown fences if present
            raw_text = re.sub(r"```json\s*", "", raw_text)
            raw_text = re.sub(r"```\s*", "", raw_text)

            result = json.loads(raw_text.strip())
            return result

        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Gemini API error: {e}")
            return None


class SentimentAnalyzer:
    """
    Main analyzer: tries Gemini first, falls back to lexicon.
    """

    def __init__(self, gemini_api_key: str = ""):
        self.gemini = GeminiAnalyzer(gemini_api_key) if gemini_api_key else None
        self.lexicon = LexiconAnalyzer()
        self._use_gemini = bool(gemini_api_key)

    def analyze_article(self, article_id: str, text: str, title: str = "", source: str = "") -> SentimentResult:
        result = None
        method = "lexicon"

        if self._use_gemini:
            try:
                result = self.gemini.analyze(text, title, source)
                if result:
                    method = "gemini"
            except Exception as e:
                logger.warning(f"Gemini failed, using lexicon fallback: {e}")

        if not result:
            result = self.lexicon.analyze(text, title)

        return SentimentResult(
            article_id=article_id,
            sentiment=result.get("sentiment", "neutral"),
            sentiment_score=float(result.get("sentiment_score", 0.0)),
            confidence=float(result.get("confidence", 0.5)),
            impact_magnitude=result.get("impact_magnitude", "low"),
            price_impact_pct=float(result.get("price_impact_pct", 0.0)),
            time_horizon=result.get("time_horizon", "short_term"),
            key_drivers=result.get("key_drivers", []),
            risk_factors=result.get("risk_factors", []),
            affected_sectors=result.get("affected_sectors", []),
            reasoning=result.get("reasoning", ""),
            analysis_method=method,
        )

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch analyze a DataFrame of articles.
        Adds sentiment columns.
        """
        results = []
        total = len(df)
        logger.info(f"Analyzing sentiment for {total} articles...")

        for i, (_, row) in enumerate(df.iterrows()):
            if i % 10 == 0:
                logger.info(f"  Sentiment progress: {i}/{total}")

            text = str(row.get("content", row.get("summary", "")))
            title = str(row.get("title", ""))
            source = str(row.get("source", ""))
            article_id = str(row.get("article_id", str(i)))

            result = self.analyze_article(article_id, text, title, source)

            results.append({
                "article_id": article_id,
                "sentiment": result.sentiment,
                "sentiment_score": result.sentiment_score,
                "confidence": result.confidence,
                "impact_magnitude": result.impact_magnitude,
                "price_impact_pct": result.price_impact_pct,
                "time_horizon": result.time_horizon,
                "key_drivers": " | ".join(result.key_drivers),
                "risk_factors": " | ".join(result.risk_factors),
                "reasoning": result.reasoning,
                "analysis_method": result.analysis_method,
            })

        sentiment_df = pd.DataFrame(results)
        merged = df.merge(sentiment_df, on="article_id", how="left")
        logger.info("Sentiment analysis complete.")
        return merged


if __name__ == "__main__":
    analyzer = SentimentAnalyzer(gemini_api_key=GEMINI_API_KEY)

    test_cases = [
        ("art1", "NVIDIA reports record $22B quarterly revenue, AI chip demand shows no signs of slowing",
         "NVIDIA Q3 earnings crushed estimates by 20% amid explosive data center demand...", "Reuters"),
        ("art2", "Federal Reserve signals prolonged high rates, recession fears intensify",
         "Fed Chair Powell stated rates will remain elevated as inflation remains stubborn...", "Bloomberg"),
        ("art3", "Oil prices stabilize amid mixed inventory data",
         "Crude futures traded flat after EIA reported a modest inventory build...", "MarketWatch"),
    ]

    for article_id, title, text, source in test_cases:
        result = analyzer.analyze_article(article_id, text, title, source)
        print(f"\n📰 {title[:60]}")
        print(f"  Sentiment: {result.sentiment} ({result.sentiment_score:+.3f}) | Method: {result.analysis_method}")
        print(f"  Impact: {result.impact_magnitude} | Price est: {result.price_impact_pct:+.1f}%")
        print(f"  Drivers: {result.key_drivers}")
        print(f"  Reasoning: {result.reasoning}")
