"""
sector_mapper.py
Maps news articles to S&P 500 sectors and key tickers using keyword/entity matching.
"""

import re
import pandas as pd
from dataclasses import dataclass, field


# ── S&P 500 GICS Sectors with representative tickers ──────────────────────────

SECTORS = {
    "Information Technology": {
        "tickers": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ACN", "IBM", "INTC", "AMD"],
        "keywords": [
            "technology", "software", "semiconductor", "chip", "cloud", "AI", "artificial intelligence",
            "machine learning", "cybersecurity", "data center", "microchip", "processor", "hardware",
            "apple", "microsoft", "nvidia", "intel", "amd", "oracle", "cisco", "broadcom",
            "SaaS", "enterprise software", "digital transformation", "quantum computing",
        ],
    },
    "Health Care": {
        "tickers": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN"],
        "keywords": [
            "healthcare", "pharmaceutical", "biotech", "drug", "clinical trial", "FDA", "medicine",
            "hospital", "insurance", "Medicaid", "Medicare", "vaccine", "therapy", "oncology",
            "johnson", "pfizer", "merck", "abbvie", "lilly", "amgen", "gilead", "biogen",
            "health plan", "medical device", "genomics", "biopharma",
        ],
    },
    "Financials": {
        "tickers": ["BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "C"],
        "keywords": [
            "bank", "banking", "financial", "credit", "loan", "mortgage", "interest rate",
            "Federal Reserve", "Fed", "interest", "investment bank", "hedge fund", "insurance",
            "JPMorgan", "Goldman", "Wells Fargo", "Citigroup", "BlackRock", "Visa", "Mastercard",
            "fintech", "payment", "capital markets", "bond", "yield", "inflation", "monetary policy",
            "treasury", "debt", "equity market", "stock market",
        ],
    },
    "Consumer Discretionary": {
        "tickers": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG", "CMG"],
        "keywords": [
            "consumer", "retail", "e-commerce", "automotive", "electric vehicle", "EV", "luxury",
            "fashion", "restaurant", "hotel", "travel", "tourism", "entertainment",
            "amazon", "tesla", "home depot", "nike", "starbucks", "target", "mcdonald",
            "spending", "discretionary", "apparel", "streaming",
        ],
    },
    "Communication Services": {
        "tickers": ["META", "GOOGL", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "WBD"],
        "keywords": [
            "social media", "advertising", "streaming", "media", "telecom", "5G", "broadband",
            "Meta", "Google", "Alphabet", "Netflix", "Disney", "Comcast", "AT&T", "Verizon",
            "content", "platform", "network", "wireless", "cable",
        ],
    },
    "Industrials": {
        "tickers": ["CAT", "GE", "HON", "UPS", "BA", "RTX", "LMT", "NOC", "DE", "MMM"],
        "keywords": [
            "industrial", "manufacturing", "aerospace", "defense", "logistics", "supply chain",
            "infrastructure", "construction", "machinery", "aviation", "airline",
            "Boeing", "Caterpillar", "General Electric", "Honeywell", "UPS", "FedEx",
            "Lockheed", "Northrop", "Raytheon", "defense spending", "government contract",
        ],
    },
    "Consumer Staples": {
        "tickers": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "KMB", "GIS"],
        "keywords": [
            "consumer staples", "food", "beverage", "household", "grocery", "supermarket",
            "Procter", "Coca-Cola", "PepsiCo", "Walmart", "Costco", "Philip Morris",
            "staples", "packaged food", "personal care", "tobacco", "household products",
        ],
    },
    "Energy": {
        "tickers": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "OXY"],
        "keywords": [
            "oil", "gas", "energy", "crude", "petroleum", "refinery", "OPEC", "natural gas",
            "LNG", "pipeline", "drilling", "shale", "renewable", "solar", "wind", "fossil fuel",
            "ExxonMobil", "Chevron", "ConocoPhillips", "Schlumberger", "Halliburton",
            "energy transition", "carbon", "emissions",
        ],
    },
    "Utilities": {
        "tickers": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "ES"],
        "keywords": [
            "utility", "utilities", "electric", "electricity", "power grid", "nuclear",
            "renewable energy", "NextEra", "Duke Energy", "Southern Company", "Dominion",
            "rate", "regulatory", "grid", "transmission",
        ],
    },
    "Real Estate": {
        "tickers": ["AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "WELL", "SPG", "O", "AVB"],
        "keywords": [
            "REIT", "real estate", "property", "commercial property", "residential", "apartment",
            "office", "industrial park", "data center REIT", "housing", "mortgage REIT",
            "Prologis", "American Tower", "Crown Castle", "Equinix", "Simon Property",
            "rent", "occupancy", "cap rate",
        ],
    },
    "Materials": {
        "tickers": ["LIN", "SHW", "APD", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM"],
        "keywords": [
            "materials", "chemicals", "mining", "metals", "steel", "copper", "gold", "silver",
            "aluminum", "lithium", "commodity", "fertilizer", "agriculture input",
            "Linde", "Sherwin-Williams", "Air Products", "DuPont", "Nucor", "Freeport",
            "raw materials", "mineral",
        ],
    },
}

# Macro / cross-sector signals
MACRO_KEYWORDS = {
    "rate_hike": ["rate hike", "interest rate increase", "Fed raises", "tightening", "hawkish"],
    "rate_cut": ["rate cut", "interest rate decrease", "Fed cuts", "dovish", "easing"],
    "recession": ["recession", "contraction", "GDP decline", "economic slowdown", "downturn"],
    "inflation": ["inflation", "CPI", "PPI", "price pressure", "cost increase", "stagflation"],
    "trade_war": ["tariff", "trade war", "trade dispute", "import duty", "export ban", "sanctions"],
    "geopolitical": ["war", "conflict", "geopolitical", "invasion", "sanctions", "military"],
    "earnings_beat": ["earnings beat", "better than expected", "exceeded estimates", "record profit"],
    "earnings_miss": ["earnings miss", "below expectations", "missed estimates", "profit warning"],
}


@dataclass
class SectorSignal:
    sector: str
    tickers: list[str]
    relevance_score: float  # 0–1
    matched_keywords: list[str]
    macro_signals: list[str]


class SectorMapper:
    """
    Maps news article text to S&P 500 sectors using keyword matching.
    Returns relevance scores for each sector.
    """

    def __init__(self):
        # Pre-compile regexes for speed
        self._sector_patterns = {}
        for sector, data in SECTORS.items():
            patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in data["keywords"]]
            self._sector_patterns[sector] = patterns

        self._macro_patterns = {}
        for signal, phrases in MACRO_KEYWORDS.items():
            self._macro_patterns[signal] = [re.compile(re.escape(p), re.IGNORECASE) for p in phrases]

    def map_article(self, text: str, title: str = "") -> list[SectorSignal]:
        """
        Map a single article to relevant sectors.
        Returns sorted list of SectorSignal (highest relevance first).
        """
        combined = f"{title} {title} {text}"  # double-weight title

        signals = []
        for sector, data in SECTORS.items():
            matches = []
            for pat in self._sector_patterns[sector]:
                found = pat.findall(combined)
                matches.extend(found)

            if not matches:
                continue

            # Score = unique match count weighted by frequency, normalized
            unique_matches = list(set(m.lower() for m in matches))
            freq_weight = min(len(matches) / 10, 1.0)
            unique_weight = min(len(unique_matches) / 5, 1.0)
            relevance = (freq_weight + unique_weight) / 2

            signals.append(SectorSignal(
                sector=sector,
                tickers=data["tickers"][:5],
                relevance_score=round(relevance, 4),
                matched_keywords=unique_matches[:8],
                macro_signals=[],
            ))

        # Detect macro signals
        macro_hits = []
        for signal, patterns in self._macro_patterns.items():
            for pat in patterns:
                if pat.search(combined):
                    macro_hits.append(signal)
                    break

        # Apply macro signals to top sectors
        for sig in signals:
            sig.macro_signals = macro_hits

        signals.sort(key=lambda x: x.relevance_score, reverse=True)
        return signals[:5]  # top 5 sectors

    def map_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch-map a DataFrame of articles.
        Adds sector mapping columns.
        """
        rows = []
        for _, row in df.iterrows():
            text = str(row.get("content", row.get("summary", "")))
            title = str(row.get("title", ""))
            signals = self.map_article(text, title)

            if signals:
                primary = signals[0]
                rows.append({
                    **row.to_dict(),
                    "primary_sector": primary.sector,
                    "primary_tickers": ",".join(primary.tickers),
                    "sector_relevance": primary.relevance_score,
                    "matched_keywords": ",".join(primary.matched_keywords),
                    "macro_signals": ",".join(primary.macro_signals),
                    "all_sectors": "|".join(s.sector for s in signals),
                    "sector_scores": "|".join(str(s.relevance_score) for s in signals),
                })
            else:
                rows.append({
                    **row.to_dict(),
                    "primary_sector": "General Market",
                    "primary_tickers": "SPY",
                    "sector_relevance": 0.1,
                    "matched_keywords": "",
                    "macro_signals": "",
                    "all_sectors": "General Market",
                    "sector_scores": "0.1",
                })

        return pd.DataFrame(rows)


if __name__ == "__main__":
    mapper = SectorMapper()

    test_texts = [
        ("Fed signals rate pause amid slowing inflation", "The Federal Reserve indicated it may hold interest rates steady..."),
        ("NVIDIA smashes earnings as AI chip demand surges", "NVIDIA reported record quarterly profits driven by data center GPU demand..."),
        ("Oil prices drop on OPEC supply increase", "Crude oil futures fell 3% after OPEC+ members agreed to lift output caps..."),
        ("JPMorgan raises loan loss provisions", "JPMorgan Chase increased its credit loss reserves citing consumer credit stress..."),
    ]

    for title, text in test_texts:
        signals = mapper.map_article(text, title)
        print(f"\n📰 {title}")
        for s in signals[:3]:
            print(f"  [{s.sector}] score={s.relevance_score:.3f} | keywords={s.matched_keywords[:3]} | macro={s.macro_signals}")
