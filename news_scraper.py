"""
news_scraper.py
Scrapes global financial news from multiple sources using BeautifulSoup.
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import random
from dataclasses import dataclass, field
from typing import Optional
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    title: str
    summary: str
    url: str
    source: str
    published: datetime
    full_text: str = ""
    article_id: str = field(init=False)

    def __post_init__(self):
        self.article_id = hashlib.md5(self.url.encode()).hexdigest()[:12]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

RSS_FEEDS = {
    "Reuters": "https://feeds.reuters.com/reuters/businessNews",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Yahoo Finance": "https://finance.yahoo.com/rss/topfinstories",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
    "Investing.com": "https://www.investing.com/rss/news.rss",
    "Financial Times": "https://www.ft.com/rss/home",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
}

SCRAPE_TARGETS = {
    "Yahoo Finance Top Stories": {
        "url": "https://finance.yahoo.com/topic/stock-market-news/",
        "article_selector": "li.js-stream-content",
        "title_selector": "h3",
        "summary_selector": "p",
        "link_selector": "a",
    },
    "MarketWatch Latest": {
        "url": "https://www.marketwatch.com/latest-news",
        "article_selector": "div.article__content",
        "title_selector": "h3.article__headline",
        "summary_selector": "p.article__summary",
        "link_selector": "a.article__headline",
    },
}


class FinancialNewsScraper:
    """
    Multi-source financial news scraper.
    Combines RSS feeds and direct HTML scraping.
    """

    def __init__(self, max_articles_per_source: int = 15, days_back: int = 2):
        self.max_articles = max_articles_per_source
        self.cutoff = datetime.utcnow() - timedelta(days=days_back)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _safe_get(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        try:
            time.sleep(random.uniform(0.5, 1.5))  # polite delay
            resp = self.session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _parse_date(self, date_str: str) -> datetime:
        """Try multiple date formats."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%a, %d %b %Y %H:%M:%S GMT",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(str(date_str)[:50], fmt)
                return dt.replace(tzinfo=None)
            except (ValueError, TypeError):
                continue
        return datetime.utcnow()

    def scrape_rss_feeds(self) -> list[NewsArticle]:
        """Scrape all configured RSS feeds."""
        articles = []
        for source_name, feed_url in RSS_FEEDS.items():
            logger.info(f"Scraping RSS: {source_name}")
            try:
                feed = feedparser.parse(feed_url)
                count = 0
                for entry in feed.entries:
                    if count >= self.max_articles:
                        break
                    try:
                        title = entry.get("title", "").strip()
                        summary = entry.get("summary", entry.get("description", "")).strip()
                        url = entry.get("link", "")
                        published_raw = entry.get("published", entry.get("updated", ""))
                        published = self._parse_date(published_raw)

                        if not title or not url:
                            continue

                        # Clean HTML from summary
                        if summary:
                            soup = BeautifulSoup(summary, "html.parser")
                            summary = soup.get_text(separator=" ").strip()

                        article = NewsArticle(
                            title=title,
                            summary=summary[:500],
                            url=url,
                            source=source_name,
                            published=published,
                        )
                        articles.append(article)
                        count += 1
                    except Exception as e:
                        logger.debug(f"Skipping entry in {source_name}: {e}")
                        continue
                logger.info(f"  → Got {count} articles from {source_name}")
            except Exception as e:
                logger.warning(f"RSS feed failed for {source_name}: {e}")
        return articles

    def scrape_article_text(self, article: NewsArticle) -> str:
        """Fetch and extract full article body text."""
        resp = self._safe_get(article.url)
        if not resp:
            return article.summary

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Try common article body selectors
        selectors = [
            "article",
            '[class*="article-body"]',
            '[class*="story-body"]',
            '[class*="post-content"]',
            '[class*="entry-content"]',
            "main",
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if len(text) > 200:
                    return text[:2000]

        # Fallback: all paragraph text
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)
        return text[:2000] if text else article.summary

    def scrape_direct_html(self) -> list[NewsArticle]:
        """Scrape specific financial news pages directly."""
        articles = []
        for source_name, config in SCRAPE_TARGETS.items():
            logger.info(f"Direct scraping: {source_name}")
            resp = self._safe_get(config["url"])
            if not resp:
                continue
            try:
                soup = BeautifulSoup(resp.text, "html.parser")
                containers = soup.select(config["article_selector"])
                count = 0
                for container in containers[:self.max_articles]:
                    try:
                        title_el = container.select_one(config["title_selector"])
                        summary_el = container.select_one(config["summary_selector"])
                        link_el = container.select_one(config["link_selector"])

                        if not title_el:
                            continue
                        title = title_el.get_text(strip=True)
                        summary = summary_el.get_text(strip=True) if summary_el else ""
                        href = link_el.get("href", "") if link_el else ""
                        if href and not href.startswith("http"):
                            href = "https://" + source_name.split()[0].lower() + ".com" + href

                        article = NewsArticle(
                            title=title,
                            summary=summary[:500],
                            url=href or config["url"],
                            source=source_name,
                            published=datetime.utcnow(),
                        )
                        articles.append(article)
                        count += 1
                    except Exception:
                        continue
                logger.info(f"  → Got {count} articles from {source_name}")
            except Exception as e:
                logger.warning(f"Direct scrape failed for {source_name}: {e}")
        return articles

    def fetch_all(self, include_full_text: bool = False) -> pd.DataFrame:
        """
        Fetch all articles from all sources.
        Returns a cleaned DataFrame.
        """
        logger.info("=== Starting news scrape ===")
        articles = []
        articles.extend(self.scrape_rss_feeds())
        articles.extend(self.scrape_direct_html())

        # Deduplicate by article_id
        seen = set()
        unique = []
        for a in articles:
            if a.article_id not in seen:
                seen.add(a.article_id)
                unique.append(a)

        logger.info(f"Total unique articles: {len(unique)}")

        if include_full_text:
            logger.info("Fetching full article text (this may take a while)...")
            for i, article in enumerate(unique):
                if not article.full_text:
                    article.full_text = self.scrape_article_text(article)
                if i % 10 == 0:
                    logger.info(f"  Fetched text for {i}/{len(unique)} articles")

        records = []
        for a in unique:
            records.append({
                "article_id": a.article_id,
                "title": a.title,
                "summary": a.summary,
                "full_text": a.full_text,
                "url": a.url,
                "source": a.source,
                "published": a.published,
                "content": (a.full_text if a.full_text else a.summary) or a.title,
            })

        df = pd.DataFrame(records)
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        df = df.dropna(subset=["title"])
        df = df[df["title"].str.len() > 10]
        df = df.sort_values("published", ascending=False).reset_index(drop=True)

        logger.info(f"=== Scrape complete: {len(df)} articles ===")
        return df


if __name__ == "__main__":
    scraper = FinancialNewsScraper(max_articles_per_source=5)
    df = scraper.fetch_all(include_full_text=False)
    print(df[["title", "source", "published"]].head(20).to_string())
    df.to_csv("data/raw_news.csv", index=False)
    print(f"\nSaved {len(df)} articles to data/raw_news.csv")
