"""
dashboard.py
Interactive visualization dashboard using Seaborn, Matplotlib, and Pandas.
Generates a multi-panel HTML + static figure showing sector signals,
sentiment distributions, and news-driven price predictions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── Visual Theme ───────────────────────────────────────────────────────────────

PALETTE = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "bullish":   "#3fb950",
    "bearish":   "#f85149",
    "neutral":   "#8b949e",
    "accent":    "#58a6ff",
    "warning":   "#e3b341",
    "text":      "#e6edf3",
    "subtext":   "#8b949e",
}

SECTOR_COLORS = {
    "Information Technology": "#58a6ff",
    "Health Care":             "#3fb950",
    "Financials":              "#e3b341",
    "Consumer Discretionary":  "#ff7b72",
    "Communication Services":  "#bc8cff",
    "Industrials":             "#79c0ff",
    "Consumer Staples":        "#56d364",
    "Energy":                  "#f78166",
    "Utilities":               "#ffa657",
    "Real Estate":             "#d2a8ff",
    "Materials":               "#7ee787",
    "General Market":          "#8b949e",
}


def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":    PALETTE["bg"],
        "axes.facecolor":      PALETTE["panel"],
        "axes.edgecolor":      PALETTE["border"],
        "axes.labelcolor":     PALETTE["text"],
        "axes.titlecolor":     PALETTE["text"],
        "xtick.color":         PALETTE["subtext"],
        "ytick.color":         PALETTE["subtext"],
        "text.color":          PALETTE["text"],
        "grid.color":          PALETTE["border"],
        "grid.alpha":          0.5,
        "legend.facecolor":    PALETTE["panel"],
        "legend.edgecolor":    PALETTE["border"],
        "legend.labelcolor":   PALETTE["text"],
        "font.family":         "DejaVu Sans",
        "font.size":           10,
    })


def _sentiment_color(score: float) -> str:
    if score > 0.1:
        return PALETTE["bullish"]
    elif score < -0.1:
        return PALETTE["bearish"]
    return PALETTE["neutral"]


def build_sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level data to sector-level summaries."""
    required = ["primary_sector", "sentiment_score", "predicted_price_impact_pct", "confidence"]
    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    summary = df.groupby("primary_sector").agg(
        article_count=("title", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        avg_predicted_impact=("predicted_price_impact_pct", "mean"),
        avg_confidence=("confidence", "mean"),
        max_impact=("predicted_price_impact_pct", lambda x: x.abs().max()),
        bullish_count=("sentiment", lambda x: (x == "bullish").sum()),
        bearish_count=("sentiment", lambda x: (x == "bearish").sum()),
        neutral_count=("sentiment", lambda x: (x == "neutral").sum()),
    ).reset_index()

    summary["bullish_ratio"] = summary["bullish_count"] / summary["article_count"]
    summary["net_sentiment"] = (summary["bullish_count"] - summary["bearish_count"]) / summary["article_count"]
    summary = summary.sort_values("avg_predicted_impact", key=abs, ascending=False)
    return summary


def plot_sector_impact_bar(ax: plt.Axes, summary: pd.DataFrame):
    """Horizontal bar chart: predicted price impact by sector."""
    sectors = summary["primary_sector"].tolist()
    impacts = summary["avg_predicted_impact"].tolist()
    confidences = summary["avg_confidence"].tolist()

    colors = [PALETTE["bullish"] if v >= 0 else PALETTE["bearish"] for v in impacts]
    alpha_vals = [0.5 + 0.5 * min(c, 1.0) for c in confidences]

    bars = ax.barh(
        sectors, impacts,
        color=colors,
        alpha=0.85,
        height=0.65,
        edgecolor=PALETTE["border"],
        linewidth=0.8,
    )

    # Confidence error bars
    ci_half = [(1 - c) * 1.5 for c in confidences]
    ax.errorbar(
        impacts, sectors,
        xerr=ci_half,
        fmt="none",
        color=PALETTE["subtext"],
        capsize=3,
        linewidth=1.2,
    )

    # Value labels
    for bar, val, conf in zip(bars, impacts, confidences):
        label = f"{val:+.2f}%"
        x_pos = val + (0.05 if val >= 0 else -0.05)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha,
                fontsize=8, color=PALETTE["text"], fontweight="bold")

    ax.axvline(0, color=PALETTE["subtext"], linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Avg Predicted Price Impact (%)", fontsize=10, labelpad=8)
    ax.set_title("📈 Predicted Price Impact by Sector", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_sentiment_heatmap(ax: plt.Axes, summary: pd.DataFrame):
    """Heatmap: sentiment breakdown per sector."""
    heat_data = summary.set_index("primary_sector")[["bullish_ratio", "avg_confidence", "avg_sentiment"]].copy()
    heat_data.columns = ["Bullish Ratio", "Confidence", "Sentiment Score"]

    cmap = LinearSegmentedColormap.from_list(
        "sentiment", [PALETTE["bearish"], PALETTE["neutral"], PALETTE["bullish"]], N=256
    )

    sns.heatmap(
        heat_data, ax=ax,
        cmap=cmap, center=0,
        annot=True, fmt=".2f",
        linewidths=0.5, linecolor=PALETTE["border"],
        cbar_kws={"shrink": 0.8, "label": "Score"},
        annot_kws={"size": 8, "color": PALETTE["text"]},
    )

    ax.set_title("🌡️ Sentiment Heatmap by Sector", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_ylabel("")
    ax.set_xlabel("")


def plot_news_volume_timeline(ax: plt.Axes, df: pd.DataFrame):
    """Timeline of article volume by sentiment over time."""
    if "published" not in df.columns or df["published"].isna().all():
        ax.text(0.5, 0.5, "No timestamp data available",
                ha="center", va="center", transform=ax.transAxes, color=PALETTE["subtext"])
        ax.set_title("📰 News Volume Timeline", fontsize=12, fontweight="bold")
        return

    df = df.copy()
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"])

    if len(df) == 0:
        ax.text(0.5, 0.5, "No valid dates", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["subtext"])
        return

    df["hour"] = df["published"].dt.floor("h")

    for sentiment, color in [("bullish", PALETTE["bullish"]),
                              ("bearish", PALETTE["bearish"]),
                              ("neutral", PALETTE["neutral"])]:
        grp = df[df["sentiment"] == sentiment].groupby("hour").size().reset_index(name="count")
        if not grp.empty:
            ax.fill_between(grp["hour"], grp["count"], alpha=0.4, color=color)
            ax.plot(grp["hour"], grp["count"], color=color, linewidth=1.5, label=sentiment.capitalize())

    ax.set_title("📰 News Volume by Sentiment (Time)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_ylabel("Article Count", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_top_news_drivers(ax: plt.Axes, df: pd.DataFrame, n: int = 10):
    """Lollipop chart of top news articles driving sector signals."""
    df = df.copy()
    df["abs_impact"] = df.get("predicted_price_impact_pct", pd.Series(0.0, index=df.index)).abs()
    top = df.nlargest(n, "abs_impact")

    labels = [t[:45] + "…" if len(str(t)) > 45 else str(t) for t in top["title"].tolist()]
    impacts = top.get("predicted_price_impact_pct", pd.Series(0.0)).tolist()
    sources = top.get("source", pd.Series("Unknown")).tolist()
    sectors = top.get("primary_sector", pd.Series("General Market")).tolist()

    y_pos = range(len(labels))
    colors = [PALETTE["bullish"] if v >= 0 else PALETTE["bearish"] for v in impacts]

    ax.hlines(y_pos, 0, impacts, colors=colors, linewidth=1.5, alpha=0.7)
    ax.scatter(impacts, y_pos, color=colors, s=60, zorder=5)
    ax.axvline(0, color=PALETTE["subtext"], linewidth=0.8, linestyle="--")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=7.5)

    # Source annotations
    for i, (imp, src, sec) in enumerate(zip(impacts, sources, sectors)):
        x_off = imp + (0.08 if imp >= 0 else -0.08)
        ha = "left" if imp >= 0 else "right"
        ax.text(x_off, i, f"{src} · {sec}",
                fontsize=6.5, va="center", ha=ha, color=PALETTE["subtext"])

    ax.set_title("🔍 Top News Drivers by Signal Strength", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Price Impact (%)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_source_distribution(ax: plt.Axes, df: pd.DataFrame):
    """Donut chart of articles by source."""
    source_counts = df["source"].value_counts().head(8)

    colors = [
        PALETTE["accent"], PALETTE["bullish"], PALETTE["warning"],
        PALETTE["bearish"], "#bc8cff", "#79c0ff", "#ffa657", PALETTE["subtext"],
    ]

    wedges, texts, autotexts = ax.pie(
        source_counts.values,
        labels=source_counts.index,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors[:len(source_counts)],
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2},
        pctdistance=0.75,
    )

    for text in texts:
        text.set_fontsize(7.5)
        text.set_color(PALETTE["text"])
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color(PALETTE["bg"])
        at.set_fontweight("bold")

    # Donut hole
    circle = plt.Circle((0, 0), 0.5, color=PALETTE["panel"])
    ax.add_patch(circle)
    ax.text(0, 0, f"{len(df)}\narticles", ha="center", va="center",
            fontsize=9, color=PALETTE["text"], fontweight="bold")

    ax.set_title("📡 Articles by Source", fontsize=12, fontweight="bold", pad=10)


def plot_model_confidence(ax: plt.Axes, df: pd.DataFrame):
    """Scatter: sentiment score vs predicted impact, sized by confidence."""
    if "predicted_price_impact_pct" not in df.columns:
        df["predicted_price_impact_pct"] = 0.0

    colors = [
        SECTOR_COLORS.get(s, PALETTE["neutral"])
        for s in df.get("primary_sector", pd.Series("General Market"))
    ]
    sizes = [max(20, 200 * float(c)) for c in df.get("confidence", pd.Series(0.5))]

    sc = ax.scatter(
        df.get("sentiment_score", pd.Series(0.0)),
        df["predicted_price_impact_pct"],
        c=colors, s=sizes,
        alpha=0.65, edgecolors=PALETTE["border"], linewidth=0.5,
    )

    ax.axhline(0, color=PALETTE["subtext"], linewidth=0.8, linestyle="--")
    ax.axvline(0, color=PALETTE["subtext"], linewidth=0.8, linestyle="--")

    # Trend line
    try:
        x = df.get("sentiment_score", pd.Series(0.0)).fillna(0).values
        y = df["predicted_price_impact_pct"].fillna(0).values
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xline = np.linspace(x.min(), x.max(), 100)
            ax.plot(xline, p(xline), color=PALETTE["accent"], linewidth=1.5, linestyle=":", alpha=0.8)
    except Exception:
        pass

    ax.set_xlabel("Sentiment Score", fontsize=9)
    ax.set_ylabel("Predicted Impact (%)", fontsize=9)
    ax.set_title("🎯 Sentiment Score vs Predicted Impact", fontsize=12, fontweight="bold", pad=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Sector legend (top 6)
    legend_items = list(SECTOR_COLORS.items())[:6]
    patches = [mpatches.Patch(color=c, label=s.replace(" ", "\n")) for s, c in legend_items]
    ax.legend(handles=patches, loc="upper left", fontsize=6, ncol=2,
              framealpha=0.7, labelspacing=0.3)


def generate_dashboard(df: pd.DataFrame, output_path: str = "outputs/dashboard.png") -> str:
    """
    Generate the full analysis dashboard.
    Returns path to saved figure.
    """
    _apply_dark_theme()

    summary = build_sector_summary(df)

    fig = plt.figure(figsize=(22, 26), facecolor=PALETTE["bg"])

    # ── Title Banner ──
    fig.text(
        0.5, 0.975,
        "⚡ SEMANTIC STOCK SIGNAL DASHBOARD",
        ha="center", va="top", fontsize=20, fontweight="bold",
        color=PALETTE["text"], family="monospace",
    )
    fig.text(
        0.5, 0.961,
        f"S&P 500 Sector Intelligence  ·  {len(df)} articles analyzed  ·  Generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        ha="center", va="top", fontsize=11, color=PALETTE["subtext"],
    )

    # ── Layout ──
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        top=0.945, bottom=0.04,
        left=0.06, right=0.97,
        hspace=0.42, wspace=0.35,
    )

    # Row 0
    ax_bar = fig.add_subplot(gs[0, :2])
    ax_donut = fig.add_subplot(gs[0, 2])

    # Row 1
    ax_heatmap = fig.add_subplot(gs[1, :2])
    ax_scatter = fig.add_subplot(gs[1, 2])

    # Row 2
    ax_lollipop = fig.add_subplot(gs[2, :2])
    ax_timeline = fig.add_subplot(gs[2, 2])

    # ── Plots ──
    plot_sector_impact_bar(ax_bar, summary)
    plot_source_distribution(ax_donut, df)
    plot_sentiment_heatmap(ax_heatmap, summary)
    plot_model_confidence(ax_scatter, df)
    plot_top_news_drivers(ax_lollipop, df)
    plot_news_volume_timeline(ax_timeline, df)

    # ── Footer stats ──
    bullish = (df.get("sentiment", pd.Series("neutral")) == "bullish").sum()
    bearish = (df.get("sentiment", pd.Series("neutral")) == "bearish").sum()
    neutral = len(df) - bullish - bearish
    avg_impact = df.get("predicted_price_impact_pct", pd.Series(0.0)).mean()

    stats = [
        f"🟢 Bullish: {bullish}",
        f"🔴 Bearish: {bearish}",
        f"⚪ Neutral: {neutral}",
        f"📊 Avg Impact: {avg_impact:+.2f}%",
        f"📡 Sources: {df['source'].nunique()}",
        f"🏢 Sectors: {df.get('primary_sector', pd.Series('–')).nunique()}",
    ]
    fig.text(
        0.5, 0.012,
        "   ·   ".join(stats),
        ha="center", va="bottom", fontsize=9.5, color=PALETTE["subtext"],
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close(fig)
    print(f"✅ Dashboard saved → {output_path}")
    return output_path


def generate_sector_report_csv(df: pd.DataFrame, output_path: str = "outputs/sector_report.csv") -> str:
    """Generate a CSV report with sector-level aggregated stats."""
    summary = build_sector_summary(df)

    # Add top article per sector
    if "predicted_price_impact_pct" in df.columns:
        top_articles = (
            df.sort_values("predicted_price_impact_pct", key=abs, ascending=False)
            .groupby("primary_sector")
            .first()[["title", "source", "url"]]
            .reset_index()
        )
        summary = summary.merge(top_articles, on="primary_sector", how="left")
        summary = summary.rename(columns={
            "title": "top_article",
            "source": "top_source",
            "url": "top_url",
        })

    summary.to_csv(output_path, index=False)
    print(f"✅ Sector report saved → {output_path}")
    return output_path


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n = 80

    sectors = [
        "Information Technology", "Health Care", "Financials", "Energy",
        "Consumer Discretionary", "Communication Services", "Industrials",
        "Consumer Staples", "Utilities", "Materials",
    ]
    sentiments = ["bullish", "neutral", "bearish"]
    sources = ["Reuters", "Bloomberg", "CNBC", "MarketWatch", "Yahoo Finance"]

    demo_df = pd.DataFrame({
        "article_id": [f"art{i}" for i in range(n)],
        "title": [f"Sample article about sector news #{i}" for i in range(n)],
        "source": np.random.choice(sources, n),
        "primary_sector": np.random.choice(sectors, n),
        "sentiment": np.random.choice(sentiments, n, p=[0.4, 0.3, 0.3]),
        "sentiment_score": np.random.uniform(-1, 1, n),
        "confidence": np.random.uniform(0.4, 0.95, n),
        "predicted_price_impact_pct": np.random.uniform(-4, 4, n),
        "sector_relevance": np.random.uniform(0.3, 1.0, n),
        "published": pd.date_range("2025-01-01", periods=n, freq="2h"),
    })

    generate_dashboard(demo_df)
    generate_sector_report_csv(demo_df)
