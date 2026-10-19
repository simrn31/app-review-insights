"""Phase D - Insights & visualisation (management-summary charts)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
RED, GREEN, GREY, BLUE = "#d9534f", "#5cb85c", "#9aa0a6", "#337ab7"


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trend_line(weekly: pd.DataFrame, split, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weekly["date"], weekly["negative_share"], marker="o", color=RED, label="% negative reviews")
    ax.set_ylabel("Negative sentiment (%)")
    ax2 = ax.twinx()
    ax2.plot(weekly["date"], weekly["avg_rating"], ls="--", color=BLUE, label="Avg. star rating")
    ax2.set_ylabel("Average rating"); ax2.grid(False)
    if split is not None:
        ax.axvline(split, color="black", ls=":", lw=2)
        ax.annotate(f"Shift detected\n{pd.Timestamp(split).date()}", (split, ax.get_ylim()[1] * .92),
                    ha="right", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left")
    ax.set_title("Weekly negative sentiment vs. average rating")
    _save(fig, out)


def pain_cloud(tokens: pd.Series, out: Path):
    freq = Counter(t for toks in tokens for t in toks)
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=1200, height=600, background_color="white",
                       colormap="Reds").generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title("Word cloud of pain - 1★ and 2★ reviews")
    except ImportError:  # graceful fallback: ranked bar chart
        top = pd.DataFrame(freq.most_common(25), columns=["word", "count"])
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(data=top, y="word", x="count", color=RED, ax=ax)
        ax.set_title("Top pain words - 1★ and 2★ reviews\n(install `wordcloud` for the cloud version)")
    _save(fig, out)


def distribution_plot(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    r = df["rating"].value_counts().sort_index()
    axes[0].bar(r.index, r.values, color=[RED, RED, GREY, GREEN, GREEN])
    axes[0].set_title("Star-rating distribution"); axes[0].set_xlabel("Stars"); axes[0].set_ylabel("Reviews")
    s = df["sentiment"].value_counts().reindex(["Negative", "Neutral", "Positive"]).fillna(0)
    axes[1].bar(s.index, s.values, color=[RED, GREY, GREEN]); axes[1].set_title("Sentiment distribution")
    sat = df["satisfaction"].value_counts().reindex(
        ["Unsatisfied (1-2★)", "Neutral (3★)", "Satisfied (4-5★)"])
    axes[2].pie(sat.values, labels=sat.index, autopct="%1.0f%%", colors=[RED, GREY, GREEN], startangle=90)
    axes[2].set_title("Satisfied vs. unsatisfied users")
    _save(fig, out)


def bar_chart(data: pd.DataFrame, x: str, y: str, title: str, out: Path, color=RED):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=data, x=x, y=y, color=color, ax=ax)
    ax.set_title(title)
    _save(fig, out)


def theme_shift_chart(shift: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    m = shift.melt(id_vars="theme", value_vars=["before_%", "after_%"], var_name="period", value_name="share")
    sns.barplot(data=m, y="theme", x="share", hue="period", palette=[GREY, RED], ax=ax)
    ax.set_xlabel("% of 1-2★ complaints"); ax.set_title("What changed after the update? Complaint themes")
    _save(fig, out)
