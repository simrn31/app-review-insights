"""Shared orchestration used by both `main.py` (CLI) and `app.py` (Streamlit dashboard)."""
from __future__ import annotations

import pandas as pd

from src import analyze, preprocess


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Phases B + C (row-level): clean text, score sentiment, tag complaint themes."""
    return analyze.tag_themes(analyze.add_sentiment(preprocess.preprocess_dataframe(raw)))


def _safe(fn, empty_cols, *args, **kw):
    try:
        return fn(*args, **kw)
    except ValueError:  # e.g. empty vocabulary on a tiny filtered slice
        return pd.DataFrame(columns=empty_cols)


def summarize(df: pd.DataFrame, update_date=None) -> dict:
    """Phase C (aggregate): everything the charts and the management summary need."""
    comp = df[df["rating"] <= 2]
    res = {"df": df, "complaints": comp, "engine": analyze.engine_name()}
    res["bigrams"] = _safe(analyze.top_ngrams, ["ngram", "count"], comp["text_processed"], n=2, top=20)
    try:
        res["gap_low"], res["gap_high"] = analyze.rating_word_analysis(df)
    except ValueError:
        empty = pd.DataFrame(columns=["term", "tfidf_1_2_star", "tfidf_4_5_star", "gap"])
        res["gap_low"], res["gap_high"] = empty, empty
    res["corr"] = analyze.rating_sentiment_correlation(df) if len(df) > 2 else float("nan")
    res["themes"] = analyze.theme_summary(df, df["rating"] <= 2)
    res["weekly"] = analyze.weekly_trend(df)
    split = pd.Timestamp(update_date) if update_date else analyze.detect_changepoint(res["weekly"])
    res["split"] = split
    if split is not None and (df["date"] < split).any() and (df["date"] >= split).any():
        res["ba"] = analyze.before_after(df, split)
        res["shift"] = analyze.theme_shift(df, split)
    else:
        res["split"], res["ba"] = None, None
        res["shift"] = pd.DataFrame(columns=["theme", "before_%", "after_%", "change_pp"])
    return res


def write_summary(res: dict, source: str) -> str:
    df, ba, shift = res["df"], res["ba"], res["shift"]
    lines = [
        "# Management Summary - App Review Insights", "",
        f"*Data source: {source}. Reviews analysed: {len(df):,} "
        f"({df['date'].min().date()} to {df['date'].max().date()}). Sentiment engine: {res['engine']}.*", "",
        "## Headline",
        f"- Average rating: **{df['rating'].mean():.2f}**; "
        f"{(df['rating'] <= 2).mean():.0%} of users are unsatisfied (1-2★), "
        f"{(df['rating'] >= 4).mean():.0%} satisfied (4-5★).",
        f"- Rating and text sentiment agree (Spearman ρ = **{res['corr']:.2f}**).", "",
        "## Did something change?",
    ]
    if ba:
        lines += [
            f"- Biggest shift in negative sentiment detected around **{ba['split_date']}**: "
            f"negative reviews went from **{ba['neg_share_before_%']}%** to **{ba['neg_share_after_%']}%**; "
            f"average rating from **{ba['avg_rating_before']}** to **{ba['avg_rating_after']}**.",
            "- Compare the date with your release calendar to confirm the cause."]
    else:
        lines.append("- Not enough weekly data in this selection to detect a shift.")
    lines += ["", "## Top complaint themes (1-2★ reviews)"]
    lines += [f"- **{r['theme']}** - {r['share_of_complaints_%']}% of complaints"
              for _, r in res["themes"].head(5).iterrows()]
    if len(shift):
        lines += ["", "## Themes that grew most after the shift"]
        lines += [f"- **{r['theme']}**: {r['before_%']}% → {r['after_%']}% ({r['change_pp']:+} pp)"
                  for _, r in shift.head(3).iterrows()]
    if len(res["bigrams"]):
        lines += ["", "## Most frequent complaint phrases (bigrams)"]
        lines += [f"- \"{r.ngram}\" ({r['count']}×)" for _, r in res["bigrams"].head(8).iterrows()]
    if len(res["gap_low"]):
        lines += ["", "## Words most associated with 1-2★ (vs 4-5★) reviews",
                  ", ".join(res["gap_low"]["term"].head(10))]
    lines += ["", "## Recommended actions",
              "1. Triage the fastest-growing theme first - it is most likely tied to the latest release.",
              "2. Add automated tests/monitoring for the top bug-type phrases above.",
              "3. Re-run this pipeline weekly to track the trend.", ""]
    return "\n".join(lines)
