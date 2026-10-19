"""Phase C - Sentiment, N-gram, TF-IDF and rating analysis, plus complaint theme tagging."""
from __future__ import annotations

import importlib
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------- sentiment
POS_WORDS = set("""love great good excellent amazing awesome perfect best smooth fast easy helpful nice
fantastic useful intuitive reliable happy clean simple works worth recommend wonderful enjoy enjoyable
okay fine decent""".split()) - {"okay", "fine", "decent"}
NEG_WORDS = set("""bad worst terrible awful horrible useless crash crashes crashing bug buggy slow lag laggy
freeze freezes frozen fail failed fails failure error broken expensive overpriced confusing annoying hate
waste disappointed disappointing frustrating unable issue problem stuck glitch scam refund uninstall
unusable poor worse drain cluttered ridiculous ruined unstable""".split())
NEGATORS = {"not", "no", "never", "cannot"}


def _lexicon_polarity(text: str) -> float:
    """Fallback scorer in [-1, 1] with simple negation handling."""
    toks, score, hits = text.split(), 0.0, 0
    for i, t in enumerate(toks):
        val = 1 if t in POS_WORDS else -1 if t in NEG_WORDS else 0
        if val:
            if any(p in NEGATORS for p in toks[max(0, i - 2):i]):
                val = -val
            score += val
            hits += 1
    return score / (hits + 1) if hits else 0.0


# Domain terms VADER's general-purpose lexicon may under-score (scale roughly -4..+4)
DOMAIN_LEXICON = {
    "crash": -2.2, "crashes": -2.2, "crashing": -2.2, "crashed": -2.2, "buggy": -2.0, "glitch": -1.6,
    "glitchy": -1.8, "laggy": -1.8, "lag": -1.4, "freezes": -1.8, "freezing": -1.8, "overpriced": -2.2,
    "paywall": -1.6, "unusable": -2.6, "uninstall": -1.8, "uninstalling": -1.8, "refund": -1.2,
    "cluttered": -1.5, "intuitive": 1.8,
}


@lru_cache(maxsize=1)
def _scorer():
    """Return (engine_name, needs_raw_text, fn). Preference: VADER > TextBlob > built-in lexicon."""
    for modpath in ("vaderSentiment.vaderSentiment", "nltk.sentiment.vader"):
        try:
            sia = importlib.import_module(modpath).SentimentIntensityAnalyzer()
            sia.lexicon.update(DOMAIN_LEXICON)
            sia.polarity_scores("test")
            # VADER uses capitalisation, punctuation and emojis as signals -> feed it the RAW review
            return "VADER", True, lambda t: sia.polarity_scores(t)["compound"]
        except Exception:
            continue
    try:
        from textblob import TextBlob
        TextBlob("test").sentiment
        return "TextBlob", False, lambda t: TextBlob(t).sentiment.polarity
    except Exception:
        return "Built-in lexicon", False, _lexicon_polarity


def engine_name() -> str:
    return _scorer()[0]


def add_sentiment(df: pd.DataFrame, pos_t: float = 0.05, neg_t: float = -0.05) -> pd.DataFrame:
    """Score every review in [-1, 1] and label Positive / Neutral / Negative (+/-0.05 cut-offs)."""
    df = df.copy()
    _, raw, fn = _scorer()
    src = df["review_text"] if (raw and "review_text" in df.columns) else df["text_clean"]
    df["polarity"] = src.map(fn)
    df["sentiment"] = np.select([df["polarity"] > pos_t, df["polarity"] < neg_t],
                                ["Positive", "Negative"], default="Neutral")
    df["satisfaction"] = pd.cut(df["rating"], [0, 2, 3, 5],
                                labels=["Unsatisfied (1-2★)", "Neutral (3★)", "Satisfied (4-5★)"])
    return df


# ------------------------------------------------------------------ n-grams
def top_ngrams(texts: pd.Series, n: int = 2, top: int = 20, min_df: int = 2) -> pd.DataFrame:
    vec = CountVectorizer(ngram_range=(n, n), min_df=min_df, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    out = pd.DataFrame({"ngram": vec.get_feature_names_out(), "count": counts})
    return out.sort_values("count", ascending=False).head(top).reset_index(drop=True)


# ------------------------------------------------- rating <-> word analysis
def rating_word_analysis(df: pd.DataFrame, top: int = 15):
    """Mean TF-IDF per group; 'distinctive' = score gap between 1-2★ and 4-5★ reviews."""
    vec = TfidfVectorizer(min_df=5, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(df["text_processed"])
    terms = np.array(vec.get_feature_names_out())
    low, high = (df["rating"] <= 2).values, (df["rating"] >= 4).values
    m_low = np.asarray(X[low].mean(axis=0)).ravel()
    m_high = np.asarray(X[high].mean(axis=0)).ravel()
    t = pd.DataFrame({"term": terms, "tfidf_1_2_star": m_low, "tfidf_4_5_star": m_high})
    t["gap"] = t["tfidf_1_2_star"] - t["tfidf_4_5_star"]
    return (t.sort_values("gap", ascending=False).head(top).reset_index(drop=True),
            t.sort_values("gap").head(top).reset_index(drop=True))


def rating_sentiment_correlation(df: pd.DataFrame) -> float:
    return float(df["rating"].corr(df["polarity"], method="spearman"))


# ----------------------------------------------------- actionable complaints
THEMES = {
    "Login & Authentication": r"\b(?:login|log in|sign in|signin|password|authentication|otp|verification|account locked)\b",
    "Crashes & Bugs": r"\b(?:crash\w*|freez\w*|bug\w*|glitch\w*|error\w*|force close|not responding|stuck)\b",
    "Pricing & Subscription": r"\b(?:subscription|price\w*|expensive|paywall|overpriced|charged|refund|premium|billing)\b",
    "UI / UX": r"\b(?:interface|confusing|layout|design|navigation|redesign|buttons?|cluttered|hard to use)\b",
    "Performance & Battery": r"\b(?:slow|lag\w*|loading|battery|drain|takes forever)\b",
    "Ads": r"\b(?:ads?|adverts?|popups?)\b",
    "Notifications & Sync": r"\b(?:notifications?|sync\w*)\b",
}


def tag_themes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, pat in THEMES.items():
        df[f"theme::{name}"] = df["text_clean"].str.contains(pat, regex=True)
    cols = [c for c in df.columns if c.startswith("theme::")]
    df["theme_other"] = ~df[cols].any(axis=1)
    return df


def theme_summary(df: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    sub = df if mask is None else df[mask]
    cols = [c for c in df.columns if c.startswith("theme::")]
    s = sub[cols].sum().rename(lambda c: c.replace("theme::", "")).sort_values(ascending=False)
    out = s.rename("complaints").to_frame()
    out["share_of_complaints_%"] = (out["complaints"] / max(len(sub), 1) * 100).round(1)
    return out.reset_index(names="theme")


# ------------------------------------------------------------------- trends
def weekly_trend(df: pd.DataFrame, min_reviews: int = 20) -> pd.DataFrame:
    w = (df.set_index("date").resample("W-MON", label="left", closed="left")
           .agg(reviews=("rating", "size"), avg_rating=("rating", "mean"),
                negative_share=("sentiment", lambda s: (s == "Negative").mean() * 100)))
    return w[w["reviews"] >= min_reviews].reset_index()


def detect_changepoint(weekly: pd.DataFrame, col: str = "negative_share", margin: int = 3):
    """Split date that maximises the (scaled) difference in mean before vs after."""
    y, best, best_i = weekly[col].values, -1.0, None
    for i in range(margin, len(y) - margin):
        a, b = y[:i], y[i:]
        stat = abs(b.mean() - a.mean()) * np.sqrt(len(a) * len(b) / (len(a) + len(b)))
        if stat > best:
            best, best_i = stat, i
    return weekly["date"].iloc[best_i] if best_i is not None else None


def before_after(df: pd.DataFrame, split: pd.Timestamp) -> dict:
    b, a = df[df["date"] < split], df[df["date"] >= split]
    return {
        "split_date": split.date(),
        "neg_share_before_%": round((b["sentiment"] == "Negative").mean() * 100, 1),
        "neg_share_after_%": round((a["sentiment"] == "Negative").mean() * 100, 1),
        "avg_rating_before": round(b["rating"].mean(), 2),
        "avg_rating_after": round(a["rating"].mean(), 2),
    }


def theme_shift(df: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    comp = df["rating"] <= 2
    before = theme_summary(df, comp & (df["date"] < split)).set_index("theme")["share_of_complaints_%"]
    after = theme_summary(df, comp & (df["date"] >= split)).set_index("theme")["share_of_complaints_%"]
    out = pd.DataFrame({"before_%": before, "after_%": after}).fillna(0)
    out["change_pp"] = (out["after_%"] - out["before_%"]).round(1)
    return out.sort_values("change_pp", ascending=False).reset_index(names="theme")
