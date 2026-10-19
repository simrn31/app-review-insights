"""Phase B - Text pre-processing (lowercase, noise removal, stop words, tokenization, lemmatization)."""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations carry meaning in complaints ("not working", "cannot login"), so we keep them.
KEEP = {"not", "no", "never", "cannot", "nor", "against"}

try:  # Prefer NLTK's stop-word list when its data is available
    from nltk.corpus import stopwords as _sw
    _BASE_STOP = set(_sw.words("english"))
except Exception:  # LookupError / ImportError -> fall back to scikit-learn's list
    _BASE_STOP = set(ENGLISH_STOP_WORDS)
STOP_WORDS = _BASE_STOP - KEEP

_CONTRACTIONS = [(r"\bcan't\b", "cannot"), (r"\bwon't\b", "will not"),
                 (r"n't\b", " not"), (r"'(re|ve|ll|d|m|s)\b", "")]
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, expand contractions, strip URLs / emojis / special characters."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = text.replace("’", "'")
    for pat, rep in _CONTRACTIONS:
        text = re.sub(pat, rep, text)
    text = _NON_ALPHA.sub(" ", text)          # removes emojis, digits, @ # $ punctuation
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text)


@lru_cache(maxsize=1)
def _lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")  # raises LookupError if wordnet is not downloaded
        return lem.lemmatize
    except Exception:
        return _light_lemma


def _light_lemma(w: str) -> str:
    """Tiny plural-only fallback used when NLTK WordNet data is unavailable."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "shes", "ches", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def remove_stopwords_and_lemmatize(tokens: list[str]) -> list[str]:
    lemma = _lemmatizer()
    return [lemma(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["review_text", "rating", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df["rating"] = df["rating"].astype(int)
    df = df.drop_duplicates(subset=["review_text", "date"])
    df["text_clean"] = df["review_text"].map(clean_text)
    df = df[df["text_clean"].str.len() > 0]
    df["tokens"] = df["text_clean"].map(tokenize).map(remove_stopwords_and_lemmatize)
    df["text_processed"] = df["tokens"].map(" ".join)
    return df.reset_index(drop=True)
