"""Phase A - Data Collection.

Scrapes reviews (text, star rating, date) from the Google Play Store using
`google-play-scraper`. Requires internet access.
"""
from __future__ import annotations

import pandas as pd


def scrape_play_store(app_id: str, count: int = 3000, lang: str = "en",
                      country: str = "us") -> pd.DataFrame:
    """Fetch the newest `count` reviews for `app_id`.

    Returns a DataFrame with: review_id, review_text, rating, date, app_version.
    """
    from google_play_scraper import Sort, reviews  # imported lazily

    collected, token = [], None
    while len(collected) < count:
        batch, token = reviews(
            app_id, lang=lang, country=country, sort=Sort.NEWEST,
            count=min(200, count - len(collected)), continuation_token=token,
        )
        collected.extend(batch)
        if token is None or not batch:
            break

    df = pd.DataFrame(collected)
    df = df.rename(columns={"reviewId": "review_id", "content": "review_text",
                            "score": "rating", "at": "date",
                            "appVersion": "app_version"})
    cols = ["review_id", "review_text", "rating", "date", "app_version"]
    df = df[[c for c in cols if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])
    return df


def validate(df: pd.DataFrame, minimum: int = 1000) -> None:
    if len(df) < minimum:
        raise ValueError(f"Only {len(df)} reviews collected; task requires >= {minimum}.")
