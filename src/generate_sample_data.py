"""Generate a SYNTHETIC review dataset so the pipeline can be demoed offline.

!! This is NOT real user data. For the actual submission, run
   `python main.py --source play --app-id <package.name>`.

The data simulates an app whose v5.0 release (2026-08-12) introduced login
failures and crashes, so the trend analysis has a known event to detect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UPDATE_DATE = pd.Timestamp("2026-08-12")

NEG = {
    "login": [
        "Can't log in after the update, login fail every time",
        "Login failed again. Keeps saying invalid password even after I reset it",
        "Stuck on the login screen, login error since the new version",
        "Unable to sign in, authentication error keeps popping up",
    ],
    "crash": [
        "App keeps crashing on startup",
        "Crashes every time I open the camera tab",
        "App crashes after the latest update, so unstable",
        "Constant crash and freezes, completely unusable",
    ],
    "pricing": [
        "Subscription is way too expensive for what it offers",
        "Why is everything behind a paywall now? Overpriced subscription",
        "Charged twice for my monthly subscription, want a refund",
        "Premium price is ridiculous, not worth it",
    ],
    "ui": [
        "The new interface is confusing, can't find the settings",
        "Terrible redesign, navigation is confusing and cluttered",
        "Hard to use, buttons are tiny and the layout makes no sense",
    ],
    "performance": [
        "So slow and laggy, takes forever to load",
        "Battery drain is awful and the app is slow",
        "Loading screen takes forever, super laggy",
    ],
    "ads": [
        "Too many ads, can't even use the app without popups",
        "Ads everywhere, ruined the experience",
    ],
    "notif": [
        "Notifications not working at all",
        "Not getting any notifications, sync not working",
    ],
}
POS = [
    "Love this app, so easy to use", "Great features and a smooth experience",
    "Excellent app, fast and reliable", "Very helpful, clean interface, worth the price",
    "Amazing update, works perfectly", "Best app in its category, highly recommend",
    "Simple and intuitive design", "Fantastic, does everything I need",
]
NEUTRAL = [
    "It is okay, does the job", "Average app, nothing special",
    "Works fine but could use more features", "Decent, but needs some improvements",
]
NOISE = ["", "", "", " !!!", " :(", " 😡😡", " 👍", " @@ ## $$", " ...", " 🙄", " 😍"]

W_BEFORE = dict(login=.04, crash=.05, pricing=.28, ui=.20, performance=.20, ads=.15, notif=.08)
W_AFTER = dict(login=.34, crash=.26, pricing=.14, ui=.10, performance=.08, ads=.05, notif=.03)


def generate(n_days: int = 180, per_day: int = 17, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-10-11") - pd.Timedelta(days=n_days)
    rows = []
    for d in range(n_days):
        day = start + pd.Timedelta(days=d)
        after = day >= UPDATE_DATE
        p_neg = 0.46 if after else 0.20
        weights = W_AFTER if after else W_BEFORE
        themes, probs = list(weights), np.array(list(weights.values()))
        probs = probs / probs.sum()
        for _ in range(rng.poisson(per_day)):
            r = rng.random()
            if r < p_neg:
                theme = rng.choice(themes, p=probs)
                text = str(rng.choice(NEG[theme]))
                rating = int(rng.choice([1, 2], p=[.6, .4]))
            elif r < p_neg + (1 - p_neg) * 0.75:
                text = str(rng.choice(POS))
                rating = int(rng.choice([4, 5], p=[.35, .65]))
            else:
                text = str(rng.choice(NEUTRAL))
                rating = 3
            if rng.random() < 0.05:  # label noise, like real stores
                rating = int(rng.integers(1, 6))
            if rng.random() < 0.3:
                text = text.lower() if rng.random() < .5 else text.upper()
            text += str(rng.choice(NOISE))
            rows.append((day + pd.Timedelta(minutes=int(rng.integers(0, 1440))), text, rating,
                         "5.0.0" if after else "4.9.2"))
    df = pd.DataFrame(rows, columns=["date", "review_text", "rating", "app_version"])
    df.insert(0, "review_id", [f"sample-{i:05d}" for i in range(len(df))])
    return df
