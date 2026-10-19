"""End-to-end pipeline:  python main.py --source sample | play --app-id com.example.app | csv --csv file.csv"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import collect, generate_sample_data, pipeline, visualize

ROOT = Path(__file__).parent
RAW, PROC = ROOT / "data/raw/reviews_raw.csv", ROOT / "data/processed/reviews_clean.csv"
FIG, TAB = ROOT / "outputs/figures", ROOT / "outputs/tables"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sample", "play", "csv"], default="sample")
    ap.add_argument("--app-id", help="Google Play package name, e.g. com.spotify.music")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--csv", help="Existing CSV with review_text, rating, date")
    ap.add_argument("--update-date", help="Known release date (YYYY-MM-DD); auto-detected if omitted")
    a = ap.parse_args()

    # Phase A - collection
    if a.source == "play":
        if not a.app_id:
            ap.error("--app-id is required with --source play")
        raw = collect.scrape_play_store(a.app_id, a.count); collect.validate(raw)
        source = f"Google Play Store ({a.app_id})"
    elif a.source == "csv":
        raw = pd.read_csv(a.csv); source = f"CSV file ({a.csv})"
    else:
        raw = generate_sample_data.generate()
        source = "SYNTHETIC sample data (demo only - not real user reviews)"
    RAW.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW, index=False)
    print(f"[A] collected {len(raw):,} reviews")

    # Phases B + C
    df = pipeline.prepare(raw)
    res = pipeline.summarize(df, a.update_date)
    print(f"[B/C] {len(df):,} reviews | sentiment engine: {res['engine']}")
    if res["ba"]:
        ba = res["ba"]
        print(f"      change point {ba['split_date']}: negative {ba['neg_share_before_%']}% -> {ba['neg_share_after_%']}%")

    # Phase D
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    visualize.trend_line(res["weekly"], res["split"], FIG / "1_trend_line.png")
    visualize.pain_cloud(res["complaints"]["tokens"], FIG / "2_word_cloud_of_pain.png")
    visualize.distribution_plot(df, FIG / "3_distribution_plot.png")
    visualize.bar_chart(res["bigrams"], "count", "ngram", "Top complaint bigrams (1-2★ reviews)", FIG / "4_top_bigrams.png")
    visualize.bar_chart(res["themes"], "share_of_complaints_%", "theme", "Complaint themes (% of 1-2★ reviews)",
                        FIG / "5_complaint_themes.png")
    if len(res["shift"]):
        visualize.theme_shift_chart(res["shift"], FIG / "6_theme_shift.png")
    for name, key in {"top_bigrams": "bigrams", "complaint_themes": "themes", "theme_shift": "shift",
                      "distinctive_words_1_2_star": "gap_low", "distinctive_words_4_5_star": "gap_high",
                      "weekly_trend": "weekly"}.items():
        res[key].to_csv(TAB / f"{name}.csv", index=False)
    df.drop(columns=["tokens"]).to_csv(PROC, index=False)
    (ROOT / "outputs/management_summary.md").write_text(pipeline.write_summary(res, source), encoding="utf-8")
    print("[D] figures, tables and outputs/management_summary.md written")


if __name__ == "__main__":
    main()
