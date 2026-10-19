"""Streamlit dashboard:  streamlit run app.py"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src import collect, generate_sample_data, pipeline, visualize

st.set_page_config(page_title="App Review Insights", page_icon="📱", layout="wide")


@st.cache_data(show_spinner="Generating sample data…")
def load_sample() -> pd.DataFrame:
    return generate_sample_data.generate()


@st.cache_data(show_spinner="Scraping Google Play…")
def load_play(app_id: str, count: int) -> pd.DataFrame:
    return collect.scrape_play_store(app_id, count)


@st.cache_data(show_spinner="Cleaning text and scoring sentiment…")
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    return pipeline.prepare(raw)


def fig_png(draw_fn, *args) -> bytes:
    buf = io.BytesIO()
    draw_fn(*args, buf)
    return buf.getvalue()


# ----------------------------------------------------------------- sidebar
st.sidebar.title("📱 App Review Insights")
src = st.sidebar.radio("Data source", ["Sample (synthetic demo)", "Upload CSV", "Google Play app"])
raw, source = None, ""
if src.startswith("Sample"):
    raw, source = load_sample(), "SYNTHETIC sample data (demo only)"
    st.sidebar.warning("Synthetic data - not real user reviews.")
elif src == "Upload CSV":
    up = st.sidebar.file_uploader("CSV with review_text, rating, date", type="csv")
    if up:
        raw, source = pd.read_csv(up), f"Uploaded CSV ({up.name})"
else:
    app_id = st.sidebar.text_input("Package name", "com.spotify.music")
    n = st.sidebar.slider("Reviews to fetch", 1000, 10000, 3000, 500)
    if st.sidebar.button("Fetch reviews"):
        try:
            st.session_state["play"] = (load_play(app_id, n), f"Google Play ({app_id})")
        except Exception as e:  # network / bad package name
            st.sidebar.error(f"Could not fetch reviews: {e}")
    raw, source = st.session_state.get("play", (None, ""))

if raw is None:
    st.info("Choose a data source in the sidebar to begin.")
    st.stop()

missing = {"review_text", "rating", "date"} - set(raw.columns)
if missing:
    st.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df_all = prepare(raw)

dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
stars = st.sidebar.multiselect("Star ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
known = st.sidebar.checkbox("I know the release date")
upd = st.sidebar.date_input("Release date", dmax) if known else None

if len(rng) != 2:
    st.stop()
df = df_all[(df_all["date"].dt.date >= rng[0]) & (df_all["date"].dt.date <= rng[1]) & df_all["rating"].isin(stars)]
if len(df) < 30:
    st.warning("Too few reviews in this selection - widen the filters.")
    st.stop()
res = pipeline.summarize(df, upd)

# --------------------------------------------------------------------- KPIs
st.title("App Review Insights - Actionable Complaints")
st.caption(f"{source} · sentiment engine: **{res['engine']}**")
c = st.columns(4)
c[0].metric("Reviews", f"{len(df):,}")
c[1].metric("Average rating", f"{df['rating'].mean():.2f}")
c[2].metric("Unsatisfied (1-2★)", f"{(df['rating'] <= 2).mean():.0%}")
c[3].metric("Negative sentiment", f"{(df['sentiment'] == 'Negative').mean():.0%}")

t1, t2, t3, t4, t5 = st.tabs(["Trend", "Complaints", "Distribution", "Review explorer", "Management summary"])

with t1:
    if res["ba"]:
        ba = res["ba"]
        st.success(f"Biggest shift around **{ba['split_date']}**: negative reviews "
                   f"{ba['neg_share_before_%']}% → {ba['neg_share_after_%']}%, "
                   f"avg rating {ba['avg_rating_before']} → {ba['avg_rating_after']}.")
    else:
        st.info("Not enough weekly data to detect a shift.")
    if len(res["weekly"]):
        w = res["weekly"].set_index("date")
        st.subheader("Weekly negative sentiment (%)"); st.line_chart(w["negative_share"])
        st.subheader("Weekly average rating"); st.line_chart(w["avg_rating"])
    if len(res["shift"]):
        st.subheader("Complaint themes: before vs after the shift")
        st.image(fig_png(visualize.theme_shift_chart, res["shift"]))

with t2:
    a, b = st.columns(2)
    with a:
        st.subheader("Complaint themes (1-2★)")
        st.bar_chart(res["themes"].set_index("theme")["share_of_complaints_%"])
        st.subheader("Top bigrams")
        st.dataframe(res["bigrams"], hide_index=True, use_container_width=True)
    with b:
        st.subheader("Word cloud of pain")
        if len(res["complaints"]):
            st.image(fig_png(visualize.pain_cloud, res["complaints"]["tokens"]))
        st.subheader("Words most tied to 1-2★ reviews")
        st.dataframe(res["gap_low"][["term", "gap"]], hide_index=True, use_container_width=True)

with t3:
    st.image(fig_png(visualize.distribution_plot, df))
    st.caption(f"Spearman correlation between stars and text sentiment: {res['corr']:.2f}")

with t4:
    themes = [c.replace("theme::", "") for c in df.columns if c.startswith("theme::")]
    f1, f2 = st.columns(2)
    pick = f1.multiselect("Theme", themes)
    sent = f2.multiselect("Sentiment", ["Negative", "Neutral", "Positive"], default=["Negative"])
    view = df[df["sentiment"].isin(sent)]
    if pick:
        view = view[view[[f"theme::{t}" for t in pick]].any(axis=1)]
    st.write(f"{len(view):,} reviews")
    show = view[["date", "rating", "sentiment", "polarity", "review_text"]].sort_values("date", ascending=False)
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.download_button("Download filtered reviews (CSV)", show.to_csv(index=False), "filtered_reviews.csv")

with t5:
    md = pipeline.write_summary(res, source)
    st.markdown(md)
    st.download_button("Download summary (.md)", md, "management_summary.md")
