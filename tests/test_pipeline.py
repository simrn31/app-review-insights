import pandas as pd

from src import analyze, preprocess


def test_clean_text_removes_noise_and_lowercases():
    out = preprocess.clean_text("LOGIN Failed!!! 😡 @support #bug $5 https://x.io")
    assert out == "login failed support bug"


def test_negations_survive_stopword_removal():
    toks = preprocess.remove_stopwords_and_lemmatize(preprocess.tokenize("the app is not working"))
    assert "not" in toks and "working" in toks and "the" not in toks


def test_contraction_expansion():
    assert preprocess.clean_text("can't login") == "cannot login"


def test_sentiment_labels():
    df = pd.DataFrame({"text_clean": ["love this great app", "terrible crash not working", "it is a app"],
                       "rating": [5, 1, 3]})
    out = analyze.add_sentiment(df)
    assert list(out["sentiment"]) == ["Positive", "Negative", "Neutral"]


def test_bigrams():
    s = pd.Series(["login fail", "login fail again", "app crash"])
    assert analyze.top_ngrams(s, top=1, min_df=1).iloc[0]["ngram"] == "login fail"


def test_pipeline_end_to_end_and_changepoint():
    import pandas as pd
    from src import generate_sample_data, pipeline
    df = pipeline.prepare(generate_sample_data.generate())
    res = pipeline.summarize(df)
    assert res["ba"]["neg_share_after_%"] > res["ba"]["neg_share_before_%"] + 15
    assert abs((res["split"] - pd.Timestamp("2026-08-12")).days) <= 10
    assert res["themes"].iloc[0]["complaints"] > 0


def test_summarize_handles_tiny_slice():
    from src import generate_sample_data, pipeline
    df = pipeline.prepare(generate_sample_data.generate()).head(40)
    res = pipeline.summarize(df)  # must not raise
    assert "df" in res
