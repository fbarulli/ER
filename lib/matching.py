"""lib/matching.py: TF-IDF cosine title matching for euromonitor resolution.

Single source for the title-similarity machinery (corpus vectorizer + cosine
scoring). Used by the EDA views for the ground-truth validation. Regex/text
parsing stays in lib/text.py; this module is the sklearn side (no regexes
here). Stopwords live in lib/sklearn_stopwords.json — same dir as this
consumer — frozen from sklearn's ENGLISH_STOP_WORDS so the applied list is
explicit and version-independent (the old stop_words="english" hid a third
stopword source outside the config SSOT). Renamed 2026-09-08 alongside the
config split: the data_pipe word-list SSOT moved to lib/pipe_stopwords.json,
so the two lists no longer share a basename.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MAX_FEATURES = 2000

_STOPWORDS_PATH = Path(__file__).resolve().parent / "sklearn_stopwords.json"


def load_english_stopwords() -> list[str]:
    """The frozen sklearn ENGLISH_STOP_WORDS list (lib/sklearn_stopwords.json).

    Same set the vectorizer historically got via stop_words="english";
    returned as a LIST (sklearn's stop_words param rejects frozensets).
    Loaded once, exposed for tests to pin against sklearn directly.
    """
    with open(_STOPWORDS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return list(data["ENGLISH_STOP_WORDS"])


def build_vectorizer(max_features: int = DEFAULT_MAX_FEATURES) -> TfidfVectorizer:
    """Corpus vectorizer: (1,2) word ngrams, sublinear TF, English stopwords.

    (1,2) ngrams let "coconut water" and "coconut" overlap even when word
    order/extra words differ; sublinear_tf dampens repeated tokens.
    """
    return TfidfVectorizer(
        stop_words=load_english_stopwords(),
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
    )
