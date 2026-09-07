"""lib/matching.py: TF-IDF cosine title matching for euromonitor resolution.

Single source for the title-similarity machinery (corpus vectorizer + cosine
scoring). Used by the EDA views for the ground-truth validation. Regex/text parsing stays in _text.py;
this module is the sklearn side (no regexes here).
"""

from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MAX_FEATURES = 2000


def build_vectorizer(max_features: int = DEFAULT_MAX_FEATURES) -> TfidfVectorizer:
    """Corpus vectorizer: (1,2) word ngrams, sublinear TF, English stopwords.

    (1,2) ngrams let "coconut water" and "coconut" overlap even when word
    order/extra words differ; sublinear_tf dampens repeated tokens.
    """
    return TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
    )
