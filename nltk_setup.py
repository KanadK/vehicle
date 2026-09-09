"""Ensure the NLTK corpora this application needs are available.

Kept in its own module so the Docker build can bake the data into the image
using exactly the resource list the app resolves at runtime, rather than a
second copy of that list in the Dockerfile that could drift out of step.

Run directly to fetch anything missing:

    python nltk_setup.py
"""
import nltk


def _ensure_nltk_resource(path, package):
    """Download `package` unless it is already present. Returns True on success."""
    # Depending on the NLTK version some corpora stay zipped (corpora/wordnet.zip)
    # while others are expanded (corpora/stopwords), so probe both spellings.
    # Getting this wrong is not fatal but re-downloads on every startup.
    for candidate in (path, path + '.zip'):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return nltk.download(package, quiet=True)


def ensure_nltk_data():
    """Fetch any corpus that is missing. A warm environment makes no network calls."""
    # word_tokenize() wants punkt_tab on NLTK >= 3.8.2 and punkt before that.
    # Try the modern name first and fall back, so either version works.
    if not _ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab'):
        _ensure_nltk_resource('tokenizers/punkt', 'punkt')

    _ensure_nltk_resource('corpora/stopwords', 'stopwords')            # stopwords.words('english')
    _ensure_nltk_resource('corpora/wordnet', 'wordnet')                # WordNetLemmatizer
    _ensure_nltk_resource('sentiment/vader_lexicon', 'vader_lexicon')  # SentimentIntensityAnalyzer


if __name__ == '__main__':
    ensure_nltk_data()
