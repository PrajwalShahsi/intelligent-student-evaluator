"""
Text Preprocessing module for Student Assignment Evaluator.
Implements Step 4:
- Lowercasing
- Acronym normalization (e.g. AI / Al <-> artificial intelligence, NLP <-> natural language processing)
- Punctuation removal
- Tokenization & stopword removal (with safe offline fallbacks)
- Lemmatization (with safe offline fallback)
- Key concept/term extraction
"""

import re
import string
from typing import Dict, List, Set, Tuple

# Built-in robust English stopword fallback to guarantee zero offline crashes
FALLBACK_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Domain acronym mappings for dataset vocabulary (including OCR typos like 'Al' for 'AI')
ACRONYM_MAP = {
    r"\bnlp\b": "natural language processing",
    r"\b(ai|al)\b": "artificial intelligence",
    r"\bcnn\b": "convolutional neural network",
    r"\bcnns\b": "convolutional neural networks",
    r"\bml\b": "machine learning",
    r"\bcv\b": "computer vision",
}

_NLTK_INITIALIZED = False
_STOPWORDS = FALLBACK_STOPWORDS
_LEMMATIZER = None


def _init_nltk():
    """Lazily and safely initialize NLTK resources with offline fallbacks."""
    global _NLTK_INITIALIZED, _STOPWORDS, _LEMMATIZER
    if _NLTK_INITIALIZED:
        return

    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

        try:
            _STOPWORDS = set(stopwords.words("english"))
        except Exception:
            _STOPWORDS = FALLBACK_STOPWORDS

        try:
            _LEMMATIZER = WordNetLemmatizer()
            _LEMMATIZER.lemmatize("running")
        except Exception:
            _LEMMATIZER = None

    except Exception:
        _STOPWORDS = FALLBACK_STOPWORDS
        _LEMMATIZER = None

    _NLTK_INITIALIZED = True


def normalize_acronyms(text: str) -> str:
    """Expand known domain acronyms to ensure consistent matching."""
    norm = text
    for pattern, expansion in ACRONYM_MAP.items():
        norm = re.sub(pattern, expansion, norm, flags=re.IGNORECASE)
    return norm


def fallback_lemmatize(word: str) -> str:
    """Lightweight suffix-based stem/lemmatize fallback if WordNet is offline."""
    w = word.lower()
    if len(w) > 4:
        if w.endswith("ies") and not w.endswith("eies"):
            return w[:-3] + "y"
        if w.endswith("ing"):
            return w[:-3]
        if w.endswith("ed"):
            return w[:-2]
        if w.endswith("es"):
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
    return w


def lemmatize_word(word: str) -> str:
    """Lemmatize a single word using NLTK WordNetLemmatizer or fallback."""
    _init_nltk()
    if _LEMMATIZER:
        try:
            lem = _LEMMATIZER.lemmatize(word, pos="v")
            if lem == word:
                lem = _LEMMATIZER.lemmatize(word, pos="n")
            return lem
        except Exception:
            pass
    return fallback_lemmatize(word)


def preprocess_text(text: str, expand_acronyms: bool = True) -> Dict[str, object]:
    """
    Execute full Step 4 NLP preprocessing pipeline:
    1. Lowercase
    2. Acronym normalization
    3. Punctuation removal
    4. Tokenization
    5. Stopword removal
    6. Lemmatization
    Returns a dictionary with cleaned_text, tokens, lemmatized_tokens, and key_terms.
    """
    _init_nltk()
    if not text:
        return {
            "raw_text": "",
            "normalized_text": "",
            "cleaned_text": "",
            "tokens": [],
            "lemmatized_tokens": [],
            "key_terms": [],
        }

    # 1. Lowercase
    lower_text = text.lower().strip()

    # 2. Acronym normalization
    norm_text = normalize_acronyms(lower_text) if expand_acronyms else lower_text

    # 3. Punctuation removal (keep alphanumerics and spaces)
    clean_punc = re.sub(r"[^\w\s]", " ", norm_text)
    clean_punc = re.sub(r"\s+", " ", clean_punc).strip()

    # 4. Tokenization (word boundary)
    tokens = [t for t in clean_punc.split() if t]

    # 5. Stopword filtering & 6. Lemmatization
    filtered_tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    lemmatized_tokens = [lemmatize_word(t) for t in filtered_tokens]

    # Cleaned text reconstruction
    cleaned_text = " ".join(lemmatized_tokens)

    # Key terms: unique important concepts/tokens preserving order
    seen = set()
    key_terms = []
    for t in filtered_tokens:
        if t not in seen:
            seen.add(t)
            key_terms.append(t)

    return {
        "raw_text": text,
        "normalized_text": norm_text,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "key_terms": key_terms,
    }
