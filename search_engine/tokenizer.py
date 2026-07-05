"""Tokenization and light stemming for the search engine.

The goal here is not linguistic perfection, it is a small, well tested,
dependency free pipeline that is good enough to build a real inverted
index on top of. Three steps happen in order:

1. Lowercase and split on anything that is not a letter or digit.
2. Drop a small, fixed English stopword list.
3. Apply a light suffix stripping stemmer (a tiny subset of the ideas
   behind the Porter stemmer: plurals, "-ing", "-ed", "-ly").
"""

import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the and or but if while of at by for with about against between
    into through during before after above below to from up down in out
    on off over under again further then once here there when where why
    how all any both each few more most other some such no nor not only
    own same so than too very s t can will just don should now is are was
    were be been being have has had do does did this that these those it
    its i you he she we they them his her their our your as
    """.split()
)


def _stem(word: str) -> str:
    """Strip a handful of common English suffixes.

    This intentionally does not try to be a full Porter stemmer. It
    handles the small set of suffix rules that give the most practical
    benefit for recall (matching "running" to "run", "cats" to "cat")
    while staying easy to read and to unit test.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        return _fix_stem_ending(word[:-3])
    if len(word) > 4 and word.endswith("ed"):
        return _fix_stem_ending(word[:-2])
    if len(word) > 4 and word.endswith("ly"):
        return word[:-2]
    return word


def _fix_stem_ending(stem: str) -> str:
    """Undo the two spelling changes English makes before "-ing"/"-ed":

    a doubled final consonant ("running" -> "runn" -> "run") or a
    dropped silent "e" ("hoping" -> "hop" -> "hope"). A word can only
    have had one of these applied, so we check doubling first since it
    is unambiguous, and only consider restoring "e" otherwise.
    """
    if _is_doubled_consonant(stem):
        return stem[:-1]
    if _needs_silent_e(stem):
        return stem + "e"
    return stem


def _is_doubled_consonant(stem: str) -> bool:
    vowels = "aeiou"
    return len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in vowels


def _needs_silent_e(stem: str) -> bool:
    """Heuristic: restore a trailing "e" dropped before "-ing"/"-ed".

    e.g. "hoping" -> stem "hop" would collide with "hopping" -> "hopp"->"hop".
    We only re-add the "e" when the stem ends in a single consonant
    preceded by a single vowel and is short (mirrors "hope"/"hoping").
    """
    vowels = "aeiou"
    if len(stem) < 2:
        return False
    return stem[-1] not in vowels and stem[-2] in vowels and len(stem) <= 5


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True) -> List[str]:
    """Turn raw text into a list of normalized tokens.

    Args:
        text: raw input text.
        remove_stopwords: drop tokens found in STOPWORDS.
        stem: apply the light suffix stemmer.

    Returns:
        A list of tokens in the order they appeared in the text.
    """
    raw_tokens = _TOKEN_RE.findall(text.lower())
    tokens = []
    for tok in raw_tokens:
        if remove_stopwords and tok in STOPWORDS:
            continue
        tokens.append(_stem(tok) if stem else tok)
    return tokens
