"""Inverted index construction and storage.

An inverted index maps each term to the set of documents it appears in,
along with per document term frequencies. This is the data structure
every real text search engine (Lucene, Elasticsearch, PostgreSQL full
text search, ...) is built on top of.
"""

import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .tokenizer import tokenize


class InvertedIndex:
    """A simple in-memory inverted index with positional postings.

    Postings store term positions (not just counts) so that phrase
    queries ("machine learning") can be evaluated without re-scanning
    the original documents.
    """

    def __init__(self):
        # term -> {doc_id: [positions]}
        self._postings: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        # doc_id -> original text (kept so we can show snippets / rebuild)
        self._documents: Dict[str, str] = {}
        # doc_id -> title, purely cosmetic metadata for search results
        self._titles: Dict[str, str] = {}
        # doc_id -> token count, needed for BM25 length normalization
        self._doc_lengths: Dict[str, int] = {}

    # -- construction ------------------------------------------------

    def add_document(self, doc_id: str, text: str, title: Optional[str] = None) -> None:
        """Tokenize `text` and merge it into the index under `doc_id`.

        Re-adding an existing doc_id replaces the previous document.
        """
        if doc_id in self._documents:
            self.remove_document(doc_id)

        tokens = tokenize(text)
        self._documents[doc_id] = text
        self._titles[doc_id] = title if title is not None else doc_id
        self._doc_lengths[doc_id] = len(tokens)

        for position, term in enumerate(tokens):
            postings_for_term = self._postings[term]
            if doc_id not in postings_for_term:
                postings_for_term[doc_id] = []
            postings_for_term[doc_id].append(position)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document and clean up any postings that reference it."""
        if doc_id not in self._documents:
            return
        for term in list(self._postings.keys()):
            postings_for_term = self._postings[term]
            if doc_id in postings_for_term:
                del postings_for_term[doc_id]
                if not postings_for_term:
                    del self._postings[term]
        del self._documents[doc_id]
        del self._titles[doc_id]
        del self._doc_lengths[doc_id]

    # -- read access ---------------------------------------------------

    def postings(self, term: str) -> Dict[str, List[int]]:
        return self._postings.get(term, {})

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        return len(self._postings.get(term, {}))

    def term_frequency(self, term: str, doc_id: str) -> int:
        return len(self._postings.get(term, {}).get(doc_id, []))

    def doc_length(self, doc_id: str) -> int:
        return self._doc_lengths.get(doc_id, 0)

    def average_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def document_count(self) -> int:
        return len(self._documents)

    def all_doc_ids(self) -> Iterable[str]:
        return self._documents.keys()

    def get_text(self, doc_id: str) -> Optional[str]:
        return self._documents.get(doc_id)

    def get_title(self, doc_id: str) -> Optional[str]:
        return self._titles.get(doc_id)

    def vocabulary(self) -> Iterable[str]:
        return self._postings.keys()

    def phrase_positions(self, terms: List[str], doc_id: str) -> List[int]:
        """Return start positions in `doc_id` where `terms` occur consecutively.

        Used by QueryParser to evaluate quoted phrase queries.
        """
        if not terms:
            return []
        first_positions = self._postings.get(terms[0], {}).get(doc_id, [])
        matches = []
        for start in first_positions:
            ok = True
            for offset, term in enumerate(terms[1:], start=1):
                positions = self._postings.get(term, {}).get(doc_id, [])
                if (start + offset) not in positions:
                    ok = False
                    break
            if ok:
                matches.append(start)
        return matches

    # -- persistence ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "postings": self._postings,
            "documents": self._documents,
            "titles": self._titles,
            "doc_lengths": self._doc_lengths,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InvertedIndex":
        idx = cls()
        idx._postings = defaultdict(dict, {k: dict(v) for k, v in data["postings"].items()})
        idx._documents = dict(data["documents"])
        idx._titles = dict(data["titles"])
        idx._doc_lengths = dict(data["doc_lengths"])
        return idx

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "InvertedIndex":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def build_from_corpus(cls, corpus: List[dict]) -> "InvertedIndex":
        """Build an index from a list of {id, title, text} dicts."""
        idx = cls()
        for doc in corpus:
            idx.add_document(doc["id"], doc["text"], title=doc.get("title"))
        return idx
