"""BM25 ranking (Okapi BM25).

BM25 is the scoring function behind most production search systems
(Elasticsearch and Lucene both use it by default). For a query Q with
terms q1..qn and a document D, the score is:

    score(D, Q) = sum over qi of IDF(qi) * ( f(qi, D) * (k1 + 1) )
                                            -------------------------
                                            f(qi, D) + k1 * (1 - b + b * |D| / avgdl)

Where:
    f(qi, D)  term frequency of qi in D
    |D|       length of D in tokens
    avgdl     average document length across the corpus
    IDF(qi)   log( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )
    N         total number of documents
    n(qi)     number of documents containing qi
    k1, b     free parameters, defaults 1.5 and 0.75 (standard in the literature)
"""

import math
from typing import List, Tuple

from .index import InvertedIndex
from .tokenizer import tokenize


class BM25Scorer:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        n = self.index.document_count()
        df = self.index.document_frequency(term)
        # +1 inside the log keeps IDF non-negative even when a term
        # appears in every document (df == n), the standard Lucene/BM25+ fix.
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_terms: List[str], doc_id: str) -> float:
        avgdl = self.index.average_doc_length()
        doc_len = self.index.doc_length(doc_id)
        if avgdl == 0 or doc_len == 0:
            return 0.0

        total = 0.0
        for term in query_terms:
            tf = self.index.term_frequency(term, doc_id)
            if tf == 0:
                continue
            idf = self.idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            total += idf * (numerator / denominator)
        return total

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Rank every document that shares at least one term with the query.

        Returns a list of (doc_id, score) sorted by descending score,
        truncated to `top_k` results. Documents with a score of 0 are
        excluded.
        """
        query_terms = tokenize(query)
        if not query_terms:
            return []

        candidate_doc_ids = set()
        for term in query_terms:
            candidate_doc_ids.update(self.index.postings(term).keys())

        scored = []
        for doc_id in candidate_doc_ids:
            s = self.score(query_terms, doc_id)
            if s > 0:
                scored.append((doc_id, s))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
