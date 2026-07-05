"""Boolean and phrase query parsing/evaluation.

Supported syntax:
    cat dog              implicit AND between bare terms
    cat AND dog          explicit AND
    cat OR dog           explicit OR
    cat NOT dog          exclude documents containing "dog"
    "cat dog"            phrase query, matches consecutive occurrence
    (cat OR dog) AND fox parentheses for grouping

The grammar (case sensitive operators, everything else case folded
during tokenization) is a small recursive descent parser:

    expr    := term (("AND" | "OR" | "NOT" | <implicit>) term)*
    term    := phrase | word | "(" expr ")"
    phrase  := '"' word+ '"'

Evaluation returns the *set* of matching doc_ids; ranking within that
set (if desired) is left to BM25Scorer so the two concerns stay
separate: query.py decides which documents are eligible, bm25.py
decides what order to show them in.
"""

import re
from typing import List, Set

from .index import InvertedIndex
from .tokenizer import tokenize

_TOKEN_PATTERN = re.compile(r'"[^"]*"|\(|\)|AND|OR|NOT|[A-Za-z0-9]+')


class QuerySyntaxError(ValueError):
    pass


class QueryParser:
    def __init__(self, index: InvertedIndex):
        self.index = index

    def parse(self, query_string: str) -> List[str]:
        """Convenience helper: returns tokens (bare words + phrase words)
        with operators stripped out, useful for BM25 ranking of a query
        that also uses boolean syntax."""
        terms = []
        for tok in _TOKEN_PATTERN.findall(query_string):
            if tok in ("AND", "OR", "NOT", "(", ")"):
                continue
            if tok.startswith('"'):
                terms.extend(tokenize(tok.strip('"')))
            else:
                terms.extend(tokenize(tok))
        return terms

    def evaluate(self, query_string: str) -> Set[str]:
        """Evaluate a boolean/phrase query and return the matching doc_ids."""
        tokens = _TOKEN_PATTERN.findall(query_string)
        if not tokens:
            return set()
        self._tokens = tokens
        self._pos = 0
        result = self._parse_expr()
        if self._pos != len(self._tokens):
            raise QuerySyntaxError(f"Unexpected token: {self._tokens[self._pos]}")
        return result

    # -- recursive descent internals -----------------------------------

    def _peek(self):
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self):
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _parse_expr(self) -> Set[str]:
        result = self._parse_term()
        while True:
            tok = self._peek()
            if tok is None or tok == ")":
                break
            if tok in ("AND", "OR", "NOT"):
                op = self._advance()
            else:
                # Two terms with no explicit operator between them are
                # implicitly ANDed, e.g. "cat dog" == "cat AND dog".
                op = "AND"
            rhs = self._parse_term()
            if op == "AND":
                result = result & rhs
            elif op == "OR":
                result = result | rhs
            else:  # NOT
                result = result - rhs
        return result

    def _parse_term(self) -> Set[str]:
        tok = self._peek()
        if tok is None:
            raise QuerySyntaxError("Unexpected end of query")
        if tok == "(":
            self._advance()
            result = self._parse_expr()
            if self._peek() != ")":
                raise QuerySyntaxError("Expected closing parenthesis")
            self._advance()
            return result
        if tok.startswith('"'):
            self._advance()
            phrase_terms = tokenize(tok.strip('"'))
            return self._docs_with_phrase(phrase_terms)
        # bare word
        self._advance()
        term = tokenize(tok)
        if not term:
            return set()
        return set(self.index.postings(term[0]).keys())

    def _docs_with_phrase(self, phrase_terms: List[str]) -> Set[str]:
        if not phrase_terms:
            return set()
        candidate_docs = set(self.index.postings(phrase_terms[0]).keys())
        matches = set()
        for doc_id in candidate_docs:
            if self.index.phrase_positions(phrase_terms, doc_id):
                matches.add(doc_id)
        return matches
