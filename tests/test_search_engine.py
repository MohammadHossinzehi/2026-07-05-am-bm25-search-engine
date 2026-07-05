"""Unit tests for the search engine.

Run with:
    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from search_engine.tokenizer import tokenize
from search_engine.index import InvertedIndex
from search_engine.bm25 import BM25Scorer
from search_engine.query import QueryParser, QuerySyntaxError


class TokenizerTests(unittest.TestCase):
    def test_lowercases_and_splits_on_punctuation(self):
        self.assertEqual(
            tokenize("Hello, World! It's 2026.", remove_stopwords=False, stem=False),
            ["hello", "world", "it", "s", "2026"],
        )

    def test_removes_stopwords(self):
        tokens = tokenize("the cat sat on the mat", stem=False)
        self.assertNotIn("the", tokens)
        self.assertNotIn("on", tokens)
        self.assertIn("cat", tokens)
        self.assertIn("mat", tokens)

    def test_stems_plurals(self):
        self.assertEqual(tokenize("cats dogs", remove_stopwords=False), ["cat", "dog"])

    def test_stems_ing_and_ed(self):
        tokens = tokenize("running jumped", remove_stopwords=False)
        self.assertEqual(tokens, ["run", "jump"])

    def test_empty_string_yields_no_tokens(self):
        self.assertEqual(tokenize(""), [])


class InvertedIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = InvertedIndex()
        self.index.add_document("a", "the cat sat on the mat")
        self.index.add_document("b", "the dog sat on the log")

    def test_document_count(self):
        self.assertEqual(self.index.document_count(), 2)

    def test_postings_track_correct_documents(self):
        postings = self.index.postings("sat")
        self.assertEqual(set(postings.keys()), {"a", "b"})

    def test_term_only_in_one_doc(self):
        self.assertIn("a", self.index.postings("cat"))
        self.assertNotIn("b", self.index.postings("cat"))

    def test_document_frequency(self):
        self.assertEqual(self.index.document_frequency("sat"), 2)
        self.assertEqual(self.index.document_frequency("cat"), 1)

    def test_remove_document_cleans_postings(self):
        self.index.remove_document("a")
        self.assertEqual(self.index.document_count(), 1)
        self.assertNotIn("a", self.index.postings("sat"))
        self.assertNotIn("cat", self.index.vocabulary())

    def test_re_adding_document_replaces_it(self):
        self.index.add_document("a", "completely different text")
        self.assertNotIn("a", self.index.postings("cat"))
        self.assertIn("a", self.index.postings("different"))

    def test_phrase_positions_detects_consecutive_terms(self):
        idx = InvertedIndex()
        idx.add_document("x", "machine learning is fun and machine learning is popular")
        positions = idx.phrase_positions(["machine", "learn"], "x")
        self.assertEqual(len(positions), 2)

    def test_phrase_positions_empty_when_not_consecutive(self):
        idx = InvertedIndex()
        idx.add_document("x", "machine code and learning theory")
        positions = idx.phrase_positions(["machine", "learn"], "x")
        self.assertEqual(positions, [])

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            self.index.save(path)
            loaded = InvertedIndex.load(path)
            self.assertEqual(loaded.document_count(), self.index.document_count())
            self.assertEqual(loaded.document_frequency("sat"), self.index.document_frequency("sat"))


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.index = InvertedIndex()
        self.index.add_document("d1", "the cat sat on the mat")
        self.index.add_document("d2", "the dog sat on the log")
        self.index.add_document("d3", "cats and dogs are common pets, cats especially so")
        self.scorer = BM25Scorer(self.index)

    def test_document_with_more_query_term_occurrences_scores_higher(self):
        # d3 mentions "cat" (stemmed) twice, d1 mentions it once.
        score_d3 = self.scorer.score(["cat"], "d3")
        score_d1 = self.scorer.score(["cat"], "d1")
        self.assertGreater(score_d3, score_d1)

    def test_score_is_zero_for_document_missing_all_terms(self):
        self.assertEqual(self.scorer.score(["zebra"], "d1"), 0.0)

    def test_search_ranks_relevant_documents_first(self):
        results = self.scorer.search("cat")
        doc_ids = [doc_id for doc_id, _ in results]
        self.assertIn("d1", doc_ids)
        self.assertIn("d3", doc_ids)
        self.assertNotIn("d2", doc_ids)
        # d3 has two occurrences of "cat", should rank above d1.
        self.assertEqual(doc_ids[0], "d3")

    def test_search_with_no_matching_terms_returns_empty(self):
        self.assertEqual(self.scorer.search("zebra giraffe"), [])

    def test_rarer_term_contributes_more_idf(self):
        # "sat" appears in 2 of 3 docs, "mat" appears in only 1 of 3 docs;
        # the rarer term should carry a higher idf weight.
        idf_common = self.scorer.idf("sat")  # d1, d2
        idf_rare = self.scorer.idf("mat")     # d1 only
        self.assertGreater(idf_rare, idf_common)


class QueryParserTests(unittest.TestCase):
    def setUp(self):
        self.index = InvertedIndex()
        self.index.add_document("d1", "cats are great pets")
        self.index.add_document("d2", "dogs are great pets")
        self.index.add_document("d3", "cats and dogs coexist peacefully")
        self.index.add_document("d4", "machine learning models require data")
        self.parser = QueryParser(self.index)

    def test_implicit_and(self):
        result = self.parser.evaluate("cat dog")
        self.assertEqual(result, {"d3"})

    def test_explicit_or(self):
        result = self.parser.evaluate("cat OR dog")
        self.assertEqual(result, {"d1", "d2", "d3"})

    def test_not_excludes_documents(self):
        result = self.parser.evaluate("pet NOT dog")
        self.assertEqual(result, {"d1"})

    def test_parentheses_group_correctly(self):
        result = self.parser.evaluate("(cat OR dog) AND pet")
        self.assertEqual(result, {"d1", "d2"})

    def test_phrase_query_matches_exact_sequence(self):
        result = self.parser.evaluate('"machine learning"')
        self.assertEqual(result, {"d4"})

    def test_phrase_query_no_match_for_reordered_terms(self):
        result = self.parser.evaluate('"learning machine"')
        self.assertEqual(result, set())

    def test_unknown_term_returns_empty_set(self):
        result = self.parser.evaluate("zebra")
        self.assertEqual(result, set())

    def test_unbalanced_parens_raises(self):
        with self.assertRaises(QuerySyntaxError):
            self.parser.evaluate("(cat AND dog")

    def test_parse_strips_operators_for_ranking(self):
        terms = self.parser.parse("cat AND NOT dog")
        self.assertEqual(terms, ["cat", "dog"])


if __name__ == "__main__":
    unittest.main()
