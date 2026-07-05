# BM25 Search Engine

A from-scratch full-text search engine in pure Python: tokenizer, positional inverted index, BM25 ranking, and a boolean/phrase query language, with a CLI on top. No external dependencies, no vector database, no third-party search library  -  just the algorithms that power real systems like Elasticsearch and Lucene, built from first principles so the mechanics are visible and testable.

## Why this exists

Most "search" in small projects is `if query in text`, which breaks down immediately: it can't rank results, can't handle multi-word queries well, and can't answer "show me documents about X but not Y." This project implements the actual pipeline a real search engine uses:

1. **Tokenizer**  -  lowercases text, strips punctuation, removes stopwords, and applies a light suffix stemmer so "running", "runs", and "run" all index to the same term.
2. **Inverted index**  -  for every term, stores which documents contain it and at *which positions*, not just a count. Positional data is what makes phrase queries ("machine learning" as an exact phrase) possible without re-scanning raw text.
3. **BM25 ranking**  -  the same scoring function (Okapi BM25) used by default in Lucene/Elasticsearch. It balances term frequency against document length and how rare a term is across the whole corpus (inverse document frequency), so common words don't dominate and longer documents aren't unfairly favored.
4. **Query language**  -  a small recursive-descent parser supporting `AND`, `OR`, `NOT`, parentheses for grouping, and quoted phrases, e.g. `(cat OR dog) AND "search engine"`.

## How to run it

Requires Python 3.8+, standard library only.

```bash
# Build an index from a JSON corpus (list of {id, title, text})
python -m search_engine.cli build sample_data/docs.json --out index.json

# Rank documents by relevance (BM25)
python -m search_engine.cli search index.json "machine learning" --mode bm25

# Boolean + phrase queries, ranked by BM25 within the matching set
python -m search_engine.cli search index.json '"search engine" OR database' --mode boolean

# Interactive query loop
python -m search_engine.cli repl index.json --mode bm25
```

To index your own documents, point `build` at any JSON file shaped like `sample_data/docs.json`:

```json
[{"id": "doc1", "title": "My Title", "text": "The body text to index."}]
```

### Running the tests

```bash
python -m unittest discover -s tests -v
```

28 tests cover the tokenizer's stemming edge cases (plurals, "-ing"/"-ed" with doubled consonants and silent-e restoration like "hoping" vs. "hopping"), inverted index construction/removal/persistence, BM25 score ordering and IDF weighting, and the query parser's boolean logic, parentheses, and phrase matching.

## Design decisions

- **Positional postings over plain counts.** Storing `[positions]` per (term, doc) instead of just a frequency costs more memory but is what enables phrase queries to be answered by intersecting position lists instead of re-tokenizing every candidate document on every query.
- **BM25's `+1` inside the IDF log.** The textbook BM25 IDF formula can go negative when a term appears in more than half the corpus. Adding `+ 1` inside the log (the fix used by Lucene and BM25+) keeps IDF non-negative without changing ranking behavior for typical corpora.
- **Query evaluation and ranking are separate concerns.** `QueryParser.evaluate()` only decides *which* documents match a boolean/phrase expression; it hands that candidate set to `BM25Scorer` to decide the *order*. This mirrors how production systems separate "match" from "rank" and keeps each piece independently testable.
- **Stemmer restores dropped letters instead of just chopping suffixes.** A naive stemmer turns "running" into "runn" (wrong) because English doubles the final consonant before "-ing". The stemmer here detects that doubling and undoes it, and separately restores a silent "e" dropped before "-ing"/"-ed" (so "hoping" and "hopping" stem to different, correct roots: "hope" and "hop").
- **JSON for index persistence.** Keeps `save`/`load` dependency-free and human-inspectable at the cost of a slightly larger file than a binary format  -  a reasonable trade-off for a project meant to be read and understood, not run at scale.

## Project structure

```
search_engine/
  tokenizer.py   tokenization + stemming
  index.py       InvertedIndex: construction, postings, persistence
  bm25.py        BM25Scorer: idf, score, search
  query.py       QueryParser: boolean/phrase grammar and evaluation
  cli.py         argparse CLI: build / search / repl
sample_data/
  docs.json      10 short sample documents for trying the CLI
tests/
  test_search_engine.py   28 unit tests across all four modules
```

## Known limitations

This is a single-process, in-memory index meant for learning and small corpora (thousands, not millions, of documents)  -  there's no on-disk B-tree, no sharding, and no incremental index merging like a production search engine would need. The stemmer is a light rule-based one, not a full Porter/Snowball implementation, so it will occasionally under- or over-stem uncommon words.
