"""Command line interface for the search engine.

Usage:
    python -m search_engine.cli build sample_data/docs.json --out index.json
    python -m search_engine.cli search index.json "machine learning" --mode bm25
    python -m search_engine.cli search index.json "cat AND NOT dog" --mode boolean
    python -m search_engine.cli repl index.json
"""

import argparse
import json
import sys
from typing import List

from .bm25 import BM25Scorer
from .index import InvertedIndex
from .query import QueryParser, QuerySyntaxError


def cmd_build(args: argparse.Namespace) -> None:
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    index = InvertedIndex.build_from_corpus(corpus)
    index.save(args.out)
    print(f"Indexed {index.document_count()} documents, "
          f"{len(list(index.vocabulary()))} unique terms -> {args.out}")


def _print_results(index: InvertedIndex, ranked, query: str) -> None:
    if not ranked:
        print(f'No results for "{query}"')
        return
    print(f'Results for "{query}":')
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        title = index.get_title(doc_id)
        snippet = (index.get_text(doc_id) or "")[:100].replace("\n", " ")
        print(f"  {rank}. [{doc_id}] {title}  (score={score:.4f})")
        print(f"     {snippet}...")


def cmd_search(args: argparse.Namespace) -> None:
    index = InvertedIndex.load(args.index)
    _run_query(index, args.query, args.mode, args.top_k)


def _run_query(index: InvertedIndex, query: str, mode: str, top_k: int) -> None:
    if mode == "bm25":
        scorer = BM25Scorer(index)
        ranked = scorer.search(query, top_k=top_k)
        _print_results(index, ranked, query)
    elif mode == "boolean":
        parser = QueryParser(index)
        try:
            doc_ids = parser.evaluate(query)
        except QuerySyntaxError as exc:
            print(f"Query error: {exc}", file=sys.stderr)
            return
        scorer = BM25Scorer(index)
        query_terms = parser.parse(query)
        ranked = sorted(
            ((doc_id, scorer.score(query_terms, doc_id)) for doc_id in doc_ids),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
        _print_results(index, ranked, query)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def cmd_repl(args: argparse.Namespace) -> None:
    index = InvertedIndex.load(args.index)
    print(f"Loaded index with {index.document_count()} documents. "
          f"Mode: {args.mode}. Type a query, or 'quit' to exit.")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue
        _run_query(index, query, args.mode, args.top_k)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="search_engine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build an index from a JSON corpus file")
    p_build.add_argument("corpus", help="Path to a JSON file: list of {id, title, text}")
    p_build.add_argument("--out", default="index.json", help="Where to save the index")
    p_build.set_defaults(func=cmd_build)

    p_search = sub.add_parser("search", help="Run a single query against a saved index")
    p_search.add_argument("index", help="Path to a saved index (from `build`)")
    p_search.add_argument("query", help="Query string")
    p_search.add_argument("--mode", choices=["bm25", "boolean"], default="bm25")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    p_repl = sub.add_parser("repl", help="Interactive query loop against a saved index")
    p_repl.add_argument("index", help="Path to a saved index (from `build`)")
    p_repl.add_argument("--mode", choices=["bm25", "boolean"], default="bm25")
    p_repl.add_argument("--top-k", type=int, default=10)
    p_repl.set_defaults(func=cmd_repl)

    return parser


def main(argv: List[str] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
