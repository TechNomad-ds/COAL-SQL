#!/usr/bin/env python3
"""
Search the skeleton FAISS index.

Given one or more query skeletons, retrieve the top-k most similar skeletons
from the index and return their associated original SQL queries.

Usage:
    # Interactive mode
    python search_index.py

    # Single query
    python search_index.py --query "select _ from _ where _"

    # Batch mode: read queries from a JSON file (list of skeleton strings)
    python search_index.py --query_file queries.json --output results.json
"""

import argparse
import json
import os
import time

import faiss
import numpy as np
import torch
from vllm import LLM

# ─── Paths ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./data/skeleton_index"
MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")

INDEX_FILE = os.path.join(OUTPUT_DIR, "skeleton.index")
METADATA_FILE = os.path.join(OUTPUT_DIR, "skeleton_metadata.json")

class SkeletonSearcher:
    """Load FAISS index + metadata, encode queries, search."""

    def __init__(self, gpu_id: int = 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"Loading FAISS index from {INDEX_FILE} ...")
        self.index = faiss.read_index(INDEX_FILE)
        print(f"  Index loaded, total vectors: {self.index.ntotal}")

        print(f"Loading metadata from {METADATA_FILE} ...")
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"  Metadata loaded, {len(self.metadata)} entries.")

        print(f"Loading embedding model from {MODEL_PATH} ...")
        t0 = time.time()
        self.model = LLM(model=MODEL_PATH, task="embed", dtype="float16",
                         max_model_len=2048, gpu_memory_utilization=0.5,
                         enforce_eager=True,
                         max_num_seqs=1,
                         max_num_batched_tokens=8192)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

    def search(self, query_skeletons: list[str], top_k: int = 5) -> list[list[dict]]:
        """Search for similar skeletons.

        Args:
            query_skeletons: List of skeleton strings to search for.
            top_k: Number of top results to return per query.

        Returns:
            List of lists. Each inner list contains top_k dicts with keys:
              - skeleton: the matched skeleton
              - score: cosine similarity score
              - count: number of SQLs with this skeleton
              - entries: list of {idx, db_id, sql, question}
        """
        # Encode queries WITHOUT instruction prefix (symmetric with document side)
        outputs = self.model.embed(query_skeletons)
        query_embs = np.array(
            [o.outputs.embedding for o in outputs], dtype=np.float32
        )
        faiss.normalize_L2(query_embs)

        # Search
        scores, indices = self.index.search(query_embs, top_k)

        # Assemble results
        all_results = []
        for q_idx in range(len(query_skeletons)):
            results = []
            for rank in range(top_k):
                faiss_idx = indices[q_idx][rank]
                if faiss_idx == -1:
                    continue
                meta = self.metadata[faiss_idx]
                results.append({
                    "rank": rank + 1,
                    "skeleton": meta["skeleton"],
                    "score": float(scores[q_idx][rank]),
                    "count": meta["count"],
                    "entries": meta["entries"],
                })
            all_results.append(results)

        return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None,
                        help="A single skeleton to search for")
    parser.add_argument("--query_file", type=str, default=None,
                        help="JSON file with a list of skeleton strings")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for batch results")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top results per query")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Which GPU to use")
    args = parser.parse_args()

    searcher = SkeletonSearcher(gpu_id=args.gpu_id)

    if args.query:
        # Single query mode
        results = searcher.search([args.query], top_k=args.top_k)
        print(f"\nQuery: {args.query}")
        print("-" * 60)
        for r in results[0]:
            print(f"  Rank {r['rank']} (score={r['score']:.4f}, {r['count']} SQLs):")
            print(f"    Skeleton: {r['skeleton']}")
            # Show first 3 SQL examples
            for entry in r["entries"][:3]:
                sql_preview = entry["sql"].replace("\n", " ")[:100]
                print(f"    SQL: {sql_preview}...")
            if r["count"] > 3:
                print(f"    ... and {r['count'] - 3} more")
            print()

    elif args.query_file:
        # Batch mode
        with open(args.query_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        print(f"Searching {len(queries)} queries ...")
        t0 = time.time()
        results = searcher.search(queries, top_k=args.top_k)
        elapsed = time.time() - t0
        print(f"  Search done in {elapsed:.1f}s")

        output_data = []
        for q, res in zip(queries, results):
            output_data.append({
                "query_skeleton": q,
                "results": res,
            })

        output_path = args.output or "search_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")

    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("SKELETON SEARCH - Interactive Mode")
        print("Type a skeleton and press Enter. Type 'quit' to exit.")
        print("=" * 60)
        while True:
            try:
                query = input("\nSkeleton> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if query.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            if not query:
                continue

            results = searcher.search([query], top_k=args.top_k)
            print(f"\nResults for: {query}")
            print("-" * 60)
            for r in results[0]:
                print(f"  Rank {r['rank']} (score={r['score']:.4f}, {r['count']} SQLs):")
                print(f"    Skeleton: {r['skeleton']}")
                for entry in r["entries"][:3]:
                    sql_preview = entry["sql"].replace("\n", " ")[:100]
                    print(f"    SQL: {sql_preview}...")
                if r["count"] > 3:
                    print(f"    ... and {r['count'] - 3} more")
                print()


if __name__ == "__main__":
    main()
