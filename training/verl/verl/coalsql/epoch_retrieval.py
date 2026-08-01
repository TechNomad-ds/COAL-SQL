"""
EpochRetriever: Lightweight retrieval augmentation for COAL-SQL training.

At the end of each epoch, this module retrieves similar questions from the
question bank for failed training samples, using pre-computed skeleton
embeddings and a FAISS index.

Design principles:
  - No embedding model is loaded at training time; all embeddings are
    pre-computed offline and stored as .npy files.
  - FAISS index runs on CPU, consuming minimal GPU memory.
  - Supports retrieval for both original training samples and previously
    retrieved question-bank samples (recursive retrieval).
  - (Optional) Error-aware LLM rerank for structural errors (id 17-30).
"""

import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class EpochRetriever:
    """
    Lightweight epoch-level retriever for curriculum augmentation.

    Loads pre-built FAISS index + metadata + pre-encoded query embeddings,
    and provides a ``retrieve_for_failed()`` method that returns new training
    samples from the question bank for failed (solve-none) samples.

    Args:
        faiss_index_path:      Path to the FAISS index file (skeleton.index).
        metadata_path:         Path to skeleton_metadata.json.
        question_bank_parquet: Path to question_bank_filtered.parquet.
        train_embeddings_path: Path to train_skeleton_embeddings.npy.
        train_idx_mapping_path: Path to bird_idx_to_parquet_row.json.
        qb_embeddings_path:    Path to qb_skeleton_query_embeddings.npy.
        top_k:                 Number of FAISS top-k skeleton templates to retrieve.
        num_per_epoch:         Target number of new samples per epoch.
        rank_weights:          Sampling weights for each rank (must sum <= 1.0).
        seed:                  Random seed for reproducible sampling.
        random_ratio:          Fraction of num_per_epoch to sample randomly from
                               the full question bank (0.0 = pure retrieval,
                               0.3 = 30% random + 70% retrieval, 1.0 = pure random).
    """

    def __init__(
        self,
        faiss_index_path: str,
        metadata_path: str,
        question_bank_parquet: str,
        train_embeddings_path: str,
        train_idx_mapping_path: str,
        qb_embeddings_path: str,
        top_k: int = 5,
        num_per_epoch: int = 2000,
        rank_weights: Optional[List[float]] = None,
        seed: int = 42,
        random_ratio: float = 0.0,
        error_aware_config: Optional[Dict[str, Any]] = None,
    ):
        self.top_k = top_k
        self.num_per_epoch = num_per_epoch
        self.rank_weights = rank_weights or [0.40, 0.25, 0.20, 0.10, 0.05]
        self.random_ratio = max(0.0, min(1.0, random_ratio))  # clamp to [0, 1]
        self.rng = random.Random(seed)

        assert len(self.rank_weights) == self.top_k, (
            f"rank_weights length ({len(self.rank_weights)}) must match top_k ({self.top_k})"
        )

        # ── Load FAISS index (CPU) ──────────────────────────────────────
        logger.info("Loading FAISS index from %s ...", faiss_index_path)
        self.index = faiss.read_index(faiss_index_path)
        logger.info("  FAISS index loaded: %d vectors, dim=%d", self.index.ntotal, self.index.d)

        # ── Load skeleton metadata ──────────────────────────────────────
        logger.info("Loading skeleton metadata from %s ...", metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict] = json.load(f)
        logger.info("  Loaded %d skeleton templates", len(self.metadata))

        # ── Load question bank parquet ──────────────────────────────────
        logger.info("Loading question bank from %s ...", question_bank_parquet)
        self.question_bank: pd.DataFrame = pl.read_parquet(question_bank_parquet).to_pandas()
        logger.info("  Loaded %d question bank entries", len(self.question_bank))

        # ── Load pre-encoded embeddings ─────────────────────────────────
        logger.info("Loading train embeddings from %s ...", train_embeddings_path)
        self.train_embeddings: np.ndarray = np.load(train_embeddings_path)
        logger.info("  Train embeddings shape: %s", self.train_embeddings.shape)

        logger.info("Loading QB query embeddings from %s ...", qb_embeddings_path)
        self.qb_embeddings: np.ndarray = np.load(qb_embeddings_path)
        logger.info("  QB query embeddings shape: %s", self.qb_embeddings.shape)

        # ── Load bird_idx → parquet_row mapping ─────────────────────────
        logger.info("Loading train idx mapping from %s ...", train_idx_mapping_path)
        with open(train_idx_mapping_path, "r", encoding="utf-8") as f:
            raw_mapping = json.load(f)
        # Keys in JSON are strings; convert to str→int mapping
        self.bird_idx_to_row: Dict[str, int] = {str(k): int(v) for k, v in raw_mapping.items()}
        logger.info("  Loaded %d bird_idx → parquet_row mappings", len(self.bird_idx_to_row))

        # ── Track cumulative added indices across epochs ────────────────
        self._added_indices: Set[int] = set()

        # ── Initialize Error-Aware Retrieval (optional) ────────────────
        self.error_aware_retriever = None
        if error_aware_config and error_aware_config.get("enable", False):
            from .error_aware_retrieval import ErrorAwareRetriever

            # Resolve prompt/taxonomy paths relative to this module
            module_dir = os.path.dirname(os.path.abspath(__file__))
            default_taxonomy = os.path.join(module_dir, "prompts", "taxonomy.json")
            default_diagnosis_prompt = os.path.join(module_dir, "prompts", "error_diagnosis.txt")
            default_rerank_prompt = os.path.join(module_dir, "prompts", "error_aware_rerank.txt")

            taxonomy_path = error_aware_config.get("taxonomy_path") or default_taxonomy
            diagnosis_prompt_path = error_aware_config.get("diagnosis_prompt_path") or default_diagnosis_prompt
            rerank_prompt_path = error_aware_config.get("rerank_prompt_path") or default_rerank_prompt

            self.error_aware_retriever = ErrorAwareRetriever(
                config=error_aware_config,
                taxonomy_path=taxonomy_path,
                diagnosis_prompt_path=diagnosis_prompt_path,
                rerank_prompt_path=rerank_prompt_path,
            )
            # Budget cap: error_aware selections cannot exceed this fraction of num_per_epoch
            self.error_aware_ratio = error_aware_config.get("error_aware_ratio", 0.3)
            # Max candidates per LLM rerank call
            self.rerank_batch_size = error_aware_config.get("rerank_batch_size", 20)
            # How many times the quota to sample as rerank candidates (e.g. 3 means
            # for a rank with quota=20, randomly draw 60 candidates for LLM rerank;
            # if rerank selects fewer than 20, fill the rest randomly from the rank pool)
            self.rerank_multiplier = error_aware_config.get("rerank_multiplier", 3)
            logger.info("[Retrieval] Error-Aware retrieval enabled. error_aware_ratio=%.2f, "
                       "rerank_batch_size=%d, rerank_multiplier=%d",
                       self.error_aware_ratio, self.rerank_batch_size, self.rerank_multiplier)

        logger.info("EpochRetriever initialized. top_k=%d, num_per_epoch=%d, random_ratio=%.2f, weights=%s",
                     self.top_k, self.num_per_epoch, self.random_ratio, self.rank_weights)

    @property
    def added_indices(self) -> Set[int]:
        """Return the set of question bank indices already added to training."""
        return self._added_indices

    def _resolve_embedding(self, split: str, index) -> Optional[np.ndarray]:
        """
        Look up the pre-computed query embedding for a failed sample.

        Args:
            split:  'train' or 'question_bank'
            index:  For train split, this is the BIRD index (needs mapping).
                    For question_bank split, this is the QB row index.

        Returns:
            1-D numpy array of shape (dim,), or None if mapping fails.
        """
        if split == "train":
            parquet_row = self.bird_idx_to_row.get(str(index))
            if parquet_row is None:
                logger.warning("bird_idx %s not found in mapping, skipping.", index)
                return None
            if parquet_row >= len(self.train_embeddings):
                logger.warning("parquet_row %d out of range for train embeddings (%d), skipping.",
                               parquet_row, len(self.train_embeddings))
                return None
            return self.train_embeddings[parquet_row]
        elif split == "question_bank":
            idx = int(index)
            if idx >= len(self.qb_embeddings):
                logger.warning("QB index %d out of range for QB embeddings (%d), skipping.",
                               idx, len(self.qb_embeddings))
                return None
            return self.qb_embeddings[idx]
        else:
            logger.warning("Unknown split '%s' for index %s, skipping.", split, index)
            return None

    def _deduplicate_queries(
        self, failed_samples: List[Tuple[str, int]]
    ) -> Tuple[np.ndarray, int]:
        """
        Collect and deduplicate query embeddings from failed samples.

        Multiple failed samples may share the same skeleton (hence the same
        embedding). We deduplicate to avoid redundant FAISS queries, but
        we track multiplicity so that frequently-failed skeletons contribute
        more candidates.

        Returns:
            query_embs: np.ndarray of shape (N_unique, dim), L2-normalized.
            n_skipped:  Number of samples skipped due to mapping failures.
        """
        seen_keys = {}  # (split, index) → embedding
        skipped = 0

        for split, index in failed_samples:
            key = (split, index)
            if key in seen_keys:
                continue  # same sample, skip
            emb = self._resolve_embedding(split, index)
            if emb is None:
                skipped += 1
                continue
            seen_keys[key] = emb

        if not seen_keys:
            return np.empty((0, self.index.d), dtype=np.float32), skipped

        query_embs = np.stack(list(seen_keys.values()), axis=0).astype(np.float32)
        faiss.normalize_L2(query_embs)
        return query_embs, skipped

    def _collect_rank_candidates(
        self, faiss_ids: np.ndarray, exclude_indices: Set[int]
    ) -> List[Set[int]]:
        """
        From FAISS search results, collect candidate QB indices per rank.

        Args:
            faiss_ids: shape (N_queries, top_k), FAISS result IDs.
            exclude_indices: QB indices to exclude (already used).

        Returns:
            List of sets, one per rank. rank_candidates[j] = set of QB indices
            from all queries' j-th nearest skeleton template.
        """
        rank_candidates: List[Set[int]] = [set() for _ in range(self.top_k)]

        for i in range(faiss_ids.shape[0]):
            for j in range(self.top_k):
                fid = int(faiss_ids[i][j])
                if fid < 0:
                    continue  # FAISS returns -1 for missing results

                # Each skeleton template maps to multiple entries (actual questions)
                entries = self.metadata[fid].get("entries", [])
                for entry in entries:
                    bank_idx = entry["idx"]
                    if bank_idx not in exclude_indices:
                        rank_candidates[j].add(bank_idx)

        return rank_candidates

    def _sample_random(self, n: int, extra_exclude: Set[int] = None) -> List[int]:
        """
        Randomly sample ``n`` QB indices from the full question bank.

        Args:
            n:              Number of random samples to draw.
            extra_exclude:  Additional indices to exclude (on top of _added_indices).

        Returns:
            List of selected QB indices (no duplicates, disjoint from excluded sets).
        """
        if n <= 0:
            return []

        exclude = self._added_indices.copy()
        if extra_exclude:
            exclude |= extra_exclude

        all_indices = set(range(len(self.question_bank)))
        available = list(all_indices - exclude)

        if len(available) <= n:
            logger.warning("[Retrieval-Random] Only %d available (requested %d), taking all.",
                           len(available), n)
            self.rng.shuffle(available)
            return available

        return self.rng.sample(available, n)

    def _sample_by_rank(
        self, rank_candidates: List[Set[int]], budget: Optional[int] = None
    ) -> List[int]:
        """
        Sample up to ``budget`` QB indices from rank-stratified candidates.

        Args:
            rank_candidates: List of candidate sets, one per rank.
            budget:          Max samples to select. Defaults to self.num_per_epoch.

        Strategy:
        1. For each rank, compute quota = weight * budget.
        2. Randomly sample min(quota, available) from that rank's candidates.
        3. If a rank has surplus capacity, its unused quota carries over to
           subsequent ranks.
        4. After all ranks, if budget remains, fill from all remaining candidates.

        Returns:
            List of selected QB indices (no duplicates).
        """
        if budget is None:
            budget = self.num_per_epoch

        selected: List[int] = []
        selected_set: Set[int] = set()
        remaining_budget = budget

        for rank in range(self.top_k):
            if remaining_budget <= 0:
                break

            quota = int(self.rank_weights[rank] * budget)
            # Remove already-selected from this rank's pool
            available = list(rank_candidates[rank] - selected_set)
            self.rng.shuffle(available)

            take = min(quota, len(available), remaining_budget)
            batch = available[:take]
            selected.extend(batch)
            selected_set.update(batch)
            remaining_budget -= take

        # ── Fill remaining budget from all unused candidates ────────────
        if remaining_budget > 0:
            all_remaining = set()
            for rc in rank_candidates:
                all_remaining |= rc
            all_remaining -= selected_set
            all_remaining_list = list(all_remaining)
            self.rng.shuffle(all_remaining_list)
            fill = all_remaining_list[:remaining_budget]
            selected.extend(fill)
            selected_set.update(fill)

        return selected

    def retrieve_for_failed(
        self,
        failed_samples: List[Tuple[str, int]],
        failed_sample_details: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Main entry point: retrieve new training samples for failed (solve-none) items.

        Args:
            failed_samples: List of (split, index) tuples.
                - split='train': index is BIRD train row index.
                - split='question_bank': index is QB filtered row index.
            failed_sample_details: Optional list of dicts for error-aware retrieval.
                Each dict should contain:
                - split: str
                - index: int
                - context: str (schema + question)
                - correct_sql: str
                - error_response: str (model's best/first incorrect response)
                Only needed when error_aware is enabled.

        Returns:
            A DataFrame (subset of question_bank_filtered.parquet) containing
            the newly selected samples. Returns empty DataFrame if nothing
            to retrieve.
        """
        if not failed_samples:
            logger.info("[Retrieval] No failed samples, skipping retrieval.")
            return pd.DataFrame()

        n_total_failed = len(failed_samples)

        # ── Step 1: Collect and deduplicate query embeddings ────────────
        query_embs, n_skipped = self._deduplicate_queries(failed_samples)
        n_queries = query_embs.shape[0]

        if n_queries == 0:
            logger.warning("[Retrieval] All %d failed samples skipped (mapping failures).", n_total_failed)
            return pd.DataFrame()

        logger.info("[Retrieval] %d failed samples → %d unique queries (%d skipped)",
                     n_total_failed, n_queries, n_skipped)

        # ── Step 2: FAISS search ────────────────────────────────────────
        scores, faiss_ids = self.index.search(query_embs, self.top_k)
        logger.info("[Retrieval] FAISS search completed. Score range: [%.4f, %.4f]",
                     float(scores.min()), float(scores.max()))

        # ── Step 3: Collect candidates per rank ─────────────────────────
        exclude = self._added_indices.copy()
        rank_candidates = self._collect_rank_candidates(faiss_ids, exclude)

        candidate_counts = [len(rc) for rc in rank_candidates]
        total_candidates = len(set().union(*rank_candidates))
        logger.info("[Retrieval] Candidates per rank: %s (total unique: %d)",
                     candidate_counts, total_candidates)

        if total_candidates == 0:
            logger.warning("[Retrieval] No candidates available after exclusion.")
            return pd.DataFrame()

        # ── Step 3.5: Error-Aware LLM Rerank (optional) ────────────────
        # If error-aware is enabled and we have sample details, diagnose errors
        # and rerank candidates for structural error types (id 17-30).
        error_aware_selected: Set[int] = set()
        if self.error_aware_retriever is not None and failed_sample_details:
            error_aware_selected = self._run_error_aware_rerank(
                failed_sample_details=failed_sample_details,
                rank_candidates=rank_candidates,
                exclude=exclude,
            )
            if error_aware_selected:
                logger.info("[Retrieval] Error-aware rerank selected %d additional samples",
                           len(error_aware_selected))

        # ── Step 4: Split budget between random and retrieval sampling ──
        # Reduce retrieval budget by the number of error-aware selections.
        # The cap in _run_error_aware_rerank guarantees len(error_aware_selected) <= ratio * num_per_epoch,
        # so effective_budget is always >= (1 - ratio) * num_per_epoch.
        effective_budget = max(0, self.num_per_epoch - len(error_aware_selected))
        n_random = int(effective_budget * self.random_ratio)
        n_retrieval = effective_budget - n_random

        # 4a. Random sampling from full question bank
        random_indices = self._sample_random(n_random, extra_exclude=error_aware_selected)
        random_set = set(random_indices)
        logger.info("[Retrieval] Random sampling: requested %d, got %d", n_random, len(random_indices))

        # 4b. Retrieval-based stratified sampling (exclude random + error_aware samples)
        # Remove randomly-sampled and error-aware indices from rank candidates
        combined_exclude = random_set | error_aware_selected
        rank_candidates_cleaned = [rc - combined_exclude for rc in rank_candidates]
        retrieval_indices = self._sample_by_rank(rank_candidates_cleaned, budget=n_retrieval)
        logger.info("[Retrieval] Retrieval sampling: requested %d, got %d", n_retrieval, len(retrieval_indices))

        # Merge all parts: error_aware + random + retrieval
        selected_indices = list(error_aware_selected) + random_indices + retrieval_indices
        logger.info("[Retrieval] Total selected: %d (error_aware=%d + random=%d + retrieval=%d) / %d target",
                     len(selected_indices), len(error_aware_selected),
                     len(random_indices), len(retrieval_indices),
                     self.num_per_epoch)

        if not selected_indices:
            return pd.DataFrame()

        # ── Step 5: Update global tracking ──────────────────────────────
        self._added_indices.update(selected_indices)

        # ── Step 6: Extract rows from question bank ─────────────────────
        new_samples_df = self.question_bank.iloc[selected_indices].copy()

        # Log retrieval summary
        logger.info("[Retrieval] Summary: %d failed → %d queries → %d candidates → %d selected | "
                     "Total added so far: %d / %d QB",
                     n_total_failed, n_queries, total_candidates, len(new_samples_df),
                     len(self._added_indices), len(self.question_bank))

        return new_samples_df

    def _run_error_aware_rerank(
        self,
        failed_sample_details: List[Dict],
        rank_candidates: List[Set[int]],
        exclude: Set[int],
    ) -> Set[int]:
        """
        Run the full error-aware rerank pipeline.

        The algorithm mirrors the standard skeleton retrieval (rank_weights-based
        stratified sampling) but replaces random sampling with LLM rerank at the
        sample level:
          1. Diagnose error types for all failed samples
          2. Group structural-error samples by error type
          3. FAISS top_k → preserve per-rank skeleton info
          4. LLM skeleton filtering (per error type)
          5. For each error type, distribute type_budget across ranks using
             rank_weights, then LLM rerank within each rank (batched)
          6. Safety cap

        Args:
            failed_sample_details: List of sample dicts with context info
            rank_candidates: Standard rank candidates from FAISS search
            exclude: QB indices to exclude (already used)

        Returns:
            Set of QB indices selected via error-aware rerank (capped by budget)
        """
        retriever = self.error_aware_retriever
        rerank_batch_size = getattr(self, 'rerank_batch_size', 20)

        # ── Step A: Diagnose errors ────────────────────────────────────────
        diagnosis_results = retriever.diagnose_errors(failed_sample_details)

        # Count how many have structural errors
        structural_samples = {
            k: v for k, v in diagnosis_results.items() if v is not None
        }
        logger.info("[ErrorAware] Diagnosis: %d/%d have structural errors (id 17-30)",
                   len(structural_samples), len(diagnosis_results))

        if not structural_samples:
            return set()

        # ── Step B: Group by error type & collect embeddings ───────────────
        error_type_to_keys: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for key, error_type in structural_samples.items():
            error_type_to_keys[error_type].append(key)

        logger.info("[ErrorAware] %d unique error types: %s",
                   len(error_type_to_keys), list(error_type_to_keys.keys()))

        # Collect embeddings for structural-error samples
        structural_query_embs = []
        structural_query_keys = []
        for sample in failed_sample_details:
            key = (sample['split'], sample['index'])
            if key in structural_samples:
                emb = self._resolve_embedding(sample['split'], sample['index'])
                if emb is not None:
                    structural_query_embs.append(emb)
                    structural_query_keys.append(key)

        if not structural_query_embs:
            logger.warning("[ErrorAware] No valid embeddings for structural-error samples.")
            return set()

        # ── Step C: FAISS search — preserve rank information ──────────────
        query_embs = np.stack(structural_query_embs, axis=0).astype(np.float32)
        faiss.normalize_L2(query_embs)
        _, faiss_ids = self.index.search(query_embs, self.top_k)

        # Build per-error-type, per-rank skeleton sets:
        #   error_type_rank_skel_ids[error_type][rank] = set of skeleton fids
        error_type_rank_skel_ids: Dict[str, List[Set[int]]] = {}
        for error_type in error_type_to_keys:
            error_type_rank_skel_ids[error_type] = [set() for _ in range(self.top_k)]

        for i, key in enumerate(structural_query_keys):
            error_type = structural_samples[key]
            for j in range(self.top_k):
                fid = int(faiss_ids[i][j])
                if fid >= 0:
                    error_type_rank_skel_ids[error_type][j].add(fid)

        # Also build flat set per error type for skeleton filtering
        error_type_to_all_skel_ids: Dict[str, Set[int]] = {}
        for error_type, rank_sets in error_type_rank_skel_ids.items():
            all_skel = set()
            for s in rank_sets:
                all_skel |= s
            error_type_to_all_skel_ids[error_type] = all_skel

        # ── Step D: LLM skeleton filtering (per error type, in parallel) ──
        filter_tasks = []
        for error_type, skel_ids in error_type_to_all_skel_ids.items():
            skeleton_infos = []
            for fid in sorted(skel_ids):
                meta = self.metadata[fid]
                skeleton_str = meta.get("skeleton", "(no skeleton)")
                skeleton_infos.append({'fid': fid, 'skeleton': skeleton_str})
            filter_tasks.append({
                'error_type': error_type,
                'skeleton_infos': skeleton_infos,
            })

        filter_results = retriever.batch_filter_skeletons(filter_tasks)

        # Convert filter results to sets for fast lookup
        filter_fid_sets: Dict[str, Set[int]] = {}
        for error_type, fid_list in filter_results.items():
            filter_fid_sets[error_type] = set(fid_list) if fid_list else set()

        # ── Step E: Per-type budget (frequency-weighted) ──────────────────
        max_error_aware = int(self.num_per_epoch * self.error_aware_ratio)
        total_structural = sum(len(keys) for keys in error_type_to_keys.values())

        type_budgets: Dict[str, int] = {}
        for error_type, keys in error_type_to_keys.items():
            weight = len(keys) / total_structural
            budget = max(1, int(weight * max_error_aware))
            type_budgets[error_type] = budget

        # Safety: scale down if sum exceeds max_error_aware
        budget_sum = sum(type_budgets.values())
        if budget_sum > max_error_aware:
            scale = max_error_aware / budget_sum
            type_budgets = {k: max(1, int(v * scale)) for k, v in type_budgets.items()}

        logger.info("[ErrorAware] Per-type budgets (max_total=%d, structural_samples=%d): %s",
                   max_error_aware, total_structural, type_budgets)

        # ── Step F: For each error type, build per-rank candidate pools,
        #            distribute budget by rank_weights, then rerank + random fill ──
        #
        # Algorithm (mirrors _sample_by_rank but uses LLM rerank instead of
        # pure random):
        #   For each rank:
        #     1. Compute quota = rank_weight × type_budget
        #     2. Randomly draw min(multiplier × quota, pool_size) candidates
        #     3. Send them to LLM rerank (batched by rerank_batch_size)
        #     4. Take up to quota from rerank results
        #     5. If rerank selected < quota, randomly fill from remaining pool
        #   This guarantees quota is always met (if pool is large enough),
        #   and LLM calls are bounded by multiplier × quota / batch_size per rank.

        rerank_multiplier = getattr(self, 'rerank_multiplier', 3)
        all_rerank_tasks = []

        # Pre-compute per-error-type, per-rank candidate pools and quotas
        # so we can build all rerank tasks first, run them in parallel, then assemble.
        type_rank_info: Dict[str, List[Dict]] = {}  # error_type -> [{pool, quota, ...}]

        for error_type, rank_skel_sets in error_type_rank_skel_ids.items():
            type_budget = type_budgets.get(error_type, 1)
            passed_fids = filter_fid_sets.get(error_type, set())

            # If filtering removed everything, fallback to all skeletons
            if not passed_fids:
                passed_fids = error_type_to_all_skel_ids.get(error_type, set())
                logger.debug("[ErrorAware] No skeletons pass filter for '%s', using all", error_type)

            # Build per-rank candidate pools (only from filtered skeletons)
            rank_candidate_pools: List[Set[int]] = [set() for _ in range(self.top_k)]
            for rank in range(self.top_k):
                for fid in rank_skel_sets[rank]:
                    if fid not in passed_fids:
                        continue  # skeleton filtered out
                    entries = self.metadata[fid].get("entries", [])
                    for entry in entries:
                        bank_idx = entry["idx"]
                        if bank_idx not in exclude:
                            rank_candidate_pools[rank].add(bank_idx)

            rank_infos = []
            selected_so_far: Set[int] = set()  # track cross-rank dedup within this type

            for rank in range(self.top_k):
                quota = int(self.rank_weights[rank] * type_budget)
                if quota <= 0:
                    rank_infos.append({'quota': 0, 'pool': set(), 'rerank_sample': []})
                    continue

                # Remove already-selected from this rank's pool
                available = rank_candidate_pools[rank] - selected_so_far
                if not available:
                    rank_infos.append({'quota': quota, 'pool': set(), 'rerank_sample': []})
                    continue

                # Randomly draw multiplier × quota candidates for LLM rerank
                available_list = list(available)
                self.rng.shuffle(available_list)
                n_draw = min(rerank_multiplier * quota, len(available_list))
                rerank_sample_indices = available_list[:n_draw]

                # Build candidate entries for the drawn sample
                candidate_entries = self._build_candidate_entries(set(rerank_sample_indices))

                # Create rerank tasks (batched by rerank_batch_size)
                for batch_start in range(0, len(candidate_entries), rerank_batch_size):
                    batch = candidate_entries[batch_start:batch_start + rerank_batch_size]
                    all_rerank_tasks.append({
                        'key': (error_type, rank, batch_start),
                        'target_error_type': error_type,
                        'candidate_entries': batch,
                        'budget': quota,  # LLM can select up to quota per batch
                    })

                # Record info for later assembly
                rank_infos.append({
                    'quota': quota,
                    'pool': available,            # full pool for random fill
                    'rerank_sample': set(rerank_sample_indices),  # what was sent to LLM
                })
                # Optimistically reserve — actual dedup happens in assembly
                selected_so_far |= set(rerank_sample_indices)

            type_rank_info[error_type] = rank_infos

            total_rank_candidates = sum(len(p) for p in rank_candidate_pools)
            logger.info(
                "[ErrorAware] Error type '%s': budget=%d, rank_candidates=%s (total=%d), "
                "rerank_tasks=%d",
                error_type, type_budget,
                [len(p) for p in rank_candidate_pools], total_rank_candidates,
                sum(1 for t in all_rerank_tasks if t.get('target_error_type') == error_type),
            )

        if not all_rerank_tasks:
            logger.warning("[ErrorAware] No rerank tasks after skeleton filtering.")
            return set()

        # ── Step G: Run all rerank tasks in parallel ──────────────────────
        rerank_results = retriever.batch_rerank(all_rerank_tasks)

        # ── Step H: Assemble results — rerank first, random fill remainder ─
        # Build lookup: (error_type, rank, batch_start) -> list of indices
        task_results: Dict[Tuple, List[int]] = {}
        for key, indices in rerank_results.items():
            task_results[key] = indices

        selected = set()
        for error_type in error_type_to_keys:
            type_budget = type_budgets.get(error_type, 1)
            type_selected: List[int] = []
            type_selected_set: Set[int] = set()
            remaining_type_budget = type_budget
            rank_infos = type_rank_info.get(error_type, [])

            for rank in range(min(self.top_k, len(rank_infos))):
                if remaining_type_budget <= 0:
                    break

                info = rank_infos[rank]
                rank_quota = min(info['quota'], remaining_type_budget)
                if rank_quota <= 0:
                    continue

                # Collect rerank results for this (error_type, rank)
                rank_reranked: List[int] = []
                batch_start = 0
                while True:
                    key = (error_type, rank, batch_start)
                    if key not in task_results:
                        break
                    for idx in task_results[key]:
                        if idx not in type_selected_set:
                            rank_reranked.append(idx)
                    batch_start += rerank_batch_size

                # Take up to rank_quota from rerank results
                take_from_rerank = rank_reranked[:rank_quota]
                for idx in take_from_rerank:
                    type_selected.append(idx)
                    type_selected_set.add(idx)
                    remaining_type_budget -= 1

                n_reranked = len(take_from_rerank)

                # Random fill if rerank didn't produce enough
                shortfall = rank_quota - n_reranked
                if shortfall > 0 and info['pool']:
                    fill_pool = info['pool'] - type_selected_set
                    if fill_pool:
                        fill_list = list(fill_pool)
                        self.rng.shuffle(fill_list)
                        fill = fill_list[:shortfall]
                        for idx in fill:
                            type_selected.append(idx)
                            type_selected_set.add(idx)
                            remaining_type_budget -= 1

                logger.debug(
                    "[ErrorAware] %s rank %d: quota=%d, reranked=%d, filled=%d",
                    error_type, rank, rank_quota, n_reranked,
                    len(take_from_rerank) + max(0, rank_quota - n_reranked) - n_reranked,
                )

            # Fill remaining type budget from all ranks' unused candidates
            if remaining_type_budget > 0:
                for info in rank_infos:
                    if remaining_type_budget <= 0:
                        break
                    fill_pool = info['pool'] - type_selected_set
                    if fill_pool:
                        fill_list = list(fill_pool)
                        self.rng.shuffle(fill_list)
                        for idx in fill_list[:remaining_type_budget]:
                            type_selected.append(idx)
                            type_selected_set.add(idx)
                            remaining_type_budget -= 1

            selected.update(type_selected)
            logger.info("[ErrorAware] Error type '%s': selected %d/%d (budget=%d)",
                       error_type, len(type_selected), type_budget, type_budget)

        # Remove any that are in exclude
        selected -= exclude

        # ── Step I: Safety cap ────────────────────────────────────────────
        if len(selected) > max_error_aware:
            logger.info("[ErrorAware] Safety cap: trimming from %d to %d (ratio=%.2f)",
                       len(selected), max_error_aware, self.error_aware_ratio)
            selected_list = list(selected)
            self.rng.shuffle(selected_list)
            selected = set(selected_list[:max_error_aware])

        logger.info("[ErrorAware] Final selection: %d samples (max_budget=%d)",
                   len(selected), max_error_aware)

        return selected

    def _build_candidate_entries(self, candidate_pool: Set[int]) -> List[Dict]:
        """
        Build candidate entry dicts (with SQL) for a set of QB indices.

        The ground-truth SQL is extracted from reward_model.ground_truth.sql,
        which is the canonical clean SQL (not the CoT target).

        Args:
            candidate_pool: Set of QB row indices

        Returns:
            List of dicts with keys: idx, sql
        """
        candidate_entries = []
        for idx in candidate_pool:
            if idx >= len(self.question_bank):
                continue
            row = self.question_bank.iloc[idx]

            # Extract SQL: priority is reward_model.ground_truth.sql (clean SQL)
            sql = ""
            if 'reward_model' in row.index:
                rm = row['reward_model']
                if isinstance(rm, dict):
                    sql = rm.get('ground_truth', {}).get('sql', '')

            if not sql:
                continue  # skip entries without valid SQL

            candidate_entries.append({
                'idx': idx,
                'sql': sql,
            })
        return candidate_entries

    def get_stats(self) -> Dict:
        """Return a dict of current retrieval statistics for logging."""
        stats = {
            "retrieval/total_added": len(self._added_indices),
            "retrieval/qb_remaining": len(self.question_bank) - len(self._added_indices),
            "retrieval/qb_utilization": len(self._added_indices) / len(self.question_bank),
        }
        # Include error-aware stats if enabled
        if self.error_aware_retriever is not None:
            stats.update(self.error_aware_retriever.get_stats())
        return stats
