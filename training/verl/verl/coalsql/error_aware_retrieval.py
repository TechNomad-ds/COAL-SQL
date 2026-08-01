"""
Error-Aware Retrieval Enhancement for EpochRetriever.

This module provides error-type-aware LLM reranking of retrieval candidates.
When a failed sample's error is diagnosed as a structural/functional type (taxonomy
id 17-30), the standard skeleton-similarity candidates are reranked by an LLM to
select those that best exercise the specific SQL construct the model struggled with.

Key design decisions:
  - Online, lightweight design: only the prompt-fill + LLM-call + JSON-parse logic,
    without offline I/O, checkpointing, etc.
  - Batch rerank: one LLM call per failed sample judges 20-30 candidates at once.
  - Diagnosis cache: avoids re-diagnosing the same (split, index) across epochs.
  - Fallback: any LLM failure gracefully falls back to original skeleton retrieval.

Dependencies:
  - core.LLMClient: LLM API client (litellm-based)
  - core.parallel_process_with_retry: parallel execution with retries
  - core.extract_json: robust JSON extraction from LLM output
"""

import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)

# Structural/functional error type IDs that trigger rerank
DEFAULT_STRUCTURAL_ERROR_IDS = set(range(17, 31))  # 17-30 inclusive


class ErrorAwareRetriever:
    """
    Error-Aware retrieval enhancer for EpochRetriever.

    Provides two capabilities:
    1. diagnose_errors(): Diagnose error types for failed samples via LLM
    2. rerank_candidates(): Rerank retrieval candidates for a target error type

    Args:
        config: dict with keys:
            - llm_api: {base_url, api_key, model}
            - max_workers: int (default 64)
            - max_retries: int (default 3)
            - structural_error_ids: list of ints (default 17-30)
        taxonomy_path: path to taxonomy.json
        diagnosis_prompt_path: path to error_diagnosis.txt
        rerank_prompt_path: path to error_aware_rerank.txt
    """

    def __init__(
        self,
        config: Dict[str, Any],
        taxonomy_path: str,
        diagnosis_prompt_path: str,
        rerank_prompt_path: str,
    ):
        # Import core utilities (add parent paths)
        import sys
        coalsql_sql_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        if coalsql_sql_root not in sys.path:
            sys.path.insert(0, coalsql_sql_root)

        from core import LLMClient, parallel_process_with_retry, extract_json
        self._parallel_process = parallel_process_with_retry
        self._extract_json = extract_json

        # Load taxonomy
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            self.taxonomy: List[Dict] = json.load(f)

        # Build lookup maps
        self.id_to_category: Dict[int, Dict] = {cat['id']: cat for cat in self.taxonomy}
        self.name_to_id: Dict[str, int] = {cat['name']: cat['id'] for cat in self.taxonomy}
        self.valid_categories: Set[str] = {cat['name'] for cat in self.taxonomy}

        # Load prompt templates
        with open(diagnosis_prompt_path, 'r', encoding='utf-8') as f:
            self.diagnosis_template = f.read()
        with open(rerank_prompt_path, 'r', encoding='utf-8') as f:
            self.rerank_template = f.read()

        # Load skeleton filter prompt (optional, for LLM-based skeleton filtering)
        skeleton_filter_prompt_path = config.get('skeleton_filter_prompt_path') or \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "skeleton_filter.txt")
        if os.path.exists(skeleton_filter_prompt_path):
            with open(skeleton_filter_prompt_path, 'r', encoding='utf-8') as f:
                self.skeleton_filter_template = f.read()
        else:
            self.skeleton_filter_template = None
            logger.warning("[ErrorAware] skeleton_filter.txt not found at %s, skeleton filtering disabled.",
                          skeleton_filter_prompt_path)

        # Config
        llm_cfg = config.get('llm_api', {})
        self.max_workers = config.get('max_workers', 64)
        self.max_retries = config.get('max_retries', 3)

        # Structural error IDs that trigger rerank
        structural_ids = config.get('structural_error_ids', None)
        if structural_ids:
            self.structural_error_ids = set(int(i) for i in structural_ids)
        else:
            self.structural_error_ids = DEFAULT_STRUCTURAL_ERROR_IDS

        # Initialize LLM client
        self.llm_client = LLMClient(
            api_base=llm_cfg.get('base_url', 'http://localhost:3000/v1'),
            api_key=llm_cfg.get('api_key', 'sk-placeholder'),
            model=llm_cfg.get('model', 'deepseek-v3.2'),
            max_retries=self.max_retries,
            max_tokens=llm_cfg.get('max_tokens', 8192),
            timeout=llm_cfg.get('timeout', 120),
        )

        # Diagnosis cache: (split, index) -> diagnosis result
        self._diagnosis_cache: Dict[Tuple[str, int], Optional[str]] = {}

        # Pre-format taxonomy JSON string for prompts
        self._taxonomy_json = json.dumps(self.taxonomy, ensure_ascii=False, indent=2)

        logger.info(
            "[ErrorAware] Initialized. structural_ids=%s, "
            "max_workers=%d, model=%s",
            sorted(self.structural_error_ids),
            self.max_workers, llm_cfg.get('model', 'unknown')
        )

    # ========================================================================
    # Error Diagnosis
    # ========================================================================

    def diagnose_errors(
        self,
        failed_samples: List[Dict],
    ) -> Dict[Tuple[str, int], Optional[str]]:
        """
        Batch-diagnose error types for failed samples.

        Each sample dict should contain:
            - split: str ('train' or 'question_bank')
            - index: int (dataset index)
            - context: str (schema + question text)
            - correct_sql: str (ground truth SQL)
            - error_response: str (model's incorrect SQL output)

        Returns:
            Dict mapping (split, index) -> target_error_type_name or None.
            None means no structural error (id 17-30) found → skip rerank.
        """
        # Filter out already-cached samples
        pending_samples = []
        for sample in failed_samples:
            key = (sample['split'], sample['index'])
            if key not in self._diagnosis_cache:
                pending_samples.append(sample)

        if pending_samples:
            logger.info("[ErrorAware] Diagnosing %d new samples (%d cached)",
                       len(pending_samples), len(failed_samples) - len(pending_samples))

            # Process in parallel
            def _diagnose_one(sample: Dict) -> Tuple[Tuple[str, int], Optional[str]]:
                return self._diagnose_single(sample)

            results, failures = self._parallel_process(
                items=pending_samples,
                process_func=_diagnose_one,
                max_workers=self.max_workers,
                max_retries=self.max_retries,
                description="[ErrorAware] Diagnosing errors",
                show_progress=True,
            )

            # Update cache from results
            for result in results:
                key, target_type = result
                self._diagnosis_cache[key] = target_type

            # For failures, cache as None (fallback to original retrieval)
            for sample, error in failures:
                key = (sample['split'], sample['index'])
                self._diagnosis_cache[key] = None
                logger.warning("[ErrorAware] Diagnosis failed for %s: %s", key, error)

        # Build result dict for all requested samples
        result_dict = {}
        for sample in failed_samples:
            key = (sample['split'], sample['index'])
            result_dict[key] = self._diagnosis_cache.get(key, None)

        return result_dict

    def _diagnose_single(self, sample: Dict) -> Tuple[Tuple[str, int], Optional[str]]:
        """
        Diagnose a single failed sample's error type.
        Returns ((split, index), target_error_type_name_or_None).
        Raises on LLM/parsing failure to trigger retry.
        """
        key = (sample['split'], sample['index'])

        # Fill prompt
        filled_prompt = self.diagnosis_template.format(
            taxonomy=self._taxonomy_json,
            input_text=sample['context'],
            correct_answer=sample['correct_sql'],
            output_text=sample['error_response'],
        )

        # Call LLM
        response = self.llm_client.chat(query=filled_prompt, temperature=0.3)

        # Parse JSON
        analysis = self._extract_json(response.content)

        # Validate
        if not isinstance(analysis, dict):
            raise ValueError(f"Expected dict, got {type(analysis)}")

        final_answer = analysis.get('final_answer', [])
        if not isinstance(final_answer, list):
            raise ValueError(f"final_answer is not a list")

        # Find the first structural error (id 17-30) in the ordered list
        target_type = None
        for error in final_answer:
            if not isinstance(error, dict):
                continue
            category_name = error.get('category', '')
            category_id = self.name_to_id.get(category_name, 0)
            if category_id in self.structural_error_ids:
                target_type = category_name
                break

        return (key, target_type)

    # ========================================================================
    # LLM Rerank
    # ========================================================================

    def rerank_candidates(
        self,
        target_error_type: str,
        candidate_entries: List[Dict],
        top_n: Optional[int] = None,
    ) -> List[int]:
        """
        Rerank candidate questions for a target error type using LLM.

        Args:
            target_error_type: Name of the target error category (e.g., "Window Functions")
            candidate_entries: List of dicts, each with:
                - idx: int (QB index)
                - question: str (question text, can be truncated)
                - sql: str (ground truth SQL)
            top_n: Max number of candidates to select. None means no limit
                   (return all LLM-selected candidates).

        Returns:
            List of QB indices (from candidate_entries[i]['idx']) in priority order.
            Returns empty list on failure.
        """
        if not candidate_entries:
            return []

        # Get error type description
        error_id = self.name_to_id.get(target_error_type, 0)
        error_info = self.id_to_category.get(error_id, {})
        error_description = error_info.get('description', target_error_type)

        # Format candidates for prompt
        candidates_text = self._format_candidates_for_prompt(candidate_entries)

        # Fill prompt
        filled_prompt = self.rerank_template.format(
            error_type_name=target_error_type,
            error_type_description=error_description,
            candidates=candidates_text,
        )

        try:
            # Call LLM
            response = self.llm_client.chat(query=filled_prompt, temperature=0.2)

            # Parse response
            result = self._extract_json(response.content)

            if not isinstance(result, dict):
                logger.warning("[ErrorAware] Rerank result is not a dict, fallback.")
                return []

            selected_ids = result.get('selected_ids', [])
            if not isinstance(selected_ids, list):
                logger.warning("[ErrorAware] selected_ids is not a list, fallback.")
                return []

            # Map candidate IDs back to QB indices
            # The candidate IDs in the prompt are 1-indexed positions
            id_to_qb_idx = {i + 1: entry['idx'] for i, entry in enumerate(candidate_entries)}
            reranked_indices = []
            for cid in selected_ids:
                cid = int(cid)
                if cid in id_to_qb_idx and id_to_qb_idx[cid] not in reranked_indices:
                    reranked_indices.append(id_to_qb_idx[cid])
                if top_n is not None and len(reranked_indices) >= top_n:
                    break

            return reranked_indices

        except Exception as e:
            logger.warning("[ErrorAware] Rerank failed for error_type='%s': %s",
                          target_error_type, e)
            return []

    def batch_rerank(
        self,
        rerank_tasks: List[Dict],
    ) -> Dict[Any, List[int]]:
        """
        Batch rerank for multiple tasks in parallel.

        Args:
            rerank_tasks: List of dicts, each with:
                - key: hashable identifier (e.g. (error_type, rank, batch_start))
                - target_error_type: str
                - candidate_entries: List[Dict] with idx, sql
                - budget: (optional) max candidates to select for this batch

        Returns:
            Dict mapping key -> list of reranked QB indices
        """
        if not rerank_tasks:
            return {}

        def _rerank_one(task: Dict) -> Tuple[Tuple[str, int], List[int]]:
            indices = self.rerank_candidates(
                target_error_type=task['target_error_type'],
                candidate_entries=task['candidate_entries'],
                top_n=task.get('budget', None),
            )
            return (task['key'], indices)

        results, failures = self._parallel_process(
            items=rerank_tasks,
            process_func=_rerank_one,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            description="[ErrorAware] Reranking candidates",
            show_progress=True,
        )

        # Build result dict
        result_dict = {}
        for key, indices in results:
            result_dict[key] = indices

        # For failures, return empty (will fallback to original)
        for task, error in failures:
            result_dict[task['key']] = []
            logger.warning("[ErrorAware] Rerank failed for %s: %s", task['key'], error)

        return result_dict

    # ========================================================================
    # Helpers
    # ========================================================================

    def _format_candidates_for_prompt(self, candidates: List[Dict]) -> str:
        """Format candidate entries as numbered list for the rerank prompt (SQL only)."""
        lines = []
        for i, entry in enumerate(candidates):
            # Use 1-indexed IDs
            cid = i + 1
            sql = entry.get('sql', '')
            lines.append(f"[ID {cid}]")
            lines.append(f"SQL: {sql}")
            lines.append("")
        return "\n".join(lines)

    def filter_skeletons(
        self,
        error_type: str,
        skeleton_infos: List[Dict],
    ) -> List[int]:
        """
        Use LLM to filter skeleton templates, keeping only those relevant
        to the target error type.

        Args:
            error_type: Name of the target error category (e.g., "Window Functions")
            skeleton_infos: List of dicts, each with:
                - fid: int (FAISS index / skeleton template ID)
                - skeleton: str (the SQL skeleton pattern)

        Returns:
            List of fid values that are relevant to the error type.
            Returns all fids on failure (fallback = no filtering).
        """
        if not self.skeleton_filter_template:
            # No filter prompt available, pass all through
            return [s['fid'] for s in skeleton_infos]

        if not skeleton_infos:
            return []

        # Get error type description
        error_id = self.name_to_id.get(error_type, 0)
        error_info = self.id_to_category.get(error_id, {})
        error_description = error_info.get('description', error_type)

        # Format skeletons for prompt (1-indexed)
        lines = []
        for i, info in enumerate(skeleton_infos):
            sid = i + 1
            skeleton = info.get('skeleton', '(empty)')
            lines.append(f"[ID {sid}] {skeleton}")
        skeletons_text = "\n".join(lines)

        # Fill prompt
        filled_prompt = self.skeleton_filter_template.format(
            error_type_name=error_type,
            error_type_description=error_description,
            skeletons=skeletons_text,
        )

        try:
            response = self.llm_client.chat(query=filled_prompt, temperature=0.2)
            result = self._extract_json(response.content)

            if not isinstance(result, dict):
                logger.warning("[ErrorAware] Skeleton filter result not a dict, fallback.")
                return [s['fid'] for s in skeleton_infos]

            relevant_ids = result.get('relevant_ids', [])
            if not isinstance(relevant_ids, list):
                logger.warning("[ErrorAware] relevant_ids not a list, fallback.")
                return [s['fid'] for s in skeleton_infos]

            # Map 1-indexed IDs back to fids
            relevant_fids = []
            for rid in relevant_ids:
                rid = int(rid)
                if 1 <= rid <= len(skeleton_infos):
                    relevant_fids.append(skeleton_infos[rid - 1]['fid'])

            return relevant_fids

        except Exception as e:
            logger.warning("[ErrorAware] Skeleton filter failed for '%s': %s, fallback.", error_type, e)
            return [s['fid'] for s in skeleton_infos]

    def batch_filter_skeletons(
        self,
        filter_tasks: List[Dict],
    ) -> Dict[str, List[int]]:
        """
        Batch skeleton filtering for multiple error types in parallel.

        Args:
            filter_tasks: List of dicts, each with:
                - error_type: str
                - skeleton_infos: List[Dict] with fid, skeleton

        Returns:
            Dict mapping error_type -> list of relevant fids
        """
        if not filter_tasks:
            return {}

        def _filter_one(task: Dict) -> Tuple[str, List[int]]:
            fids = self.filter_skeletons(
                error_type=task['error_type'],
                skeleton_infos=task['skeleton_infos'],
            )
            return (task['error_type'], fids)

        results, failures = self._parallel_process(
            items=filter_tasks,
            process_func=_filter_one,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            description="[ErrorAware] Filtering skeletons",
            show_progress=True,
        )

        result_dict = {}
        for error_type, fids in results:
            result_dict[error_type] = fids

        # For failures, return all fids (no filtering = fallback)
        for task, error in failures:
            result_dict[task['error_type']] = [s['fid'] for s in task['skeleton_infos']]
            logger.warning("[ErrorAware] Skeleton filter failed for '%s': %s", task['error_type'], error)

        return result_dict

    def get_structural_error_names(self) -> List[str]:
        """Return the names of all structural error types (for logging)."""
        return [
            self.id_to_category[i]['name']
            for i in sorted(self.structural_error_ids)
            if i in self.id_to_category
        ]

    def get_stats(self) -> Dict:
        """Return statistics for logging."""
        cached = len(self._diagnosis_cache)
        structural_count = sum(
            1 for v in self._diagnosis_cache.values() if v is not None
        )
        return {
            "error_aware/diagnosed_total": cached,
            "error_aware/structural_errors": structural_count,
            "error_aware/non_structural": cached - structural_count,
        }
