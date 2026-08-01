"""
Parallel processing utilities.

Provides:
- ThreadPoolExecutor-based parallelism
- Progress bar integration (tqdm)
- Error handling and retry logic
"""
import logging
from typing import List, Callable, Optional, Any, TypeVar, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def parallel_process(
    items: List[T],
    process_func: Callable[[T], R],
    max_workers: int = 8,
    description: str = "Processing",
    show_progress: bool = True,
    on_error: Optional[Callable[[T, Exception], None]] = None,
) -> List[Optional[R]]:
    """
    Process items in parallel with progress tracking.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of parallel workers
        description: Progress bar description
        show_progress: Whether to show progress bar
        on_error: Optional callback for errors (item, exception)

    Returns:
        List of results in the same order as inputs.
        Failed items will have None as result.

    Usage:
        def process_item(item: dict) -> dict:
            return {"id": item["id"], "result": ...}

        results = parallel_process(
            items=items,
            process_func=process_item,
            max_workers=10,
        )
    """
    if not items:
        return []

    results: List[Optional[R]] = [None] * len(items)

    iterator = tqdm(enumerate(items), total=len(items), desc=description, disable=not show_progress)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_func, item): idx
            for idx, item in iterator
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                if on_error:
                    on_error(items[idx], e)
                results[idx] = None

    return results


def parallel_process_with_retry(
    items: List[T],
    process_func: Callable[[T], R],
    max_workers: int = 8,
    max_retries: int = 3,
    description: str = "Processing",
    show_progress: bool = True,
    on_success: Optional[Callable[[T, R], None]] = None,
) -> Tuple[List[R], List[Tuple[T, Exception]]]:
    """
    Process items in parallel with automatic retry for failures.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of parallel workers
        max_retries: Maximum retries for failed items
        description: Progress bar description
        show_progress: Whether to show progress bar
        on_success: Optional callback when a task succeeds (item, result)

    Returns:
        Tuple of (successful_results, failed_items_with_errors)

    Usage:
        successes, failures = parallel_process_with_retry(
            items=items,
            process_func=process_item,
            max_retries=3,
        )
    """
    if not items:
        return [], []

    # Track results and failures
    results: dict[int, R] = {}
    pending_indices = list(range(len(items)))
    failures: List[Tuple[T, Exception]] = []

    for attempt in range(max_retries):
        if not pending_indices:
            break

        current_items = [(idx, items[idx]) for idx in pending_indices]
        new_pending = []

        desc = f"{description} (attempt {attempt + 1}/{max_retries})" if attempt > 0 else description

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks (no progress bar here, submission is instant)
            future_to_idx = {
                executor.submit(process_func, item): idx
                for idx, item in current_items
            }

            # Collect results with progress bar (shows actual completion)
            with tqdm(total=len(current_items), desc=desc, disable=not show_progress) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                        # Call on_success callback if provided
                        if on_success:
                            on_success(items[idx], result)
                    except Exception as e:
                        if attempt < max_retries - 1:
                            new_pending.append(idx)
                            logger.warning(f"Item {idx} failed, will retry: {e}")
                        else:
                            failures.append((items[idx], e))
                            logger.error(f"Item {idx} failed after {max_retries} attempts: {e}")
                    pbar.update(1)

        pending_indices = new_pending

    # Build ordered result list
    ordered_results = [results[i] for i in sorted(results.keys())]

    return ordered_results, failures


class ParallelProcessor:
    """
    Reusable parallel processor with configuration.
    """

    def __init__(
        self,
        max_workers: int = 8,
        show_progress: bool = True,
        max_retries: int = 3,
    ):
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.max_retries = max_retries
        self.errors: List[Tuple[Any, Exception]] = []

    def process(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        description: str = "Processing",
    ) -> List[Optional[R]]:
        """Process items with error tracking."""
        self.errors = []

        def on_error(item: T, e: Exception):
            self.errors.append((item, e))

        return parallel_process(
            items=items,
            process_func=process_func,
            max_workers=self.max_workers,
            description=description,
            show_progress=self.show_progress,
            on_error=on_error,
        )

    def process_with_retry(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        description: str = "Processing",
    ) -> Tuple[List[R], List[Tuple[T, Exception]]]:
        """Process items with retry and error tracking."""
        results, failures = parallel_process_with_retry(
            items=items,
            process_func=process_func,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            description=description,
            show_progress=self.show_progress,
        )
        self.errors = failures
        return results, failures

    def get_error_summary(self) -> dict:
        """Get summary of processing errors."""
        return {
            "total_errors": len(self.errors),
            "error_types": list(set(type(e).__name__ for _, e in self.errors)),
        }
