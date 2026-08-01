# Data Augmentation

The Coverage-Guided Augmentation pipeline: it builds a SQL skeleton pool from a user-provided SQL collection, selects diverse skeletons that complement the seed set via K-Center Greedy, and synthesizes verified question-SQL training samples with a three-stage LLM pipeline.

## Input

You need to provide a SQL collection as input. It can come from any Text-to-SQL corpus (BIRD, Spider, your own data, etc.) and is not tied to any specific dataset format. Place it at `./data/sql_pool.json` as a JSON array where each entry provides at least a SQL query (read from one of `sql` / `SQL` / `query` / `gold_sql`); `db_id` and `question` are optional:

```json
[
  {"db_id": "concert_singer", "sql": "SELECT ...", "question": "..."},
  {"sql": "SELECT ..."}
]
```

You also need a seed set at `./data/train/bird_train.json` (e.g. BIRD train), the target databases for execution verification, and one or more OpenAI-compatible LLM endpoints for synthesis.

## Quick Start

```bash
conda activate coalsql
bash run_augmentation.sh
```

## Configuration

Override defaults via environment variables:

| Variable | Description |
|----------|-------------|
| `EMBEDDING_MODEL_PATH` | Embedding model path (defaults to `Qwen3-Embedding-0.6B`) |
| `TARGET_K` | Number of augmentation skeletons to select (default `3500`) |
| `SYNTH_API_URLS` | Comma-separated LLM endpoints for synthesis |
| `SYNTH_API_MODEL` | Model name served at the endpoints |
| `SYNTH_API_KEY` | API key for the endpoints (default `EMPTY`) |

Per-script command-line options (batch size, GPU count, outlier threshold, few-shot sizes, etc.) are available via each script's `--help`.

## Outputs

All artifacts are written to `./data/skeleton_augmentation/`:

| File | Description |
|------|-------------|
| `skeleton_pool_unique.json` | Deduplicated skeleton pool metadata |
| `skeleton_pool_embeddings.npy` | Normalized pool embeddings aligned by `pool_id` |
| `outlier_scores.npy` / `outlier_stats.json` | Per-skeleton outlier scores and distribution |
| `selected_{TARGET_K}_skeletons.json` | Final selected augmentation skeletons |
| `selected_{TARGET_K}_stats.json` | Selection configuration and summary statistics |
| `synthesis/synthetic_coalsql_{TARGET_K}.json` | Final synthesized text2sql dataset |
| `synthesis/synthetic_coalsql_{TARGET_K}_stats.json` | Success/failure distribution and token usage |

Each synthesized sample contains `db_id`, `question`, `evidence`, `SQL`, `output_reasoning`, `output_sql`, skeleton metadata, and selection statistics. Synthesis is resumable: completed skeletons are skipped, and failures are logged to `synthesis/synthetic_coalsql_{TARGET_K}_failures.jsonl`.
