# SQL Retrieval

A SQL-skeleton-based retrieval system that builds a FAISS index over a question bank and serves structurally-complementary candidates to the Coverage-Guided Augmentation module (`../data_augmentation/`).

## Input

You need to provide a question bank as input. It can come from any Text-to-SQL dataset (BIRD, Spider, your own data, etc.). Place it at `./data/question_bank/question_bank.json` as a JSON array where each entry contains an executable gold `sql`, its `db_id`, and the natural-language `question`:

```json
{
  "db_id": "concert_singer",
  "sql": "SELECT name FROM singer WHERE age > 30",
  "question": "List the names of singers older than 30."
}
```

The corresponding database schemas (`tables.json`) must be available for schema-aware skeleton parsing. All SQLs are expected to be executable on the target databases.

## Quick Start

```bash
conda activate coalsql
bash run_preprocess.sh
```

## Configuration

| Variable | Description |
|----------|-------------|
| `EMBEDDING_MODEL_PATH` | Embedding model path (defaults to `Qwen3-Embedding-0.6B`) |

Common command-line options (see each script's `--help`):

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu_id` | 0 | GPU device id |
| `--batch_size` | 256 | Embedding encoding batch size |
| `--top_k` | 5 | Number of results returned per query |

## Outputs

All artifacts are written to `output/`:

| File | Description |
|------|-------------|
| `skeletons.json` | Per-record skeletons: `{idx, db_id, sql, skeleton, question}` |
| `skeleton.index` | FAISS `IndexFlatIP` index over unique skeletons |
| `skeleton_metadata.json` | Metadata for each indexed vector |
| `skeleton_stats.json` | Skeleton frequency statistics |
| `train_skeleton_embeddings.npy` | Precomputed training-set skeleton embeddings |
| `qb_skeleton_query_embeddings.npy` | Precomputed question-bank skeleton embeddings |
