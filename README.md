<h1 align="center">COAL-SQL</h1>

<p align="center">
  <strong>Coverage-Guided Augmentation and Failure-Driven Learning<br>
  for Text-to-SQL Post-Training</strong>
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2609.20842">Paper</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#citation">Citation</a>
</p>

**COAL-SQL** is a unified post-training framework that improves Text-to-SQL through **Co**verage-Guided **A**ugmentation and Failure-Driven **L**earning. It augments seed data with complementary SQL structures and uses failures observed during training to guide corrective supervision and additional practice.

**COAL-SQL achieves 64.9% execution accuracy on the BIRD development set with approximately 12.6K distinct post-training examples.**

## Overview

![COAL-SQL framework: Coverage-Guided Augmentation and Failure-Driven Learning](figures/overview.png)

COAL-SQL addresses two complementary challenges: covering the SQL structures required by the target task and helping the model learn to solve them.

### Coverage-Guided Augmentation (CGA)

CGA expands a seed training set with structurally complementary examples:

1. **Abstract SQL structures.** Convert seed queries and an external SQL collection into schema-independent SQL *skeletons*.
2. **Select complementary skeletons.** Apply metric *K*-center selection, keeping seed skeletons fixed as existing centers, to prioritize structures that complement the seed set.
3. **Synthesize verified examples.** Instantiate the selected skeletons on the target training databases and generate verified question–SQL pairs.

### Failure-Driven Learning (FDL)

FDL supplements GRPO with supervision and practice derived from **Solve-None** examples—training examples for which none of the sampled rollouts is execution-correct:

- **Step-level expansion:** Construct corrective chain-of-thought traces for newly observed failures.
- **Epoch-level expansion:** Generate structurally related practice examples from failures accumulated during training.

Together, CGA and FDL connect structural data coverage with learning from the model's observed failures.

## Installation

Run the following commands from the repository root:

```bash
conda create -n coalsql python=3.10
conda activate coalsql

# Install COAL-SQL dependencies and package
pip install -r requirements.txt
pip install -e .

# Install the bundled verl training backend
cd training/verl
pip install -e .
cd ../..
```

If installing `flash-attn` fails, use a prebuilt wheel compatible with your CUDA and PyTorch versions from the [FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases).

### Configuration

Copy the environment template and edit it for your setup:

```bash
cp .env.example .env
```

Configure the following in `.env`:

| Configuration             | Purpose                                                     |
| ------------------------- | ----------------------------------------------------------- |
| Model path                | Model used for post-training                                |
| `SQL_DATASET_DIR_TRAIN`   | Training database directory used by the execution reward    |
| `SQL_DATASET_DIR_DEV`     | Development database directory used by the execution reward |
| `SQL_DATASET_DIR_QB`      | Additional database directory used by the execution reward  |
| Training result directory | Output location for training results                        |
| LLM API settings          | API used by error-aware retrieval                           |

See `.env.example` for the exact variable names and expected values. Update the data paths and options in the pipeline and training scripts as needed.

## Usage

Run all commands below from the repository root.

### 1. Prepare the data

COAL-SQL requires a target Text-to-SQL task, such as BIRD, and a user-provided external SQL collection for augmentation.

| Input                   | Requirements                                                 | Purpose                                         |
| ----------------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| Seed training set       | Parquet data containing questions, gold SQL queries, and `db_id` | GRPO training and reference structural coverage |
| Target databases        | Associated SQLite databases                                  | SQL instantiation and execution-based rewards   |
| External SQL collection | Records containing executable gold SQL; `db_id` and questions are optional | Source of SQL skeletons for augmentation        |

Datasets are not bundled with this repository. Set the paths in `.env` and the scripts to your own data. See [the skeleton retrieval documentation](sql_retrieval/README.md) for the external collection's expected format.

### 2. Run Coverage-Guided Augmentation

First, prepare the skeleton pool and retrieval index:

```bash
# Build the skeleton pool, FAISS index, and precomputed embeddings
bash sql_retrieval/run_preprocess.sh
```

Then, select complementary skeletons and synthesize verified examples:

```bash
bash data_augmentation/run_augmentation.sh
```

The combined seed and synthesized examples form the post-training data. See [the data augmentation documentation](data_augmentation/README.md) for individual pipeline steps and configurable options.

### 3. Train with Failure-Driven Learning

Launch GRPO training on the combined seed and CGA data, with both step-level and epoch-level FDL enabled:

```bash
bash scripts/train_coalsql.sh
```

## Repository Structure

```text
COAL-SQL/
├── core/                 # Shared LLM client, parallel helpers, and reward utilities
│   └── reward/           # SQL execution and format rewards
├── data_augmentation/    # CGA: skeleton selection and example synthesis
├── sql_retrieval/        # Skeleton extraction, embeddings, and FAISS indexing
├── training/             # FDL and GRPO training
│   └── verl/             # Bundled verl backend
├── scripts/              # Training launch scripts
├── figures/              # README figures
├── requirements.txt
└── setup.py
```

The COAL-SQL training implementation is located in `training/verl/verl/coalsql/`.

Detailed module documentation:

- [Coverage-Guided Augmentation](data_augmentation/README.md)
- [SQL Skeleton Retrieval](sql_retrieval/README.md)

## Citation

If you use COAL-SQL in your research, please cite our paper:

```bibtex
@article{cai2026coal,
  title={COAL-SQL: Coverage-Guided Augmentation and Failure-Driven Learning for Text-to-SQL Post-Training},
  author={Cai, Qifeng and Pan, Xuanguang and Liang, Hao and Xu, Chang and Zhang, Wentao},
  journal={arXiv preprint arXiv:2609.20842},
  year={2026}
}
```
