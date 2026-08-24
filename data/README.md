# Dataset: ASSISTments (Skill-Builder 2009–2010)

We use the ASSISTments 2009–2010 "skill-builder" dataset — the standard
benchmark for knowledge tracing (used in the original DKT paper).

## Download

The dataset is hosted by the ASSISTments team / mirrored on Kaggle:

- Primary: https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data
- Kaggle mirror (search "ASSISTments 2009 skill builder") — easiest path if
  working in a Kaggle notebook: use "Add Data" and search for it, no manual
  download needed.

Expected raw file: `skill_builder_data.csv` (or `skill_builder_data_corrected.csv`).
Place it at `data/raw/skill_builder_data.csv`.

## Columns we actually use

| column           | meaning                                  |
|-------------------|-------------------------------------------|
| `user_id`         | student identifier                        |
| `order_id`        | interaction order (proxy for timestamp)   |
| `assignment_id`    | grouping of problems                     |
| `problem_id`       | question identifier                      |
| `skill_id`         | skill/concept tag (may be missing/multi) |
| `correct`          | 0/1 whether the first attempt was correct |
| `attempt_count`     | number of attempts on this problem       |
| `ms_first_response` | response time in ms                      |

Rows with missing `skill_id` are dropped (standard practice for KT on this
dataset — a meaningful fraction of rows lack skill tags). Multi-skill cells
written as `"12,34"` keep only the first id; unparseable ids are dropped.

## Preprocessing summary (see `src/data/preprocess.py`)

1. Drop rows with null `skill_id` or `correct`.
2. Keep students with >= `min_interactions_per_student` (config) interactions.
3. Sort each student's interactions by `order_id`.
4. Map `skill_id` and `problem_id` to contiguous integer indices.
5. Build per-student sequences of `(skill_idx, correct)`.
6. Split by **student** (not by interaction) into train/val/test to avoid
   leakage — 70/15/15 by default.
7. Save as a pickled list of sequences per split under `data/processed/`.

Run:

```bash
python scripts/run_preprocessing.py --config configs/config.yaml
```
