# tracewise — Knowledge Tracing + Adaptive Intervention Selection

**Research question:** Can a learned student model be used to dynamically select
teaching interventions that improve simulated learning outcomes, compared to
rule-based and random baselines?

This is not "an AI tutor chatbot." The core contribution is a **student modeling +
decision-making system**. An LLM sits on top of it purely as a natural-language
interface (explanation/question generation, Arabic/English), and does **not**
decide what the student needs — the student model and policy do.

```
Dataset (ASSISTments)
        │
        ▼
Student interaction sequences
        │
        ▼
┌─────────────────────┐
│ Student Modeling     │   BKT (classical) vs DKT (LSTM, PyTorch)
└──────────┬───────────┘
           ▼
     Student state s_t
           │
           ▼
┌─────────────────────┐
│ Adaptive Policy      │   Random → Rule-based → Contextual Bandit
└──────────┬───────────┘
           ▼
      Intervention a_t
           │
           ▼
┌─────────────────────┐
│ Student Simulator    │   offline/simulated evaluation only
└──────────┬───────────┘
           ▼
     Reward / outcome → update student state
           │
           ▼
┌─────────────────────┐
│ LLM Tutor (optional) │   turns (state, intervention) into text, EN/AR
└─────────────────────┘
```

## Project status

Tracking against the 14-day plan. See `docs/roadmap.md`.
The KT and contextual-bandit work through M5.7 is complete. M6 delayed-effect
calibration is complete; DQN, LLM/RAG, and the user interface are not yet
implemented.

- [x] Day 1–2: Dataset + preprocessing
- [x] Day 3: BKT (val ROC-AUC 0.7652)
- [x] Day 4–6: DKT (val ROC-AUC 0.8386 @ epoch 11)
- [x] Day 7: BKT vs DKT evaluation (table + slices + calibration — see `docs/experiments_kt.md`)
- [x] Day 8–9: Rule-based policy + contextual bandit
- [x] Day 10: Student simulator + offline policy comparison (see `docs/experiments_policy.md`)
- [x] Day 10.5: KT-aware policy — oracle vs BKT/DKT estimated state (M5.5)
- [x] M6 Step 1: Delayed-effect calibration (no DQN yet)
- [x] M6 Step 2: Episodic RL environment + POMDP/leakage specification
- [ ] M6 Step 3: DQN training + held-out policy evaluation
- [ ] Day 11–12: LLM + RAG integration
- [ ] Day 13: Arabic + Gradio interface
- [ ] Day 14: Evaluation write-up + docs

## Repo structure

```
tracewise/
├── configs/            # yaml configs (paths, hyperparams)
├── data/
│   ├── raw/             # untouched downloaded data (gitignored)
│   └── processed/       # cleaned train/val/test splits (gitignored)
├── src/
│   ├── data/            # download + preprocessing + PyTorch Dataset
│   ├── models/          # bkt.py, dkt.py
│   ├── policy/          # random, rule_based, bandit, simulator, kt_state
│   ├── rl/              # episodic POMDP environment; DQN follows in Phase 3
│   ├── llm/             # tutor.py (LLM + RAG glue)
│   ├── evaluation/       # metrics, comparison scripts
│   └── utils/            # seeding, config loading, logging
├── scripts/              # thin CLI entry points that call into src/
├── notebooks/            # Kaggle/Colab notebooks (exploration + reports)
├── results/              # metrics tables, plots, saved checkpoints (gitignored)
└── docs/                 # architecture.md, roadmap.md, experiment logs
```

## Setup

The reproducible core environment targets Python 3.10.11. From the
`tracewise/` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux: `python -m venv .venv && source .venv/bin/activate`.

Then, after placing `data/raw/skill_builder_data.csv`:

```powershell
python scripts/run_preprocessing.py --config configs/config.yaml
python scripts/fit_bkt.py --config configs/config.yaml
python scripts/train_dkt.py --config configs/config.yaml
```

For development or CI without the optional LLM/RAG packages:

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

The test suite uses small synthetic fixtures; it does not require the private
ASSISTments CSV or saved model checkpoints. Run only the end-to-end policy
smoke checks with `python -m pytest tests/test_smoke.py`.

Kaggle notebook (where actual BKT/DKT training happens, T4 x2):

Use [`notebooks/kaggle_kt_experiments.ipynb`](notebooks/kaggle_kt_experiments.ipynb).

1. Upload that notebook to Kaggle (File → Import).
2. Settings → GPU on, Internet **ON**.
3. **Add Data** → search ASSISTments 2009 skill-builder.
4. Run All. First run only preprocesses (`RUN_BKT` / `RUN_DKT` stay `False`).
5. The notebook clones `https://github.com/HusseinHanafy207/tracewise.git` into `/kaggle/working/tracewise`.

## Dataset

ASSISTments (skill-builder dataset). See `data/README.md` for download
instructions and the exact columns we use.

## Reproducibility

- Fixed seeds via `src/utils/seed.py`.
- All experiment configs live in `configs/`, not hardcoded in notebooks.
- Every result reported in `docs/` should be traceable to a config + git commit.
- Nineteen deterministic unit/smoke tests cover data splitting, KT models,
  policy updates, hidden-state boundaries, and short end-to-end simulations.
- GitHub Actions runs the core test environment on every push and pull request.
- Artifact fingerprints, checkpoint lineage, and the two distinct DKT runs are
  recorded in [`docs/artifacts.md`](docs/artifacts.md).
- Generated data/results remain gitignored. Rebuild them with the commands in
  the artifact manifest; obtain the raw dataset using [`data/README.md`](data/README.md).
