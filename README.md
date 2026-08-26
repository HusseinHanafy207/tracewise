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
Code for Days 1–10 is scaffolded; nothing has been trained or evaluated yet.

- [ ] Day 1–2: Dataset + preprocessing (Kaggle notebook ready; CSV not yet run)
- [ ] Day 3: BKT
- [ ] Day 4–6: DKT
- [ ] Day 7: BKT vs DKT evaluation
- [ ] Day 8–9: Rule-based policy + contextual bandit
- [ ] Day 10: Student simulator + offline policy comparison
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
│   ├── policy/          # random, rule_based, bandit, simulator
│   ├── llm/             # tutor.py (LLM + RAG glue)
│   ├── evaluation/       # metrics, comparison scripts
│   └── utils/            # seeding, config loading, logging
├── scripts/              # thin CLI entry points that call into src/
├── notebooks/            # Kaggle/Colab notebooks (exploration + reports)
├── results/              # metrics tables, plots, saved checkpoints (gitignored)
└── docs/                 # architecture.md, roadmap.md, experiment logs
```

## Setup

From the `tracewise/` directory:

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
