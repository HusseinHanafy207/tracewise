# 14-Day Roadmap

Hard milestone at day 14: BKT, DKT, BKT-vs-DKT results, rule-based policy,
contextual bandit, offline simulated policy comparison, basic LLM tutor,
end-to-end demo. Everything else is optional.

| Day | Goal | Module(s) | Status |
|---|---|---|---|
| 1–2 | Dataset + preprocessing | `src/data/preprocess.py`, `scripts/run_preprocessing.py`, `notebooks/kaggle_kt_experiments.ipynb` | Kaggle notebook ready; not yet run on ASSISTments |
| 3 | BKT implementation | `src/models/bkt.py`, `scripts/fit_bkt.py` | implemented, not evaluated |
| 4–6 | DKT implementation | `src/models/dkt.py`, `src/data/dataset.py`, `scripts/train_dkt.py` | implemented, not trained |
| 7 | BKT vs DKT evaluation | `src/evaluation/metrics.py` | metrics helper ready, no results yet |
| 8–9 | Rule-based policy + contextual bandit | `src/policy/rule_based.py`, `src/policy/bandit.py`, `src/policy/random_policy.py` | implemented, not compared |
| 10 | Student simulator + offline comparison | `src/policy/simulator.py` | implemented, not run |
| 11–12 | LLM + RAG integration | `src/llm/tutor.py` | prompt stub only |
| 13 | Arabic + Gradio interface | `app.py` (TBD) | not started |
| 14 | Evaluation write-up + docs | `docs/`, `README.md` | not started |

## Priority order if time runs short

1. Student modeling: BKT → DKT (this is the intellectual core)
2. Evaluation: BKT vs DKT comparison table + analysis
3. Adaptive policy: rule-based → contextual bandit
4. Offline simulated policy comparison
5. (nice to have) LLM + RAG
6. (nice to have) Arabic interface
7. (future) handwriting/vision, offline RL, PPO

Do not let 5–7 eat into time needed for 1–4.

## Experiment log

Record every run here with: date, git commit hash, config used, and
result. This becomes the evidence trail for the "paper-quality"
documentation on Day 14.

| Date | Commit | Experiment | Config | Result |
|---|---|---|---|---|
| | | | | |
