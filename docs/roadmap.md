# 14-Day Roadmap

Hard milestone at day 14: BKT, DKT, BKT-vs-DKT results, rule-based policy,
contextual bandit, offline simulated policy comparison, basic LLM tutor,
end-to-end demo. Everything else is optional.

| Day | Goal | Module(s) | Status |
|---|---|---|---|
| 1–2 | Dataset + preprocessing | `src/data/preprocess.py`, `scripts/run_preprocessing.py`, `notebooks/kaggle_kt_experiments.ipynb` | done (3119 students, 123 skills, 454232 interactions) |
| 3 | BKT implementation | `src/models/bkt.py`, `scripts/fit_bkt.py` | done on val (see experiment log) |
| 4–6 | DKT implementation | `src/models/dkt.py`, `scripts/train_dkt.py`, `scripts/eval_kt.py` | done on Kaggle (best val AUC 0.8386 @ epoch 11) |
| 7 | BKT vs DKT evaluation | `src/evaluation/metrics.py`, `scripts/eval_kt.py`, `scripts/analyze_kt.py` | headline table done; slice/calibration script ready, not yet run |
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
| 2026-08-26 | (push M2 before Kaggle re-run) | preprocess | `configs/config.yaml` | train/val/test students 2183/467/469; 123 skills; 454232 interactions; 0 split overlap |
| 2026-08-26 | (Kaggle confirmed) | BKT val | `configs/config.yaml` | ROC-AUC **0.7652**, acc 0.7479, n=56559; 114/123 skills fitted, 9 defaulted; 79s Kaggle CPU (129s local) |
| 2026-08-26 | (Kaggle) | DKT val | `configs/config.yaml` | best val AUC **0.8386** @ epoch 11; early stop epoch 14; 179,579 params; 12s T4 |
| 2026-08-26 | (Kaggle) | BKT vs DKT | `configs/config.yaml` | aligned next-step: val DKT 0.8386 vs BKT 0.7473; test DKT **0.8531** vs BKT 0.7598 (n=33319/33829). BKT_full val 0.7652 / test 0.7418 |
