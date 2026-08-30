# 14-Day Roadmap

Hard milestone at day 14: BKT, DKT, BKT-vs-DKT results, rule-based policy,
contextual bandit, offline simulated policy comparison, basic LLM tutor,
end-to-end demo. Everything else is optional.

| Day | Goal | Module(s) | Status |
|---|---|---|---|
| 1–2 | Dataset + preprocessing | `src/data/preprocess.py`, `scripts/run_preprocessing.py`, `notebooks/kaggle_kt_experiments.ipynb` | done (3119 students, 123 skills, 454232 interactions) |
| 3 | BKT implementation | `src/models/bkt.py`, `scripts/fit_bkt.py` | done on val (see experiment log) |
| 4–6 | DKT implementation | `src/models/dkt.py`, `scripts/train_dkt.py`, `scripts/eval_kt.py` | done on Kaggle (best val AUC 0.8386 @ epoch 11) |
| 7 | BKT vs DKT evaluation | `src/evaluation/metrics.py`, `scripts/eval_kt.py`, `scripts/analyze_kt.py` | done — slices + calibration in `docs/experiments_kt.md` |
| 8–9 | Rule-based policy + contextual bandit | `src/policy/rule_based.py`, `src/policy/bandit.py`, `src/policy/random_policy.py` | done — compared in M5 |
| 10 | Student simulator + offline comparison | `src/policy/simulator.py`, `scripts/eval_policies.py` | done — see `docs/experiments_policy.md` |
| 10.5 | KT-aware policy (estimated state) | `src/policy/kt_state.py`, `scripts/eval_kt_policies.py` | done — oracle vs BKT/DKT/no-state; see M5.5 in `docs/experiments_policy.md` |
| 10.7 | Mastery basis diagnostic | `--suite basis` | done — M5.7 closes bandit section |
| 11a | M6 delayed I/F + DQN (oracle) | delayed simulator, `src/rl/environment.py`, DQN TBD | calibration + episodic POMDP environment done; DQN not started |
| 11–12 | LLM + RAG integration | `src/llm/tutor.py` | prompt stub only |
| 13 | Arabic + Gradio interface | `app.py` (TBD) | not started |
| 14 | Evaluation write-up + docs | `docs/`, `README.md` | not started |

## Priority order if time runs short

1. Student modeling: BKT → DKT (this is the intellectual core)
2. Evaluation: BKT vs DKT comparison table + analysis
3. Adaptive policy: rule-based → contextual bandit
4. Offline simulated policy comparison (oracle + estimated KT state)
5. (nice to have) LLM + RAG
6. (nice to have) Arabic interface
7. (future) handwriting/vision, offline RL, PPO

Do not let 5–7 eat into time needed for 1–4.

## Experiment log

Record every run here with: date, git commit hash, config used, and
result. This becomes the evidence trail for the "paper-quality"
documentation on Day 14.

Artifact SHA-256 fingerprints and the Kaggle/local DKT run distinction are in
`docs/artifacts.md`. A dagger (†) marks a historical policy run made before its
implementation was committed; `b328e47` is the first durable code snapshot,
not an invented claim about the exact historical worktree. These runs will be
regenerated from a clean tagged environment in Phase 1.

| Date | Commit | Experiment | Config | Result |
|---|---|---|---|---|
| 2026-08-26 | `3629892` | preprocess | `configs/config.yaml` | train/val/test students 2183/467/469; 123 skills; 454232 interactions; 0 split overlap |
| 2026-08-26 | `5a65222` | BKT val | `configs/config.yaml` | ROC-AUC **0.7652**, acc 0.7479, n=56559; 114/123 skills fitted, 9 defaulted; 79s Kaggle CPU (129s local) |
| 2026-08-26 | `ec7db0f` | DKT val (Kaggle checkpoint) | `configs/config.yaml` | best val AUC **0.8386** @ epoch 11; early stop epoch 14; 179,579 params; 12s T4 |
| 2026-08-26 | `ec7db0f` | BKT vs DKT | `configs/config.yaml` | aligned next-step: val DKT 0.8386 vs BKT 0.7473; test DKT **0.8531** vs BKT 0.7598 (n=33319/33829). BKT_full val 0.7652 / test 0.7418 |
| 2026-08-26 | `ec7db0f` | KT slices + calibration | `configs/config.yaml` | DKT wins all length/position/skill-freq slices (Δ≈0.08–0.11); ECE 0.011 vs BKT 0.020; Brier 0.145 vs 0.186; 180k vs 492 params |
| 2026-08-27 | `ec7db0f` | DKT val (local/current checkpoint) | `configs/config.yaml` | best val AUC **0.8383** @ epoch 14; current `dkt_best.pt`; used for KT-aware policy experiments |
| 2026-08-26 | `b328e47`† | policy sim (5 seeds) | `configs/config.yaml` | random≺bandits≺rule on mastery; LinUCB best overall acc (0.700) and final-window acc (0.828); state ablation helps acc; **simulated only** — see `docs/experiments_policy.md` |
| 2026-08-27 | `b328e47`† | KT-aware policy (M5.5) | `configs/config.yaml` | rule_oracle≫rule_bkt/dkt on mastery; LinUCB ≈ flat across oracle/BKT/DKT/no-state; \|est−true\|≈0.15–0.22; **simulated + partial observability** — see M5.5 in `docs/experiments_policy.md` |
| 2026-08-27 | `b328e47`† | M5.5 robustness 2000×50×10 | `configs/config.yaml` | same conclusions; LinUCB mastery gain 0.0136–0.0137 across oracle/BKT/DKT/no-state; rule gap persists; stop increasing N |
| 2026-08-27 | `b328e47`† | M5.6 hetero latent z_i | `--heterogeneous` | types implemented + hidden; LinUCB still flat / no_state slightly ahead; rule still needs accurate state — see M5.6 |
| 2026-08-27 | `b328e47`† | M5.7 mastery basis diagnostic | `--suite basis` | oracle_basis > no_state on mastery + final-window acc (0.841 vs 0.827); BKT/DKT basis do not retain gap; linear map was a bottleneck — see M5.7 |
| 2026-08-27 | `b328e47`† | M6 Step 1 delayed I/F calibration | `delayed_effects` | OFF: myopic→m_T=1.0; ON: myopic early-best but interleave wins m_T (0.565 vs 0.522); I/F latent — ready for DQN |
| 2026-08-30 | Phase 2 worktree; source SHA in result JSON | episodic RL environment validation, 500 paired episodes | `configs/config.yaml -> rl` | all gates pass; delayed OFF myopic 1.000 > interleave 0.831 final mastery; delayed ON interleave 0.558 > myopic 0.518 while myopic wins early reward |
