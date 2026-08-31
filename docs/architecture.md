# Architecture

## System diagram

```
Dataset (ASSISTments)
        │
        ▼
Student interaction sequences (src/data/preprocess.py)
        │
        ▼
┌─────────────────────────────┐
│ Student Modeling             │
│  - BKT   (src/models/bkt.py) │
│  - DKT   (src/models/dkt.py) │
│  - Online trackers (src/policy/kt_state.py) │
└──────────────┬───────────────┘
               ▼
   Estimated state ŝ_t  (from observed skill, correct)
   [oracle mode: true mastery — eval upper bound only]
               │
               ▼
┌─────────────────────────────┐
│ Adaptive Policy               │
│  - Random (src/policy/random_policy.py) │
│  - Rule-based (src/policy/rule_based.py) │
│  - Contextual bandit (src/policy/bandit.py) │
│  - DQN (src/rl/dqn.py; oracle + estimated-state evaluation) │
└──────────────┬───────────────┘
               ▼
        Intervention a_t
               │
    ┌──────────┴──────────┐
    ▼                       ▼
Student Simulator      LLM Tutor (src/llm/tutor.py)
(src/policy/simulator.py)   generates EN/AR explanation/question
    │                       for the chosen intervention
    ▼
Reward / outcome → KT tracker update (ŝ), env mastery update
```

The sequential RL interface is `src/rl/environment.py`. It wraps the simulator
as a fixed-horizon POMDP with Gymnasium-style `reset`/`step` returns. The first
learned sequential policy is the replay-buffer DQN in `src/rl/dqn.py`, trained
by `scripts/train_dqn.py`. Phase 4 loads fresh BKT/DKT online trackers through
`src/rl/state_tracking.py` and performs paired episode evaluation through
`src/rl/evaluation.py`. See `docs/rl_environment.md` for the observation,
hidden-state, reward, terminal, and leakage contracts these components follow.

## Key architectural decision: separation of concerns

The LLM is a **communication layer**, not a **decision-making layer**.

- Student model (BKT/DKT) decides *what the student knows*.
- Policy (rule-based/bandit/DQN) decides *what intervention to give next*.
- LLM decides *how to phrase that intervention*, in the student's
  preferred language, grounded in retrieved course material.

This separation is what makes the project an ML/RL system with a GenAI
interface, rather than a GenAI wrapper. Keep this distinction explicit in
any writeup, presentation, or PhD application materials — it's the
central argument for why this is more than "LLM + RAG".

## Evaluation honesty

The student simulator (`src/policy/simulator.py`) is used ONLY for
offline policy comparison. Results from it describe simulated learning
outcomes under stated modeling assumptions (see `intervention_effect()` in
that file) — they are not evidence about real student learning. State
this explicitly wherever these results are reported.

**Partial observability (M5.5):** Deployable conditions feed the policy only
estimated mastery from online BKT/DKT updates on `(skill, correct)`. Oracle
true-mastery context is an evaluation upper bound, not a production mode.
