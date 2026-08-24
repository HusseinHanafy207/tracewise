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
└──────────────┬───────────────┘
               ▼
        Student state s_t
               │
               ▼
┌─────────────────────────────┐
│ Adaptive Policy               │
│  - Random (src/policy/random_policy.py) │
│  - Rule-based (src/policy/rule_based.py) │
│  - Contextual bandit (src/policy/bandit.py) │
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
Reward / outcome → state update
```

## Key architectural decision: separation of concerns

The LLM is a **communication layer**, not a **decision-making layer**.

- Student model (BKT/DKT) decides *what the student knows*.
- Policy (rule-based/bandit) decides *what intervention to give next*.
- LLM decides *how to phrase that intervention*, in the student's
  preferred language, grounded in retrieved course material.

This separation is what makes the project an ML/RL system with a GenAI
interface, rather than a GenAI wrapper. Keep this distinction explicit in
any writeup, presentation, or PhD application materials — it's the
central argument for why this is more than "LLM + RAG".

## Evaluation honesty

The student simulator (`src/policy/simulator.py`) is used ONLY for
offline policy comparison. Results from it describe simulated learning
outcomes under stated modeling assumptions (see `INTERVENTION_EFFECT` in
that file) — they are not evidence about real student learning. State
this explicitly wherever these results are reported.
