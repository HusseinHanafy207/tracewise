# BKT vs DKT (ASSISTments 2009 skill-builder)

Student-level 70/15/15 split, seed 42. Config: `configs/config.yaml`.
DKT: embedding 128, LSTM 128, 1 layer, max_seq_len 200, early stop on val AUC.

DKT predicts P(correct) for the **next** interaction given (skill_t, correct_t).
It cannot score t=0, and long histories are truncated at 200. **BKT_aligned**
uses the same positions. **BKT_full** is the original per-step BKT protocol
from M2 (all timesteps, no truncation) — not a matched comparison.

## Results (Kaggle T4, 2026-08-26)

| Model | Split | ROC-AUC | Accuracy | N |
|---|---|---|---|---|
| BKT_full | val | 0.7652 | 0.7479 | 56559 |
| BKT_aligned | val | 0.7473 | 0.7176 | 33319 |
| DKT | val | **0.8386** | 0.7732 | 33319 |
| BKT_full | test | 0.7418 | 0.7493 | 71191 |
| BKT_aligned | test | 0.7598 | 0.7224 | 33829 |
| DKT | test | **0.8531** | 0.7836 | 33829 |

DKT: 179,579 parameters; best checkpoint epoch 11; train ~12s.

## How to talk about this

- Headline: neural KT beats classical BKT by ~0.09 AUC on the aligned next-step task.
- Do not quote BKT_full vs DKT as the fair gap; n and the prediction target differ.
- Test AUC > val AUC is a split effect (student-level holdout), not a leak: users do not overlap.
- n drops vs BKT_full mainly because of max_seq_len=200 (long-tail students) plus dropping t=0.

## Remaining M4 analysis (required before M5)

Headline AUC is not enough for a research writeup. After pushing, on Kaggle:

```python
%cd /kaggle/working/tracewise
!python scripts/analyze_kt.py --config configs/config.yaml
```

That script does not retrain. It writes slice tables, Brier/ECE, a reliability diagram, and `results/kt_slices.json`. Paste the printed tables here before starting the policy/RL layer.
