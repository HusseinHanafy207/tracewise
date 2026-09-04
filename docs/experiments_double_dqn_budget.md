# Double DQN training budget: 2,000 versus 5,000 episodes

## Question

Does giving Double DQN 5,000 training episodes instead of 2,000 improve mean
held-out final mastery or reduce variability across training seeds?

All results use the oracle-state student simulator. No intervention policy was
evaluated on real students.

## Controlled protocol

The experiment uses ten matched training seeds (42–51). Only
`dqn.train_episodes` changes. Architecture, optimizer, replay buffer, target
updates, epsilon schedule, validation protocol, environment, and evaluation
seeds remain fixed.

Each 5,000-episode run was trained from scratch rather than resumed. The saved
histories verify that, for every seed, its first 2,000 training records and all
validation records through episode 2,000 exactly equal the corresponding
2,000-episode run. Each reference has 100,000 environment steps; each candidate
has 250,000.

Evaluation uses 500 common held-out episodes (`200000..200499`). Episode-level
comparisons are paired within each trained-agent pair. Algorithm-level
intervals use a paired bootstrap over the ten training seeds, which are the
replication unit.

## Two checkpoint estimands

The project normally deploys the checkpoint with the best score on 100 fixed
validation episodes. A 5,000-episode run has 50 checkpoint candidates, while a
2,000-episode run has 20. Therefore two comparisons are reported:

- **Validation-selected:** best checkpoint available within each budget. This
  answers whether a larger training-and-selection budget yields a better
  deployable checkpoint.
- **Exact endpoint:** checkpoint at exactly episode 2,000 or 5,000. This
  isolates what happens when training simply continues and avoids unequal
  checkpoint-selection opportunities.

## Results

| Checkpoint estimand | 2,000 mean ± seed SD | 5,000 mean ± seed SD | 5,000 − 2,000 mean, seed-bootstrap 95% CI | SD difference, seed-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Validation-selected | 0.5676 ± 0.0147 | **0.5756 ± 0.0087** | **+0.0080 [+0.0018, +0.0157]** | -0.0060 [-0.0113, +0.0005] |
| Exact endpoint | **0.5556 ± 0.0166** | 0.5414 ± 0.0122 | **-0.0142 [-0.0230, -0.0026]** | -0.0044 [-0.0100, +0.0012] |

For validation-selected checkpoints, five seeds find a new best after episode
2,000 and improve held-out mastery; five retain their earlier checkpoint and
tie exactly. The later selected episodes are 2,100, 4,500, 2,500, 2,900, and
2,400 for seeds 42, 43, 44, 48, and 50.

At the exact endpoint, 8/10 episode-5,000 policies are worse than their matched
episode-2,000 policies. Only seeds 48 and 50 improve. Thus the selected-policy
gain does not mean learning improves monotonically with more updates.

Both 5,000-episode conditions have smaller observed seed SDs, but both 95%
bootstrap intervals for the SD difference include zero. The experiment does
not support the claim that additional experience reduces training variability.

## Conclusion

The larger budget is useful only together with validation-based checkpoint
selection. It raises the mean quality of the selected policy by 0.0080, but
continuing every agent to episode 5,000 lowers endpoint performance by 0.0142.
Training remains non-monotonic after epsilon reaches its minimum at episode
1,200, so early stopping is essential.

The defensible statement is:

> More Double DQN experience creates additional opportunities to find a better
> checkpoint, but the final 5,000-episode networks are worse on average and the
> data do not establish lower training-seed variability.

The fixed validation set is reused at every checkpoint, so the selected result
may include validation-selection optimism. A future confirmation could use a
larger independent selection set or nested evaluation. That limitation does
not affect the exact-endpoint degradation result.

## Reproduction

Train each candidate seed from 42 through 51 with a matching tag:

```powershell
python scripts/train_dqn.py --config configs/double_dqn_train_5000.yaml --observation-mode oracle --seed 42 --tag double_budget5000_seed42 --device cpu
python scripts/eval_double_dqn_budget.py --config configs/double_dqn_budget_evaluation.yaml --device cpu
```

The evaluator requires all matched artifacts, audits their metadata and exact
2,000-episode prefixes, and evaluates both checkpoint estimands. Raw output is
written to `results/double_dqn_budget_comparison.json`, with a plot at
`results/figures/double_dqn_budget_comparison.png`. These remain gitignored.
The durable summary is
[`examples/double_dqn_budget_results.json`](examples/double_dqn_budget_results.json).
