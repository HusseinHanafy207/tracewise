# DQN result in one page

## Question

Can a value-based policy use sequential student state to improve final
simulated mastery when interventions have delayed effects?

**Scope:** this is an offline simulator study. Oracle mastery is available only
as an experimental upper bound. No real students were evaluated.

## Protocol

- Six actions, 50 interactions per episode, reward equal to mastery change.
- Delayed independence and fatigue make an early reward-maximizing action
  potentially harmful later.
- Each DQN trains for 2,000 episodes and selects a checkpoint on 100 fixed
  validation episodes.
- Final evaluation uses 500 unseen seeds (`200000..200499`) shared by all
  policies. Intervals for comparisons bootstrap paired episode differences.
- Three independently trained oracle-state agents test training stability;
  BKT, DKT, no-state, and discount ablations isolate design choices.

## Headline results

| Policy or condition | Mean final mastery | 95% episode-bootstrap CI |
|---|---:|---:|
| Random | 0.4533 | [0.4432, 0.4635] |
| Myopic oracle | 0.5196 | [0.5110, 0.5285] |
| Explain/harder interleaving | 0.5576 | [0.5431, 0.5725] |
| DQN oracle, seed 42 | **0.5700** | **[0.5551, 0.5855]** |
| DQN oracle, seed 43 | 0.5313 | [0.5211, 0.5413] |
| DQN oracle, seed 44 | 0.5637 | [0.5490, 0.5783] |
| DQN BKT state | 0.5144 | [0.5052, 0.5237] |
| DQN DKT state | 0.5151 | [0.5057, 0.5247] |
| DQN no state | 0.5105 | [0.5015, 0.5192] |

The canonical seed-42 DQN beats interleaving by **+0.0124** final mastery in
the paired comparison, with 95% CI **[+0.0064, +0.0184]**. That single run is
not the complete conclusion: the three DQN seeds average **0.5550** with
standard deviation **0.0208**, slightly below interleaving at **0.5576**.

## Interpretation

The DQN learned a functioning long-horizon policy and can be competitive with
a strong hand-designed schedule. The experiment does **not** establish that
DQN reliably beats that schedule because training-seed variation changes the
ranking.

Oracle state matters substantially: the matched seed-42 oracle policy reaches
0.5700, compared with 0.5144 for BKT, 0.5151 for DKT, and 0.5105 with no state.
Estimated state adds a consistent but small gain over no state, and DKT's
superior predictive AUC does not translate into a meaningful policy advantage
in the current single-focus-skill simulator.

With architecture and seed fixed, `gamma=1.0` reaches 0.5482 versus 0.5700 for
`gamma=0.99`. This is an optimization outcome, not a change to the undiscounted
evaluation metric.

## Double DQN follow-up

Changing only the bootstrap target, ten Double DQN agents average **0.5676**
final mastery versus **0.5474** for ten matched vanilla agents. The mean paired
training-seed difference is **+0.0202**, with 95% seed-bootstrap interval
**[+0.0104, +0.0300]**, and Double DQN wins 8/10 seeds.

The original three-seed stability signal does not survive the larger sample.
Across-seed standard deviations are 0.0147 for Double DQN and 0.0157 for
vanilla; their difference is -0.0010 with interval [-0.0101, +0.0065].

The mechanism diagnostic compares learned `max_a Q(s,a)` against 64 empirical
Monte Carlo greedy-policy returns at 100 states per checkpoint. On on-policy
states, vanilla bias is -2.50 and Double DQN bias is -5.14: both underestimate,
and Double DQN's absolute error is higher by +2.38 [0.83, 4.05]. Thus this
experiment supports a performance improvement, not improved stability or a
demonstrated reduction of positive Q overestimation. See
[`experiments_double_dqn.md`](experiments_double_dqn.md) for the controlled
protocol, diagnostics, and limitations.

## Reproduce or inspect

```powershell
python scripts/train_dqn.py --config configs/dqn_train.yaml --device cpu
python scripts/eval_dqn_suite.py --config configs/dqn_evaluation.yaml --suite full --device cpu
python scripts/eval_double_dqn.py --config configs/double_dqn_evaluation.yaml --device cpu
python scripts/eval_q_overestimation.py --config configs/q_overestimation.yaml --device cpu --state-bank-mode on_policy
python scripts/demo_policy_episode.py --replay docs/examples/dqn_episode_seed200000.json
```

The first two commands require the relevant local checkpoints and perform the
expensive experiment. The replay command is lightweight, trains nothing, and
works from the committed trace. See
[`experiments_dqn_phase4.md`](experiments_dqn_phase4.md) for all pairwise
results and [`artifacts.md`](artifacts.md) for provenance.
