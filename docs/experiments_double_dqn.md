# Double DQN: ten-seed performance and mechanism study

## Questions

1. Does Double DQN improve held-out policy performance or training-seed
   stability over vanilla DQN when every other choice is fixed?
2. If behavior changes, is it accompanied by the reduction in positive
   `max_a Q(s,a)` bias that Double DQN was designed to address?

All outcomes are from the oracle-state student simulator. No intervention
policy was evaluated on real students.

## Controlled algorithm change

Vanilla DQN selects and evaluates its bootstrap action with the target network:

`max_a Q_target(s', a)`

Double DQN selects with the online network and evaluates with the target
network:

`Q_target(s', argmax_a Q_online(s', a))`

Only `dqn.double_dqn` changes. Network, optimizer, replay, target sync, epsilon
schedule, reward scaling, `gamma=0.99`, simulator, and checkpoint selection are
identical. This follows the separation introduced by [van Hasselt, Guez, and
Silver (2015)](https://arxiv.org/abs/1509.06461).

## Protocol

- 10 matched training seeds: 42–51;
- 2,000 episodes and exactly 100,000 environment steps per agent;
- 100 fixed validation episodes for checkpoint selection;
- 500 common held-out episodes (`200000..200499`) per checkpoint;
- paired episode bootstrap within each fixed pair of agents;
- paired training-seed bootstrap for algorithm-level mean and SD differences.

The evaluator verifies the algorithm flag embedded in every checkpoint. The
training-seed bootstrap is the primary algorithm-level uncertainty estimate;
the 500 episodes within one trained agent are not treated as independent model
replications.

## Held-out performance

| Seed | Vanilla | Double DQN | Double minus vanilla, paired episode 95% CI |
|---:|---:|---:|---:|
| 42 | 0.5700 | 0.5686 | -0.0014 [-0.0029, +0.0002] |
| 43 | 0.5313 | 0.5686 | +0.0373 [+0.0313, +0.0435] |
| 44 | 0.5637 | 0.5588 | -0.0049 [-0.0068, -0.0032] |
| 45 | 0.5312 | 0.5715 | +0.0403 [+0.0350, +0.0461] |
| 46 | 0.5567 | 0.5934 | +0.0368 [+0.0289, +0.0447] |
| 47 | 0.5323 | 0.5657 | +0.0334 [+0.0275, +0.0394] |
| 48 | 0.5397 | 0.5604 | +0.0207 [+0.0158, +0.0260] |
| 49 | 0.5634 | 0.5847 | +0.0212 [+0.0171, +0.0254] |
| 50 | 0.5316 | 0.5387 | +0.0070 [+0.0042, +0.0100] |
| 51 | 0.5540 | 0.5656 | +0.0116 [+0.0081, +0.0151] |

| Algorithm | Ten-seed mean | Across-seed SD | Range |
|---|---:|---:|---:|
| Vanilla DQN | 0.5474 | 0.0157 | 0.0388 |
| Double DQN | **0.5676** | 0.0147 | 0.0547 |

Double DQN wins 8/10 matched seeds. Its mean seed-level gain is **+0.0202**,
with 95% training-seed bootstrap interval **[+0.0104, +0.0300]**. This supports
a performance advantage under the fixed simulator and training protocol.

The initial three-seed SD reduction does not replicate. The SD difference is
-0.0010 with training-seed bootstrap interval **[-0.0101, +0.0065]**; the
Double DQN range is actually larger. There is no supported stability claim.

## Q-overestimation diagnostic

For 100 states per checkpoint (20 held-out episodes × steps 0, 10, 20, 30,
and 40), the diagnostic records the learned online-network `max_a Q(s,a)`.
From a fork of the exact simulator state, it then takes that greedy action and
runs 64 independently seeded continuations under the checkpoint's greedy
policy. Returns use the checkpoint's reward scale and `gamma`, so Q and Monte
Carlo values share units. Signed bias is:

`learned max Q - mean Monte Carlo discounted return`

Positive values indicate overestimation and negative values underestimation.
The original paper similarly compared learned action values with empirical
discounted returns along evaluation trajectories.

Two state distributions guard against an interpretation artifact:

- **Shared interleaving bank:** every checkpoint sees identical latent states,
  generated independently of either learned policy.
- **On-policy bank:** each checkpoint supplies states reached by its own greedy
  policy. Episode seeds and sampled steps match, but latent states are not
  paired across algorithms. This is closer to the paper's evaluation approach.

| State bank | Vanilla bias | Double bias | Double − vanilla bias, seed-bootstrap 95% CI | Vanilla MAE | Double MAE | Double − vanilla MAE, seed-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Shared interleaving | -7.34 | -8.78 | -1.43 [-3.44, -0.07] | 7.63 | 9.00 | +1.37 [+0.05, +3.30] |
| On-policy | -2.50 | -5.14 | -2.65 [-4.43, -1.01] | 3.09 | 5.47 | +2.38 [+0.83, +4.05] |

Both diagnostics show **underestimation**, not the positive maximization bias
the proposed explanation requires. Double DQN is more negative and has larger
absolute calibration error. The on-policy overestimation rate falls from 0.416
to 0.214, but that occurs by shifting farther below the empirical return; it is
not improved value calibration.

## Conclusion

The expanded experiment supports this claim:

> In this oracle-state tutoring simulator, Double DQN improves held-out final
> mastery across 10 matched training seeds, but does not measurably improve
> training-seed stability, and the improvement is not explained by reduced
> positive Q overestimation in the Monte Carlo diagnostic.

This is a useful negative mechanism result. Possible contributors include the
short bounded horizon, checkpoint selection, target-network dynamics, and the
simulator's reward/state structure. They are hypotheses, not findings. A
descriptive correlation between bias and performance is not causal, so no
`bias → stability → performance` chain is claimed.

## Reproduction

Train missing seeds with the same commands, varying `--seed` from 42 through
51 and using a matching `--tag`:

```powershell
python scripts/train_dqn.py --config configs/dqn_train.yaml --observation-mode oracle --seed 45 --tag seed45 --device cpu
python scripts/train_dqn.py --config configs/double_dqn_train.yaml --observation-mode oracle --seed 45 --tag double_seed45 --device cpu
python scripts/eval_double_dqn.py --config configs/double_dqn_evaluation.yaml --device cpu
python scripts/eval_q_overestimation.py --config configs/q_overestimation.yaml --device cpu --state-bank-mode shared_interleave
python scripts/eval_q_overestimation.py --config configs/q_overestimation.yaml --device cpu --state-bank-mode on_policy
```

Raw outputs are written to `results/double_dqn_comparison.json`,
`results/q_overestimation_diagnostic.json`, and
`results/q_overestimation_diagnostic_on_policy.json`; plots are under
`results/figures/`. These remain gitignored. The durable compact summary is
[`examples/double_dqn_results.json`](examples/double_dqn_results.json).
