# Phase 4: Controlled DQN Evaluation

## Question

Phase 3 showed that one oracle-state DQN checkpoint reached final simulated
mastery 0.5700. Phase 4 asks whether that result survives stronger controls:

1. Does DQN beat baselines on the same simulated students?
2. Is the result stable across independent DQN initialization/exploration
   seeds?
3. What is lost when true mastery is replaced by online BKT/DKT estimates or
   removed entirely?
4. Does the Bellman discount of 0.99 materially change the result relative to
   the environment's undiscounted objective?

All outcomes are produced by the simulator. Oracle mastery is an evaluation
upper bound, and none of these results demonstrate effects on real students.

## Protocol

Every DQN trains for 2,000 episodes and selects its checkpoint using the same
100 validation episode seeds. Final comparisons use the same 500 episode seeds
(`200000..200499`) for every policy. Those episodes are excluded from training
and checkpoint selection.

The comparison retains one row per policy and episode. Reported 95% intervals
are deterministic percentile-bootstrap intervals over the 500 paired episodes.
Candidate-minus-reference intervals bootstrap the within-episode differences,
not two unrelated samples.

The controlled conditions are:

- oracle DQN with algorithm seeds 42, 43, and 44;
- oracle seed 42 with `gamma=0.99` and `gamma=1.0`;
- independently trained oracle, BKT, DKT, and no-state DQNs at seed 42;
- random, oracle rule-based, myopic oracle, explain/harder interleaving, and
  always-explain baselines.

BKT/DKT policies receive a fresh online tracker per episode. Trackers update
only from observed `(focus_skill, correct)` pairs. True simulator mastery,
independence, and fatigue are read only after action selection for aggregate
evaluation.

## Held-out results

| Policy | Mean final mastery | Episode-bootstrap 95% CI |
|---|---:|---:|
| Random | 0.4533 | [0.4432, 0.4635] |
| Rule oracle | 0.4928 | [0.4832, 0.5022] |
| Myopic oracle | 0.5196 | [0.5110, 0.5285] |
| Explain/harder interleaving | **0.5576** | [0.5431, 0.5725] |
| Always explain | 0.4493 | [0.4414, 0.4572] |
| DQN oracle, seed 42 | **0.5700** | [0.5551, 0.5855] |
| DQN oracle, seed 43 | 0.5313 | [0.5211, 0.5413] |
| DQN oracle, seed 44 | 0.5637 | [0.5490, 0.5783] |
| DQN oracle, gamma 1.0 | 0.5482 | [0.5347, 0.5618] |
| DQN BKT state | 0.5144 | [0.5052, 0.5237] |
| DQN DKT state | 0.5151 | [0.5057, 0.5247] |
| DQN no state | 0.5105 | [0.5015, 0.5192] |

For the canonical seed-42 DQN, paired final-mastery differences were:

| Reference | Mean DQN difference | Paired 95% CI | DQN win rate |
|---|---:|---:|---:|
| Random | +0.1166 | [+0.1092, +0.1242] | 99.6% |
| Rule oracle | +0.0771 | [+0.0704, +0.0844] | 100.0% |
| Myopic oracle | +0.0504 | [+0.0422, +0.0584] | 39.0% |
| Interleaving | **+0.0124** | **[+0.0064, +0.0184]** | 60.2% |
| Always explain | +0.1207 | [+0.1129, +0.1288] | 100.0% |

The positive mean against myopic despite a 39% win rate indicates a skewed
effect: the DQN produces larger gains for a minority of episodes rather than a
small improvement for most students. Reporting only the mean would hide that
behavior.

## Training-seed stability

The three oracle DQNs have held-out means 0.5700, 0.5313, and 0.5637. Their
mean across independently trained agents is **0.5550**, with standard deviation
**0.0208** across only three training seeds. Interleaving reaches 0.5576.

Paired differences against interleaving are:

- seed 42: +0.0124, 95% CI [+0.0064, +0.0184];
- seed 43: -0.0263, 95% CI [-0.0323, -0.0203];
- seed 44: +0.0061, 95% CI [+0.0003, +0.0121].

Therefore, the defensible conclusion is not "DQN beats interleaving." One of
three agents loses clearly, and the across-agent mean is slightly below the
heuristic. Phase 4 demonstrates a working learned sequential policy and exposes
its instability.

## State and discount ablations

Against the matched oracle seed-42 DQN, BKT, DKT, and no-state DQNs lose 0.0556,
0.0548, and 0.0595 final mastery respectively; all paired intervals exclude
zero. Accurate state is valuable under the simulator's assumptions.

However, estimated state adds little over no state:

- BKT minus no-state: +0.00395, 95% CI [+0.00249, +0.00545];
- DKT minus no-state: +0.00466, 95% CI [+0.00315, +0.00627].

The effects are consistent but small. BKT and DKT are effectively tied at the
policy level despite DKT's stronger predictive AUC. The simulator still uses a
single focus skill, so it cannot exploit DKT's cross-skill representation.

With architecture and seed fixed, `gamma=1.0` loses 0.0218 final mastery to
`gamma=0.99`, with paired 95% CI [-0.0266, -0.0172]. This is an optimization
result, not evidence that discounting changes the undiscounted evaluation
metric: the two discounts led training to different policies.

## What this phase establishes

The project now has a reproducible sequential-RL evaluation rather than a
single favorable checkpoint. Its strongest portfolio-level points are:

- DQN can discover a policy competitive with a hand-designed long-horizon
  interleaving heuristic;
- paired evaluation detects a small canonical advantage that an unpaired
  comparison could not characterize cleanly;
- repeated training reveals meaningful instability and prevents an overstated
  superiority claim;
- oracle-to-estimated and estimated-to-no-state gaps quantify the current
  student-state bottleneck.

Remaining RL improvements include more training seeds, recurrent DQN for the
POMDP, Double DQN, a multi-skill/prerequisite simulator, and joint tuning of the
KT signal and policy. These are future improvements, not prerequisites for the
Phase 4 result.

## Reproduction

The full experiment uses `configs/dqn_evaluation.yaml`, which inherits the
canonical training settings from `configs/dqn_train.yaml` and the common
project settings from `configs/config.yaml`.
Starting from the Phase 3 checkpoint and existing BKT/DKT checkpoints, run:

```powershell
python scripts/train_dqn.py --config configs/dqn_train.yaml --observation-mode oracle --seed 43 --tag seed43 --device cpu
python scripts/train_dqn.py --config configs/dqn_train.yaml --observation-mode oracle --seed 44 --tag seed44 --device cpu
python scripts/train_dqn.py --config configs/dqn_train.yaml --observation-mode oracle --gamma 1.0 --tag gamma1 --device cpu
python scripts/train_dqn.py --config configs/dqn_train.yaml --observation-mode bkt --device cpu
python scripts/train_dqn.py --config configs/dqn_train.yaml --observation-mode dkt --device cpu
python scripts/train_dqn.py --config configs/dqn_train.yaml --observation-mode no_state --device cpu
python scripts/eval_dqn_suite.py --config configs/dqn_evaluation.yaml --suite full --device cpu
```

Generated checkpoints, JSON traces, and plots remain under the gitignored
`results/` directory. `results/dqn_phase4_evaluation.json` contains raw
per-episode metrics, action frequencies, paired effects, source/checkpoint
hashes, and git provenance. The durable plot is regenerated as
`results/figures/dqn_phase4_evaluation.png`.
