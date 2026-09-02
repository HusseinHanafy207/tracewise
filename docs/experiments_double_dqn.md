# Double DQN: controlled three-seed comparison

## Question

Does Double DQN improve performance or training-seed stability over the
project's vanilla DQN when every other training and evaluation choice is held
fixed?

**Scope:** all outcomes are from the oracle-state student simulator. No
intervention policy was evaluated on real students.

## Controlled change

Vanilla DQN selects and evaluates the bootstrap action with the target network:

`max_a Q_target(s', a)`

Double DQN selects with the online network and evaluates with the target
network:

`Q_target(s', argmax_a Q_online(s', a))`

The comparison changes only `dqn.double_dqn` from `false` to `true`. Both
algorithms use the same:

- `[128, 128]` network, optimizer, learning rate, replay buffer, and target-sync
  interval;
- epsilon schedule, reward scaling, `gamma=0.99`, and delayed-effect simulator;
- 2,000 training episodes per agent and training seeds 42, 43, and 44;
- 100 fixed validation episodes for checkpoint selection;
- 500 held-out episodes (`200000..200499`) shared by every checkpoint;
- 5,000 deterministic bootstrap resamples for episode-level intervals.

Each generated history contains exactly 2,000 rows and ends at 100,000
environment steps. The evaluator also verifies the `double_dqn` flag embedded
in every checkpoint before running it.

## Held-out performance

| Training seed | Vanilla DQN | Double DQN | Double minus vanilla, paired 95% CI |
|---:|---:|---:|---:|
| 42 | 0.5700 | 0.5686 | -0.0014 [-0.0029, +0.0002] |
| 43 | 0.5313 | 0.5686 | +0.0373 [+0.0313, +0.0435] |
| 44 | 0.5637 | 0.5588 | -0.0049 [-0.0068, -0.0032] |

The interval in each row bootstraps paired differences over the same 500
simulated episodes for that training seed. It measures episode variation for a
fixed pair of trained agents; it does not measure uncertainty across training
seeds.

## Stability across trained agents

| Algorithm | Three-seed mean | Across-seed std | Across-seed range | Selected checkpoint episodes |
|---|---:|---:|---:|---|
| Vanilla DQN | 0.5550 | 0.0208 | 0.0387 | 1800, 2000, 1800 |
| Double DQN | **0.5653** | **0.0057** | **0.0098** | 1300, 2000, 1600 |

Double DQN improves the observed mean by **+0.0103** and reduces the observed
across-seed standard deviation by about **73%**. Its range is also roughly 75%
smaller.

The result is promising for stability but not decisive for superiority. Double
DQN is better for only one of the three matched training seeds: the mean gain
mostly comes from fixing vanilla seed 43's weak run. Seed 42 is statistically
indistinguishable at the episode level, while seed 44 favors vanilla. With only
three training seeds, the variance estimate itself is uncertain.

## Conclusion

Under this simulator and training budget, Double DQN produced a substantially
tighter cluster of final-mastery results without lowering the across-agent
mean. It is therefore the more stable observed variant in this experiment.
The defensible claim is **"Double DQN improved observed three-seed stability"**,
not **"Double DQN always outperforms vanilla DQN."** More independent training
seeds would be required for a strong algorithm-level performance claim.

## Reproduction

```powershell
python scripts/train_dqn.py --config configs/double_dqn_train.yaml --observation-mode oracle --seed 42 --tag double_seed42 --device cpu
python scripts/train_dqn.py --config configs/double_dqn_train.yaml --observation-mode oracle --seed 43 --tag double_seed43 --device cpu
python scripts/train_dqn.py --config configs/double_dqn_train.yaml --observation-mode oracle --seed 44 --tag double_seed44 --device cpu
python scripts/eval_double_dqn.py --config configs/double_dqn_evaluation.yaml --device cpu
```

The raw result is written to `results/double_dqn_comparison.json`, and the plot
to `results/figures/double_dqn_comparison.png`. These remain gitignored. A
lightweight durable summary is committed as
[`examples/double_dqn_results.json`](examples/double_dqn_results.json).
