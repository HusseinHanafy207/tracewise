# Memory ablation with matched complete-episode replay

## Question

Does the GRU advantage remain when a feed-forward Double DQN receives the same
complete-episode replay, 150 transitions per loss, episode-boundary update
timing, and 62,000 optimizer updates?

All outcomes come from the oracle-mastery simulator. Independence, fatigue,
learner preferences, and response noise remain hidden. No intervention policy
was evaluated on real students.

## Controlled protocol

Ten matched training seeds (42–51) compare:

- **Feed-forward episode replay:** the original two-layer 128-unit MLP, with
  complete episodes flattened into a memoryless transition loss.
- **GRU episode replay:** a 64-unit GRU and 64-unit head that consumes each
  episode causally and carries hidden state between timesteps.

Both conditions use Double DQN, 5,000 episodes, 250,000 environment steps,
three complete 50-step episodes per loss, 150 transitions per update, 62,000
updates, a 2,000-episode replay capacity, and updates deferred to episode
boundaries. Simulator, actions, observations, reward, discount, epsilon,
optimizer, target synchronization, training seeds, 100 validation seeds, and
500 held-out seeds (`200000..200499`) are identical.

The networks are near parameter matched: 19,206 parameters for the MLP and
19,910 for the GRU, a 3.7% difference. The training-artifact audit checks the
effective settings and recorded histories for every seed before evaluation.

Two estimands are retained:

- **Validation-selected:** best checkpoint among evaluations every 100
  episodes.
- **Exact endpoint:** the network at episode 5,000.

Architecture-level intervals use a paired bootstrap over the ten training
seeds, which are the replication unit.

## Results

| Checkpoint estimand | Feed-forward mean ± seed SD | GRU mean ± seed SD | GRU − feed-forward, seed-bootstrap 95% CI | GRU wins |
|---|---:|---:|---:|---:|
| Validation-selected | 0.5742 ± 0.0053 | **0.8374 ± 0.0068** | **+0.2632 [+0.2577, +0.2695]** | 10/10 |
| Exact episode 5,000 | 0.5525 ± 0.0133 | **0.8212 ± 0.0106** | **+0.2687 [+0.2563, +0.2805]** | 10/10 |

The episode-replay MLP does not jump toward the GRU. Its selected mean is
0.5742, essentially the same as the original independent-transition MLP mean
of 0.5756. At the endpoint it reaches 0.5525 rather than the original 0.5414,
but this modest difference is far smaller than the recurrent advantage.

The conclusion also does not depend on validation checkpoint selection. The
exact endpoint advantage is slightly larger, and the GRU wins every matched
seed under both estimands.

Seed variability is not the central result. The GRU selected-checkpoint SD is
0.0015 higher, with bootstrap interval [-0.0021, +0.0059]. Its endpoint SD is
0.0027 lower, with interval [-0.0091, +0.0046]. Neither difference is
established.

## Conclusion

Matching the replay representation, loss-batch size, update timing, update
count, and all environment/evaluation settings does not reproduce the GRU
gain in a memoryless MLP. This removes the main replay/training-scheme confound
from the previous experiment and provides strong evidence that access to
interaction history drives the improvement in this POMDP.

The defensible statement is:

> With complete-episode replay and training exposure matched, GRU Double DQN
> improves simulated final mastery by about 0.26 over a near-parameter-matched
> feed-forward Double DQN and wins all ten training seeds.

This is still not an absolute causal proof about the hidden state alone. The
architectures differ in their computation and by 3.7% in parameter count. A
GRU trained and evaluated with hidden state reset at every timestep would be a
stricter architecture-preserving ablation. It is optional follow-up work, not
part of this result.

## Reproduction

Train each feed-forward control seed from 42 through 51 with a unique tag:

```powershell
py -3.10 scripts/train_episode_replay_dqn.py --config configs/episode_replay_double_dqn_train.yaml --device cpu --seed 42 --tag episode_replay_seed42
py -3.10 scripts/eval_memory_ablation.py --config configs/memory_ablation_evaluation.yaml --device cpu
```

The evaluator requires the ten matched feed-forward and GRU training
artifacts, audits them, and evaluates both checkpoint estimands. Raw output is
written to `results/memory_ablation_comparison.json`, with a plot at
`results/figures/memory_ablation_comparison.png`; both remain gitignored. The
durable summary is
[`examples/memory_ablation_results.json`](examples/memory_ablation_results.json).
