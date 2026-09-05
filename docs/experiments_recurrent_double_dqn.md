# Feed-forward versus recurrent Double DQN

## Question

Does remembering a longer interaction history help Double DQN handle hidden
independence and fatigue and improve final simulated mastery?

This is an oracle-mastery simulator experiment. Mastery is visible to both
agents, but independence, fatigue, learner preferences, and response noise are
hidden. No intervention policy was evaluated on real students.

## Controlled protocol

Ten matched training seeds (42–51) compare the existing feed-forward Double
DQN with a GRU Double DQN. Both train for 5,000 episodes and share the same
simulator, six actions, reward, discount, epsilon schedule, target-network
schedule, optimizer settings, training episode seeds, 100 validation seeds,
and 500 held-out seeds (`200000..200499`).

The feed-forward network has two 128-unit hidden layers and 19,206 parameters.
The recurrent network uses a 64-unit GRU followed by a 64-unit head and has
19,910 parameters, only 3.7% more. The GRU carries hidden state through each
50-step episode and resets it between episodes.

Recurrent replay stores complete episodes so each Q estimate receives its
causal observation history. Each update uses three episodes, or 150 correlated
transitions, whereas feed-forward replay samples 128 independent transitions.
Both perform exactly 62,000 optimizer updates. Recurrent scheduled updates are
deferred to the end of the episode after its complete sequence is available.
This replay difference is necessary for this implementation but remains a
confound when attributing the result specifically to memory.

Two checkpoint estimands are reported:

- **Validation-selected:** the best checkpoint among evaluations every 100
  episodes.
- **Exact endpoint:** the network at episode 5,000, avoiding checkpoint-
  selection differences.

Architecture-level intervals use a paired bootstrap over the ten matched
training seeds, which are the replication unit.

## Results

| Checkpoint estimand | Feed-forward mean ± seed SD | GRU mean ± seed SD | GRU − feed-forward, seed-bootstrap 95% CI | GRU wins |
|---|---:|---:|---:|---:|
| Validation-selected | 0.5756 ± 0.0087 | **0.8374 ± 0.0068** | **+0.2618 [+0.2555, +0.2691]** | 10/10 |
| Exact episode 5,000 | 0.5414 ± 0.0122 | **0.8212 ± 0.0106** | **+0.2798 [+0.2684, +0.2907]** | 10/10 |

The result is not caused only by unequal checkpoint opportunities: it is
larger at the exact endpoint. Validation selection helps both architectures,
raising the GRU mean by 0.0161 and the feed-forward mean by 0.0342 relative to
their final networks.

Observed seed SD is lower for the GRU by 0.0019 for selected checkpoints and
0.0015 for endpoints. The bootstrap intervals, respectively [-0.0061,
+0.0030] and [-0.0080, +0.0046], include zero. The experiment therefore does
not establish a reduction in training-seed variability.

The learned behaviors also differ descriptively. Selected GRU policies use
`harder_problem` on 62.4% of steps and finish with mean independence 0.802,
compared with 37.4% and 0.406 for feed-forward policies. GRU policies use
`prerequisite_review` much less often (9.8% versus 30.4%). These statistics are
consistent with the recurrent policy tracking delayed learner dynamics, but
they are not a causal mechanism test.

## Conclusion

Under this implementation, longer-history GRU Double DQN decisively improves
simulated final mastery over the near-parameter-matched feed-forward agent.
The advantage appears for every matched seed and for both checkpoint
estimands. This supports using recurrent state in the tutoring POMDP.

The strongest defensible statement is:

> A GRU Double DQN trained with complete-episode replay substantially
> outperforms the matched feed-forward Double DQN in this partially observable
> simulator; the experiment does not isolate memory from recurrent replay or
> establish lower training-seed variance.

The comparison remains an oracle-state, hand-designed simulator study. It
does not establish classroom effectiveness, and the GRU result should not be
interpreted as evidence about real students.

## Matched-replay follow-up

The subsequent
[`memory ablation`](experiments_memory_ablation.md) retrains the feed-forward
network with the same complete-episode replay, 150 transitions per loss,
episode-boundary timing, and 62,000 updates. Its selected mean remains 0.5742,
while the GRU remains 0.8374, for a matched-seed difference of +0.2632
[0.2577, 0.2695]. That experiment removes the replay/training-scheme confound
identified above and provides strong evidence that access to history drives
the gain.

## Reproduction

Train all seeds using the recurrent configuration and a unique tag:

```powershell
py -3.10 scripts/train_recurrent_dqn.py --config configs/recurrent_double_dqn_train.yaml --device cpu --seed 42 --tag gru_seed42
py -3.10 scripts/eval_recurrent_double_dqn.py --config configs/recurrent_double_dqn_evaluation.yaml --device cpu
```

Repeat the training command for seeds 42 through 51. The evaluator requires
all recurrent and feed-forward 5,000-episode artifacts, audits their metadata,
and evaluates both estimands. Raw output is written to
`results/recurrent_double_dqn_comparison.json`, with a plot at
`results/figures/recurrent_double_dqn_comparison.png`; both remain gitignored.
The durable compact result is
[`examples/recurrent_double_dqn_results.json`](examples/recurrent_double_dqn_results.json).
