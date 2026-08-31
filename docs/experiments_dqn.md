# Phase 3: Oracle-State DQN

## Scope

This phase establishes the first learned sequential policy for the delayed
tutoring simulator. It intentionally trains only an **oracle-state DQN**. The
agent observes true simulated mastery, so this is an evaluation upper bound and
not a deployable student-facing policy. Comparison against all baselines and
estimated-state ablations belongs to Phase 4.

All results below are simulated. They are not evidence of learning gains for
real students.

## Method

The agent receives the fourteen-dimensional observation defined in
`docs/rl_environment.md`: eight observable/oracle state features plus a
six-way one-hot encoding of its previous intervention. Including the previous
action lets a feed-forward network represent action sequences such as
interleaving without exposing any additional latent simulator state.

The Q-network is a 14-128-128-6 ReLU MLP. Training uses standard DQN with:

- replay capacity 100,000, batch size 128, and 2,000-transition warm-up;
- Adam at 5e-4, Smooth L1 loss, and gradient clipping at 10;
- gamma 0.99, target synchronization every 500 learning updates;
- one learning update every four environment steps;
- epsilon annealed linearly from 1.0 to 0.05 over 60,000 steps;
- mastery-gain rewards multiplied by 100 only for numerical conditioning in
  the Bellman update. Reported returns remain in the environment's raw units.

The run used 2,000 training episodes (100,000 environment steps). Checkpoint
selection used 100 fixed validation episodes every 100 training episodes. The
selected checkpoint was then evaluated once on 500 held-out episodes. Seed
ranges were disjoint:

- training: 10,000-11,999;
- validation: 100,000-100,099;
- held-out: 200,000-200,499.

## Result

The best validation checkpoint occurred at episode 1,800.

| Evaluation | Episodes | Mean final mastery | Standard deviation | Mean return |
|---|---:|---:|---:|---:|
| Untrained network, validation | 100 | 0.4205 | 0.2507 | 0.1336 |
| Best DQN, validation | 100 | 0.5530 | 0.1821 | 0.2661 |
| Best DQN, held-out | 500 | **0.5700** | 0.1757 | **0.2728** |

On held-out episodes, overall correctness was 0.4405 and final-window
correctness was 0.4976. The greedy action mix was 29.75% worked examples,
29.59% harder problems, 39.71% prerequisite review, and less than 1% across
the remaining actions. Final simulated independence averaged 0.4579 and
fatigue averaged 0.2775.

The held-out mastery point estimate is slightly above the earlier calibrated
interleaving policy result (0.558), but these were not yet evaluated as a
paired, replicated comparison. Phase 4 performs that comparison in
`docs/experiments_dqn_phase4.md`; its multi-seed result does **not** support a
robust claim that DQN outperforms the heuristic.

## Interpretation and limitations

Validation mastery stayed close to 0.50 through much of training and improved
late, peaking at episode 1,800 before falling to 0.5331 at episode 2,000. This
shows that checkpoint selection is necessary and that one seed is not enough
to characterize DQN stability. The result is meaningful as evidence that the
implementation learns a nontrivial sequential policy, but it is not yet a
robust algorithm comparison.

Other limitations at the Phase 3 cutoff were deliberate:

- the policy receives oracle mastery rather than BKT/DKT-estimated state;
- simulator dynamics define the result and may reward unrealistic behavior;
- no multi-seed confidence interval over independently trained agents is
  reported;
- the six-action distribution is concentrated on three actions;
- no Double DQN, recurrent model, or hyperparameter ablation has been run.

Phase 4 addresses the training-seed, observation-state, and discount-factor
items. Double DQN and recurrent-policy comparisons remain future work.

## Reproduction

From the repository root with the Python 3.10 environment installed:

```powershell
python scripts/train_dqn.py --config configs/config.yaml --device cpu
```

The reference CPU run took about 106 seconds. It writes the best and final
checkpoints, a full JSON training trace, and a learning-curve figure under
`results/`. These generated artifacts are gitignored. The JSON records the
source commit, dirty-worktree flag, exact seed ranges, configuration, and
SHA-256 hashes of the environment, DQN, training script, and config used for
the run.
