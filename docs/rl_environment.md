# Sequential RL environment specification

## Purpose and scope

`src/rl/environment.py` turns the existing student simulator into a fixed-
horizon episodic decision process for Phase 3 DQN experiments. It follows the
Gymnasium `reset`/`step` return convention without adding Gymnasium as a
dependency.

This remains a simulated research environment. Its outcomes are not evidence
about real students.

## POMDP definition

One episode is one simulated student completing `H=50` interactions. At each
step, the agent chooses one of six teaching interventions and receives the
resulting mastery change.

### Hidden environment state

The generative state contains true mastery, independence, fatigue, latent
learner preferences, problem difficulty, focus skill, and response history.
Independence, fatigue, and learner preferences are never observation features.
True mastery is hidden in deployable modes. This makes the deployed problem a
POMDP rather than a fully observed MDP.

### Observation

The agent receives a fourteen-dimensional vector:

1. mastery signal: oracle truth, BKT/DKT estimate, or zero in the ablation
2. recent accuracy over the last five responses
3. base difficulty of the next problem
4. normalized episode progress `t/H`
5. accuracy improvement between the last two five-response windows
6. recent worked-example success rate
7. consecutive failures divided by five
8. recent Socratic-hint success rate
9. six one-hot features identifying the agent's previous intervention

Observation modes are explicit:

- `oracle`: true mastery; diagnostic upper bound only
- `estimated`: a fresh online BKT/DKT tracker per episode
- `no_state`: student-response features zeroed; difficulty, progress, and the
  agent's own previous-action one-hot stay

`reset()` and `step()` omit hidden diagnostics by default. Evaluation code may
set `include_diagnostics=True`, but diagnostic values must never be passed to a
policy or replay buffer.

### Actions

The action id is the stable index in `configs/config.yaml -> policy.actions`:

0. explain
1. worked example
2. Socratic hint
3. easier problem
4. harder problem
5. prerequisite review

The environment also accepts action names for readable baseline validation.
DQN uses integer ids.

### Transition and response

The existing simulator remains the single source of dynamics. The intervention
has a mastery- and learner-dependent effect; delayed mode scales learning by
hidden independence and fatigue; difficulty affects the sampled response; and
the KT tracker receives only `(focus_skill, correct)` after the response. The
next observation is built after the transition and tracker update.

### Reward and horizon

The immediate reward is `r_t = mastery_(t+1) - mastery_t`. There are currently
no intervention costs or hint penalties. With undiscounted return, rewards
telescope to `m_H - m_0`, so maximizing return is exactly maximizing final
mastery. Phase 3 starts with `gamma=0.99`; Phase 4 must include a
discount-factor ablation because discounting mildly favors earlier gains.

The episode terminates naturally after 50 interactions. It is not reported as
a time-limit truncation.

## Leakage boundary

Deployable policies may consume only the observation returned by the
environment. They may not access the underlying `SimulatedStudent`, the
diagnostic `info` block, true rewards before acting, or tracker internals.

Oracle policies are labeled upper bounds. Evaluation may read diagnostics only
after choosing an action and only for aggregate reporting.

## Calibration gate before DQN

Run:

```powershell
python scripts/validate_rl_env.py --config configs/config.yaml --n-episodes 500
```

The environment is ready for DQN only if all three checks pass:

- delayed effects off: myopic oracle beats explain/harder interleaving on final mastery
- delayed effects on: interleaving beats myopic oracle on final mastery
- delayed effects on: myopic oracle still wins early reward

The script writes `results/rl_env_validation.json`. It performs no DQN training.

### Phase 2 validation result

The gate was run over 500 paired episode seeds (`42..541`) with horizon 50:

- Delayed effects off: myopic oracle reached final mastery `1.000`; interleaving
  reached `0.831`.
- Delayed effects on: myopic oracle reached final mastery `0.518`; interleaving
  reached `0.558`.
- With delays on, myopic still had higher early reward (`0.0128` versus
  `0.0072`), while interleaving had higher late reward (`0.0036` versus
  `0.0023`).
- All three calibration checks passed.

The generated JSON records the HEAD commit, dirty/clean status, and SHA-256
fingerprints for the config, environment, simulator, and validation script. The
metrics above were produced from the Phase 2 working tree. Rerun the command
after committing so the ignored local artifact records a clean HEAD commit.

## Current limitation

Each simulated student still has one focus skill. This is sufficient for the
first delayed-credit DQN experiment but does not exercise DKT's cross-skill
representation advantage. A later research-strengthening phase will introduce
multiple skills and prerequisites.
