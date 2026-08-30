# Adaptive policy comparison (simulated)

**Disclaimer:** Every number below is a *simulated* learning outcome under
the state-dependent intervention-effect assumptions in
`src/policy/simulator.py`. This is offline evaluation for comparing
decision rules. It is **not** evidence that any policy improves real
student learning.

## Setup

| | |
|---|---|
| Population | 500 simulated students / seed |
| Horizon | 50 interactions / student |
| Seeds | 5 (42–46); report mean ± std |
| Actions | explain, worked_example, socratic_hint, easier_problem, harder_problem, prerequisite_review |
| Reward | `r_t = mastery_{t+1} - mastery_t` |
| Context (8-d) | mastery, recent_acc, difficulty, attempts_norm, improvement, (unused), failures_norm, (unused) |

### Why effects are state-dependent

If every intervention had a *fixed* mastery bump, the optimal policy would
collapse to “always pick the largest bump,” and a contextual bandit would
have nothing real to learn. Effects are therefore functions of current
mastery (stated modeling assumption):

- low mastery → prerequisite / explain help most
- mid mastery → worked example / Socratic hint help most
- high mastery → harder problem helps; easier problem hurts

Easier/harder actions also shift problem difficulty (±0.15).

## Policies

1. **random** — uniform over actions
2. **rule_based** — thresholds on mastery + consecutive failures
3. **eps_greedy** — linear contextual bandit, ε=0.1, full state
4. **linucb** — LinUCB (Li et al. 2010), α=1.0, full state
5. **eps_greedy_no_state** — ablation: same bandit, mastery / accuracy /
   improvement / failures zeroed in the context

## Results (local, 2026-08-26)

`final-window acc` is defined as mean accuracy over the last 5 timesteps
(t=46–50), averaged across students.

| Policy | mean mastery gain | final mastery | overall acc | final-window acc |
|---|---|---|---|---|
| random | 0.0112 ± 0.0001 | 0.856 ± 0.002 | 0.634 ± 0.004 | 0.794 ± 0.008 |
| rule_based | **0.0141 ± 0.0002** | **1.000 ± 0.000** | 0.683 ± 0.003 | 0.798 ± 0.010 |
| eps_greedy | 0.0137 ± 0.0002 | 0.979 ± 0.003 | 0.683 ± 0.004 | 0.807 ± 0.009 |
| linucb | 0.0135 ± 0.0002 | 0.969 ± 0.001 | **0.700 ± 0.003** | **0.828 ± 0.010** |
| eps_greedy_no_state | 0.0136 ± 0.0002 | 0.978 ± 0.003 | 0.675 ± 0.010 | 0.804 ± 0.013 |

### Action mix (fraction of steps)

| Policy | explain | worked_example | socratic_hint | easier | harder | prerequisite |
|---|---|---|---|---|---|---|
| random | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 |
| rule_based | 0.07 | 0.20 | 0.00 | 0.00 | 0.63 | 0.11 |
| eps_greedy | 0.26 | 0.09 | 0.24 | 0.02 | 0.26 | 0.14 |
| linucb | 0.13 | 0.18 | 0.22 | 0.05 | 0.26 | 0.16 |
| eps_greedy_no_state | 0.32 | 0.14 | 0.24 | 0.02 | 0.24 | 0.04 |

Figures: `results/figures/policy_learning_curves.png`,
`results/figures/policy_action_mix.png`.

## How to read this (interview)

1. **Random is the floor.** Every structured policy beats it on mastery and
   accuracy — the decision problem is not noise.
2. **Rule-based maximizes mastery under these assumptions.** Its thresholds
   line up with the same pedagogical regimes encoded in the simulator, and
   it saturates at mastery ≈ 1.0 (then mostly assigns harder problems). That
   is expected when the hand-written rule matches the generative process —
   say so; do not overclaim.
3. **LinUCB wins on accuracy.** It trades a bit of mastery for a better
   correct-rate by mixing difficulty-shifting actions. Reward was mastery
   gain; accuracy is a secondary outcome — reporting both is the honest move.
4. **Ablation (state vs no-state).** `eps_greedy` vs `eps_greedy_no_state`:
   similar mastery, but with-state is better on accuracy (0.683 vs 0.675
   overall; 0.807 vs 0.804 final window). Conditioning on the student state
   helps; the gap is modest because the bandit can still learn a decent
   open-loop mix from difficulty/attempts alone.
5. **RL framing.** This is a *contextual bandit* (one-step reward, no
   long-horizon value function), not PPO. That is the right first RL method
   for tutoring intervention selection.

## What this does *not* show

- Real classroom learning gains
- That LinUCB would win if the reward were accuracy instead of mastery gain

---

## M5.5 — Closing the loop: estimated KT state (not oracle mastery)

**Gap closed:** In M5 the policy saw *true* simulator mastery. In a real system
the policy only sees what a KT model can infer from `(skill_id, correct)`.
M5.5 re-runs the same simulator with online BKT / DKT trackers feeding the
policy context. Reward remains true mastery gain (environment feedback);
estimated modes **never** put true mastery into the policy context.

| | |
|---|---|
| Script | `scripts/eval_kt_policies.py` |
| Trackers | `src/policy/kt_state.py` (`BKTOnlineTracker`, `DKTOnlineTracker`) |
| Checkpoints | `results/checkpoints/bkt.pkl`, `results/checkpoints/dkt_best.pt` |
| Skills | 123 ASSISTments skill IDs assigned per simulated student |
| Domain caveat | DKT/BKT were fit on real ASSISTments; the generative process here is still the abstract mastery simulator — expect non-zero \|est−true\| |

### Conditions

| Condition | Policy | State to policy |
|---|---|---|
| `linucb_oracle` | LinUCB | true mastery (upper bound) |
| `linucb_bkt` / `linucb_dkt` | LinUCB | online BKT / DKT estimate |
| `linucb_no_state` | LinUCB | mastery / acc / improvement / failures zeroed |
| `rule_oracle` | rule-based | true mastery |
| `rule_bkt` / `rule_dkt` | rule-based | online BKT / DKT estimate |

### Results (local, 2026-08-27)

500 students × 50 steps × 5 seeds; DKT checkpoint val AUC ≈ 0.8383 (local CPU).

| Condition | mastery gain | final mastery | overall acc | final-window acc | \|est−true\| |
|---|---|---|---|---|---|
| linucb_oracle | 0.0134 ± 0.0001 | 0.969 ± 0.001 | 0.703 ± 0.007 | 0.826 ± 0.008 | n/a |
| linucb_bkt | 0.0134 ± 0.0001 | 0.966 ± 0.002 | 0.699 ± 0.008 | 0.825 ± 0.004 | 0.155 |
| linucb_dkt | 0.0134 ± 0.0001 | 0.967 ± 0.001 | 0.701 ± 0.007 | 0.823 ± 0.007 | 0.205 |
| linucb_no_state | 0.0134 ± 0.0001 | 0.966 ± 0.001 | 0.705 ± 0.007 | 0.827 ± 0.006 | n/a |
| rule_oracle | **0.0140 ± 0.0001** | **1.000 ± 0.000** | 0.686 ± 0.007 | 0.798 ± 0.006 | n/a |
| rule_bkt | 0.0130 ± 0.0002 | 0.946 ± 0.002 | 0.620 ± 0.007 | 0.772 ± 0.006 | 0.179 |
| rule_dkt | 0.0128 ± 0.0001 | 0.940 ± 0.004 | 0.650 ± 0.009 | 0.797 ± 0.007 | 0.215 |

Figure: `results/figures/kt_policy_learning_curves.png`.

### How to read M5.5

1. **Rule-based needs accurate state.** Oracle saturates mastery; BKT/DKT
   estimates drop final mastery (~0.94–0.95) and overall accuracy (especially
   BKT). Threshold rules misfire when \(\hat P(mastery)\) is biased.
2. **LinUCB is robust here.** Oracle / BKT / DKT / no-state are essentially
   tied on mastery gain. With a scalar reward each step, the bandit can learn
   a good action mix even under noisy or missing state — unlike hard-coded
   thresholds.
3. **\|est−true\| is real.** Mean absolute error ≈ 0.15–0.22; BKT tracks this
   simulator slightly closer than DKT (domain mismatch: DKT saw ASSISTments
   dynamics, not this generative process).
4. **Oracle is an upper bound, not a deployable mode.** Interview line:
   “We closed the observability gap; rule policies degrade under partial
   observability; bandits degrade less.”

### Reproduce M5.5

```bash
# need bkt.pkl + dkt_best.pt under results/checkpoints/
python scripts/eval_kt_policies.py --config configs/config.yaml
```

Writes `results/kt_policy_comparison.json` and the figure above.

### Robustness check (2026-08-27)

One larger Monte Carlo run (same generative process; not a new phase):

```bash
python scripts/eval_kt_policies.py --config configs/config.yaml \
  --n-students 2000 --n-seeds 10 --suite linucb_rule --tag robustness
```

| Condition | mastery gain | final mastery | overall acc | final-window acc | \|est−true\| |
|---|---|---|---|---|---|
| linucb_oracle | 0.0136 ± 0.0001 | 0.977 ± 0.001 | 0.709 ± 0.002 | 0.830 ± 0.003 | n/a |
| linucb_bkt | 0.0136 ± 0.0001 | 0.981 ± 0.001 | 0.703 ± 0.003 | 0.822 ± 0.004 | 0.157 |
| linucb_dkt | 0.0136 ± 0.0001 | 0.979 ± 0.001 | 0.706 ± 0.002 | 0.822 ± 0.003 | 0.210 |
| linucb_no_state | 0.0137 ± 0.0001 | 0.984 ± 0.000 | 0.710 ± 0.002 | 0.822 ± 0.003 | n/a |
| rule_oracle | 0.0140 ± 0.0001 | 1.000 ± 0.000 | 0.685 ± 0.002 | 0.798 ± 0.003 | n/a |
| rule_bkt | 0.0129 ± 0.0001 | 0.947 ± 0.002 | 0.620 ± 0.002 | 0.773 ± 0.003 | 0.178 |
| rule_dkt | 0.0128 ± 0.0001 | 0.941 ± 0.001 | 0.648 ± 0.003 | 0.797 ± 0.002 | 0.213 |

**Verdict:** conclusions stable. LinUCB remains essentially flat across oracle /
BKT / DKT / no-state; rule-based still needs accurate state. Stop increasing N —
next bottleneck is homogeneous `Effect(a,m)`.

---

## M5.6 — Heterogeneous learners (latent \(z_i\))

**Change:** `Effect(a, m, z_i) = z_i[a] · Effect(a, m)` with three latent types
(`example_preferring`, `socratic_preferring`, `challenge_seeking`). The
simulator samples \(z_i\); the policy never sees it. Bandit context dims 5/7
carry recent worked-example / Socratic success rates so type can be inferred
from experience only.

```bash
python scripts/eval_kt_policies.py --config configs/config.yaml \
  --heterogeneous --suite linucb_rule --tag hetero --n-students 500 --n-seeds 5
```

| Condition | mastery gain | final mastery | overall acc | final-window acc | \|est−true\| |
|---|---|---|---|---|---|
| linucb_oracle | 0.0130 ± 0.0001 | 0.950 ± 0.002 | 0.695 ± 0.004 | 0.819 ± 0.008 | n/a |
| linucb_bkt | 0.0130 ± 0.0001 | 0.951 ± 0.001 | 0.691 ± 0.004 | 0.814 ± 0.009 | 0.153 |
| linucb_dkt | 0.0130 ± 0.0001 | 0.949 ± 0.002 | 0.691 ± 0.004 | 0.817 ± 0.006 | 0.201 |
| linucb_no_state | **0.0131 ± 0.0001** | **0.954 ± 0.002** | **0.700 ± 0.003** | **0.823 ± 0.009** | n/a |
| rule_oracle | 0.0139 ± 0.0001 | 0.996 ± 0.000 | 0.671 ± 0.004 | 0.796 ± 0.009 | n/a |
| rule_bkt | 0.0124 ± 0.0001 | 0.919 ± 0.003 | 0.608 ± 0.004 | 0.758 ± 0.011 | 0.178 |
| rule_dkt | 0.0124 ± 0.0001 | 0.918 ± 0.004 | 0.640 ± 0.003 | 0.786 ± 0.013 | 0.211 |

### How to read M5.6

1. **Heterogeneity is real in the generative process** (same mastery, different
   best action across types) — but a *global* LinUCB still prefers a population
   average: `no_state` remains competitive / slightly ahead.
2. **State ablation did not open the hoped gap** (unlike the intended story).
   Likely causes: shared linear model across students (not per-student), type
   cues still weak vs mastery/difficulty, multipliers moderate. Next levers:
   stronger \(z_i\), richer history features, or per-student / clustered bandits.
3. **Rule-based still depends on accurate mastery** under hetero; estimated
   state hurts more than for LinUCB.
4. Still **simulated only**; \(z_i\) never leaked to the policy.

---

## M5.7 — Mastery basis features for LinUCB (representation diagnostic)

**Hypothesis:** LinUCB assumes \(E[r|x,a]=\theta_a^\top x\). Raw mastery \(m\) cannot
express mid-peaked `Effect(a,m)` (worked example / Socratic) or thresholded
easier/harder effects, so even oracle mastery may be unusable.

**Change:** simulator unchanged. Only the feature map for LinUCB:

- linear (8-d): \([m,\ldots]\)
- basis (12-d): \([m, m^2, I(m<0.4), I(0.4\le m<0.7), I(m\ge 0.7),\ldots]\)

Same \(\alpha=1.0\), same seeds, homogeneous population (isolate representation).

```bash
python scripts/eval_kt_policies.py --config configs/config.yaml \
  --suite basis --tag basis --n-students 500 --n-seeds 5 --no-heterogeneous
```

| Condition | mastery gain | final mastery | overall acc | final-window acc | \|est−true\| |
|---|---|---|---|---|---|
| linucb_oracle_linear | 0.0132 ± 0.0001 | 0.960 ± 0.001 | 0.700 ± 0.007 | 0.821 ± 0.007 | n/a |
| linucb_no_state | 0.0134 ± 0.0001 | 0.966 ± 0.001 | **0.705 ± 0.007** | 0.827 ± 0.006 | n/a |
| linucb_oracle_basis | **0.0135 ± 0.0001** | **0.972 ± 0.001** | 0.697 ± 0.008 | **0.841 ± 0.007** | n/a |
| linucb_bkt_basis | 0.0132 ± 0.0001 | 0.957 ± 0.001 | 0.691 ± 0.008 | 0.807 ± 0.006 | 0.155 |
| linucb_dkt_basis | 0.0133 ± 0.0001 | 0.962 ± 0.001 | 0.693 ± 0.008 | 0.809 ± 0.006 | 0.201 |

### Verdict (mixed Outcome A)

1. **Oracle + basis beats no_state** on mastery gain, final mastery, and especially
   final-window accuracy (+1.4 pp vs no_state). Student state *was* useful once
   LinUCB could express pedagogical regimes — the linear feature map was the
   bottleneck, not the absence of information.
2. **Overall accuracy still slightly favors no_state** (0.705 vs 0.697). Basis
   helps long-horizon / late accuracy more than mean accuracy over the whole
   trajectory — report both.
3. **BKT/DKT + basis do not retain the oracle advantage.** Estimation error
   (\(\approx 0.15\)–\(0.20\)) still dominates; richer features on a noisy \(\hat m\)
   can hurt. Closing the observability gap remains harder than fixing the
   function class.
4. Interview line: *“Information ≠ usable representation under LinUCB’s linear
   model; with a mastery basis, oracle state helps — estimated state still
   lags.”*

**Bandit section closed.** M5–M5.7 are the contextual-bandit results. Further
gains under one-step \(\Delta m\) are limited; M6 moves to delayed consequences
and full RL.

---

## M6 Step 1 — Delayed Independence / Fatigue (calibration only)

**Goal:** Make myopic \(\max \Delta m\) ≠ long-horizon optimal, without DQN yet.

Latent variables (hidden from every policy):

| Variable | Support actions (explain / hint / …) | Harder problem |
|---|---|---|
| Independence \(I_t\) | decreases (dependence) | increases (esp. if correct) |
| Fatigue \(F_t\) | decreases | increases |

Applied effect: `Effect(a,m) * learning_scale(I,F)`. Config flag
`simulator.delayed_effects` (default **false** so M5 numbers stay reproducible).

```bash
python scripts/calibrate_delayed_effects.py --config configs/config.yaml
```

### delayed_effects = OFF (control)

| Policy | early Δm | late Δm | mean Δm | final m |
|---|---|---|---|---|
| always_explain | 0.0210 | 0.0046 | 0.0117 | 0.882 |
| myopic_oracle | **0.0308** | 0.0000 | **0.0140** | **1.000** |
| interleave_explain_harder | 0.0109 | 0.0105 | 0.0107 | 0.835 |

Myopic saturates — no long-horizon penalty.

### delayed_effects = ON

| Policy | early Δm | late Δm | mean Δm | final m | final I |
|---|---|---|---|---|---|
| always_explain | 0.0087 | 0.0015 | 0.0030 | 0.452 | 0.000 |
| myopic_oracle | **0.0127** | 0.0023 | 0.0045 | 0.522 | 0.000 |
| **interleave_explain_harder** | 0.0071 | **0.0037** | **0.0053** | **0.565** | 0.343 |
| always_harder | -0.0032 | -0.0004 | -0.0020 | 0.201 | 1.000 |

**Calibration verdict:** With delayed effects on, myopic oracle wins *early*
reward but loses on final mastery to a simple explain↔harder interleave that
preserves independence. That is the temporal credit-assignment gap DQN should
exploit in Step 2–3. \(I,F\) stay latent (not in the policy state).

## Reproduce M5 (oracle-state policies)

```bash
python scripts/eval_policies.py --config configs/config.yaml
```

Writes `results/policy_comparison.json` and the two M5 figures.
