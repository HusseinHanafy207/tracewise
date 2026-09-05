# Artifact and experiment provenance

This file identifies the data, checkpoints, and result files behind the
numbers reported in `docs/`. Generated artifacts remain gitignored because
the raw dataset and model outputs are too large or externally licensed. The
SHA-256 fingerprints below make local copies verifiable.

## Milestone snapshot

- First committed snapshot containing the complete KT + contextual-bandit
  implementation: `b328e4719735431b585e5b77d94e99600363224a`.
- Milestone tag: `v0.2-bandit`.
- Canonical configuration at that snapshot: `configs/config.yaml`, Git blob
  `c594305238dbdb9766d559a1b3786626335d8d4e`.

The policy experiments were originally run before their implementation was
committed. `b328e47` is therefore the first durable implementation snapshot,
not a claim that the historical JSON files were produced from a clean checkout
of that commit. Treat those JSON files as legacy evidence until they are rerun
from a tagged environment. This limitation is recorded instead of assigning a
fabricated historical commit.

## DKT run reconciliation

Two DKT runs appear in the repository and answer different provenance needs:

- **Kaggle evaluation run:** best validation ROC-AUC `0.838554` at epoch 11.
  It produced `results/kt_comparison.json`, `results/kt_slices.json`, and the
  headline test ROC-AUC `0.853079`. The evaluation code is captured by commit
  `ec7db0fc130e9e530ebb6642c42880038d387c8d`; its config Git blob is
  `c4e9204b69eebda5ec14fe6f6333a7672a0ed2cd`. That checkpoint was later
  replaced locally and is no longer present as a separate binary.
- **Local CPU run / current checkpoint:** validation ROC-AUC `0.838277` at
  epoch 14, recorded by `results/dkt_val.json`. Its checkpoint is the current
  `results/checkpoints/dkt_best.pt`. The DKT model, dataset, training, and KT
  evaluation code did not change between `ec7db0f` and `b328e47`; only policy
  and simulator configuration fields were added. This checkpoint is the one
  referenced by the later KT-aware policy experiments.

The README uses the Kaggle run for the fair BKT-versus-DKT headline. The policy
documentation labels its DKT checkpoint as the local `~0.8383` run. These
numbers should not be merged into a single training history.

## Confirmed code lineage

- Preprocessing output was generated on 2026-08-26. The nearest committed
  preprocessing snapshot is `362989216a2b59cde1ba33ca0999663f849f4ad0`.
- The BKT implementation and current BKT checkpoint are captured by
  `5a65222d6b8cf0186ab3639b157eef8645497e4a`.
- DKT comparison and slice-analysis code are captured by
  `ec7db0fc130e9e530ebb6642c42880038d387c8d`.
- Contextual-bandit, KT-aware policy, heterogeneous learner, mastery-basis,
  and delayed-effect calibration code first entered Git in
  `b328e4719735431b585e5b77d94e99600363224a`.

## Data fingerprints

- `data/raw/skill_builder_data.csv`
  - bytes: `83201656`
  - SHA-256: `c602374cfb673d83ee98f47a324f70973c54b71e3e79954946266c2bdbcc30f8`
- `data/processed/meta.json`
  - SHA-256: `26ac12c87e7791caca847379881de8b9a28bd85d09d615686c74701c7c01d3e3`
- `data/processed/train.pkl`
  - SHA-256: `1c4e69f6a30f95e348b35cd8ae90792ed3eb06807cb072b583570759ea2e6dd4`
- `data/processed/val.pkl`
  - SHA-256: `f17fd73788a5a3cbe3a982d2ef575f818c261359cad54fc82d7ea8275650d7f6`
- `data/processed/test.pkl`
  - SHA-256: `e618f85a876d7d19427930cc01141ed63b1defcc639fcb849c2b31a2b9d4c188`

## Checkpoint fingerprints

- `results/checkpoints/bkt.pkl`
  - SHA-256: `f8d3bbce66ee70c71a99c003ba387e81d994f9052429fc07360f2cfc42417b46`
- `results/checkpoints/dkt_best.pt` (local CPU run, epoch 14)
  - SHA-256: `6c3c00e2afeb4f3942454ac7ff5943e1fc45d860f9b92b0be530ee1c426930bb`
- `results/checkpoints/dqn_oracle_best.pt` (canonical seed-42 oracle DQN)
  - SHA-256: `8ff6c47950ee058ad12be0dff29514cd1f044c741d67e99992558901fbf6244c`
- `results/checkpoints/dqn_oracle_best_double_seed42.pt`
  - SHA-256: `6a418f261c16b5f56428357daab33a9e6df8db4f790c93c602a90a89a7e73f89`
- `results/checkpoints/dqn_oracle_best_double_seed43.pt`
  - SHA-256: `dea0fb9ab61b934b08eed00bfa144b741e08594f86699e1787fdce6b3e732d23`
- `results/checkpoints/dqn_oracle_best_double_seed44.pt`
  - SHA-256: `4fb42ad11f158a7b9cdda9b2fa757cc7206ef5602be132e403b02353393ebc69`

## Result fingerprints

- `results/bkt_val.json`: `14224b4e5f907ca6af6f0ea884a21bd033b14c868354465d6a2f83ad06b002d5`
- `results/dkt_val.json`: `d8a1349e7d0f29aabd42ef4a7e7c7f4b15c809433a10796318b731a8e6906792`
- `results/kt_comparison.json`: `e1817f2cfdae476ef6c2d5e9da1d4c3ecd069c67390e8213dfa52d24e1b3e693`
- `results/kt_slices.json`: `12a9500d0c948582b26197a382f25f0b14f7006f0dc6140e1295ede00c999399`
- `results/policy_comparison.json`: `f75fcf153c24fce6d027297f936c573810bde44a7e8daf13cbedc4ea407661bc`
- `results/kt_policy_comparison.json`: `0bad8cb4f9a1d9d1588919a030414fad807c0079de50e1045823059f02e66d14`
- `results/kt_policy_robustness.json`: `744c6b7f4a5ce6589170e57e5a2afedf889eed2b5a6f94957c13d551732fdbe2`
- `results/kt_policy_hetero.json`: `3811fecd33227c46490ed2eea9c4fcb779009be6c85000e8a97227b2b12733d5`
- `results/kt_policy_basis.json`: `8bef1aa454073d418111fd3b684495e17e1876a4aa900ae5c3f366293a375505`
- `results/delayed_effects_calibration.json`: `2d140a3f6f2d8bf3841223d10733543b13c6eb11556fd55c1a27323a6a60bca6`
- `results/dqn_phase4_evaluation.json`: `83462590efb219338506476ab6cc8d7bf92f2c8315c371f6244baad3505b9cb0`
- `results/double_dqn_comparison.json`: `062f17f413f87f9ad297ead4f463598303137781b33602c121559827f36c6069`
- `results/q_overestimation_diagnostic.json` (shared state bank): `058a92f1856bd74368ed13e5d2d07d3039b59271ac644249d4d955386f9861f6`
- `results/q_overestimation_diagnostic_on_policy.json`: `db7fa9da4d4398ff7adb5f896138388dacd2b7de06e99bc79625f58e15cb34b7`
- `results/double_dqn_budget_comparison.json`: `9b481df3e3db33643972a0c3c04f604e1798d06b4fb3e1b5b5ce51ea7af8ec29`
- `results/recurrent_double_dqn_comparison.json`: `ce1aef8675ddddc8650962ce2004f302ae00f49b3c0ea18944faaf587b4cc0e4`
- `results/memory_ablation_comparison.json`: `9643e4bfbec23352dfee0b1bebf0039f0821acc43b31474d0e2a5a9a05490c54`
- `results/multiskill_environment_validation.json`: `da53311f2c4bc17e61931f60559a2e2c4636fee55a713e7c5955e30ba290aae0`

## Portfolio example artifacts

Small artifacts are committed so the policy behavior can be inspected on a
fresh clone without downloading a checkpoint or retraining:

- `docs/examples/dqn_episode_seed200000.json` is a full 50-step trace from the
  canonical oracle-state DQN. Seed `200000` is the first fixed held-out seed,
  not an outcome-selected episode. The trace records the checkpoint and demo
  generator SHA-256 hashes plus Git worktree provenance.
  - SHA-256: `e69fd213924b08dde13f04f0397ca35985e76d6617ca451184dc2071736c7229`
- `docs/media/dqn_episode.gif` is rendered directly from that trace.
  - SHA-256: `badb2cec68718fb9fe4756a9bdeb5abd529c1e2605b567b7a517d88ec6ba2b78`
- `docs/examples/portfolio_results.json` is a lightweight, machine-readable
  copy of the headline metrics already documented in the experiment reports.
  - SHA-256: `fb52154aeea4f0f337b54ce98fca16009bce1653ba60fb8508352211f6d8149c`
- `docs/examples/double_dqn_results.json` preserves the controlled comparison's
  compact ten-seed performance/mechanism summary and raw-result fingerprints.
  - SHA-256: `c9458ca93582d6423ab38e8acb2034fe712508ac9572b5a44baee2939532bd1b`
- `docs/examples/double_dqn_budget_results.json` preserves the compact
  ten-seed 2,000-versus-5,000 episode comparison.
  - SHA-256: `2433445255cc9834b1121934367262323563fe01d33a41f029b71e150da50209`
- `docs/examples/recurrent_double_dqn_results.json` preserves the compact
  ten-seed feed-forward-versus-GRU comparison and replay limitation.
  - SHA-256: `6b5a8264183578268a68ceb926e9889e4b60fc1a8064310dff51047a4a9cfddd`
- `docs/examples/memory_ablation_results.json` preserves the replay-matched
  ten-seed MLP-versus-GRU memory ablation.
  - SHA-256: `d107e476a3dd005000ea973982db08ef43cc6542a2aa832e6f9f119067f28504`
- `docs/examples/multiskill_environment_results.json` preserves the compact
  four-skill prerequisite calibration and validation checks.
  - SHA-256: `3c6a9e0e0b9db5a78d4a4123986f118b3b4f9dc1700e3aaddd5e7caa5ef35fe7`

These files do not contain real student records. The episode is entirely
simulated, and oracle mastery is included for explanatory evaluation only.

Replay or re-render it with:

```powershell
python scripts/demo_policy_episode.py --replay docs/examples/dqn_episode_seed200000.json
python scripts/demo_policy_episode.py --replay docs/examples/dqn_episode_seed200000.json --save-gif docs/media/dqn_episode.gif
```

## Reproducing artifacts

Obtain the ASSISTments CSV using `data/README.md`, then run from the repository
root:

```powershell
python scripts/run_preprocessing.py --config configs/config.yaml
python scripts/fit_bkt.py --config configs/config.yaml
python scripts/train_dkt.py --config configs/config.yaml
python scripts/eval_kt.py --config configs/config.yaml
python scripts/analyze_kt.py --config configs/config.yaml
python scripts/eval_policies.py --config configs/config.yaml
python scripts/eval_kt_policies.py --config configs/config.yaml
python scripts/eval_kt_policies.py --config configs/config.yaml --suite basis --tag basis --no-heterogeneous
python scripts/calibrate_delayed_effects.py --config configs/config.yaml
```

The larger robustness and heterogeneous runs are documented with their exact
CLI arguments in `docs/experiments_policy.md`. Phase 1 will establish a pinned
Python environment and rerun the legacy policy artifacts from a clean tag.
