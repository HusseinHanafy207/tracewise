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
