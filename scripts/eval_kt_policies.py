"""M5.5 — connect KT student models to the adaptive policy.

Compares LinUCB (and rule-based) when the context uses:
  - oracle:          true simulator mastery (upper bound; not available in reality)
  - bkt_estimated:   online BKT P(correct) from observed responses
  - dkt_estimated:   online DKT P(correct|skill) from observed responses
  - no_state:        mastery features zeroed (open-loop / difficulty-only)

Requires ASSISTments-fitted checkpoints:
  results/checkpoints/bkt.pkl
  results/checkpoints/dkt_best.pt

IMPORTANT: simulated outcomes only. Skill ids come from the ASSISTments
vocabulary; the generative process is the abstract simulator — this tests
whether estimated student state helps intervention selection under partial
observability, not real classroom gains.

Usage:
    python scripts/eval_kt_policies.py --config configs/config.yaml
    python scripts/eval_kt_policies.py --config configs/config.yaml \\
        --n-students 2000 --n-seeds 10 --suite linucb_rule \\
        --tag robustness
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dkt import DKT
from src.policy.bandit import LinUCB
from src.policy.kt_state import (
    CONTEXT_DIM_BASIS,
    CONTEXT_DIM_LINEAR,
    BKTOnlineTracker,
    DKTOnlineTracker,
)
from src.policy.rule_based import RuleBasedPolicy
from src.policy.simulator import run_kt_aware_simulation
from src.utils.seed import load_config, set_seed

# Named condition suites for cheap robustness / ablation runs.
SUITES = {
    "full": [
        "linucb_oracle",
        "linucb_bkt",
        "linucb_dkt",
        "linucb_no_state",
        "rule_oracle",
        "rule_bkt",
        "rule_dkt",
    ],
    "linucb": [
        "linucb_oracle",
        "linucb_bkt",
        "linucb_dkt",
        "linucb_no_state",
    ],
    "linucb_rule": [
        "linucb_oracle",
        "linucb_bkt",
        "linucb_dkt",
        "linucb_no_state",
        "rule_oracle",
        "rule_bkt",
        "rule_dkt",
    ],
    # Feature-representation diagnostic (simulator unchanged; only x for LinUCB).
    "basis": [
        "linucb_no_state",
        "linucb_oracle_linear",
        "linucb_oracle_basis",
        "linucb_bkt_basis",
        "linucb_dkt_basis",
    ],
}


def _mean_std(rows: List[dict], key: str):
    vals = np.array([r[key] for r in rows if key in r], dtype=np.float64)
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def load_bkt(ckpt_dir: Path):
    path = ckpt_dir / "bkt.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/fit_bkt.py first (or copy from Kaggle)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dkt(ckpt_dir: Path, device: torch.device):
    path = ckpt_dir / "dkt_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/train_dkt.py on Kaggle/local and "
            "place dkt_best.pt under results/checkpoints/."
        )
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    dkt_cfg = blob.get("dkt_cfg") or {}
    model = DKT(
        num_skills=blob["num_skills"],
        embedding_dim=dkt_cfg.get("embedding_dim", 128),
        hidden_dim=dkt_cfg.get("hidden_dim", 128),
        num_layers=dkt_cfg.get("num_layers", 1),
        dropout=0.0,
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob["num_skills"]


def run_condition(
    name: str,
    policy_factory: Callable,
    tracker_factory: Optional[Callable],
    state_mode: str,
    for_rule: bool,
    n_students: int,
    n_interactions: int,
    num_skills: int,
    seeds: List[int],
    heterogeneous: bool = False,
    feature_mode: str = "linear",
) -> dict:
    rows = []
    for seed in seeds:
        policy = policy_factory(seed)
        result = run_kt_aware_simulation(
            policy,
            tracker_factory=tracker_factory,
            n_students=n_students,
            n_interactions=n_interactions,
            num_skills=num_skills,
            seed=seed,
            state_mode=state_mode,
            for_rule=for_rule,
            heterogeneous=heterogeneous,
            feature_mode=feature_mode,
        )
        rows.append(result)

    summary = {
        "name": name,
        "state_mode": state_mode,
        "feature_mode": feature_mode,
        "n_seeds": len(seeds),
        "heterogeneous": heterogeneous,
    }
    for key in (
        "mean_reward",
        "mean_final_mastery",
        "overall_accuracy",
        "final_accuracy",
        "mean_abs_estimation_error",
    ):
        mu, sd = _mean_std(rows, key)
        summary[key] = mu
        summary[f"{key}_std"] = sd

    curves = np.stack([np.asarray(r["accuracy_curve"], dtype=np.float64) for r in rows])
    summary["accuracy_curve_mean"] = curves.mean(axis=0).tolist()
    return summary


def print_table(summaries: List[dict]) -> str:
    header = (
        "| Condition | mastery gain | final mastery | overall acc | "
        "final-window acc | |est-true| |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = ""
    for s in summaries:
        err = s.get("mean_abs_estimation_error", float("nan"))
        err_s = f"{err:.3f}" if err == err else "n/a"  # NaN check
        rows += (
            f"| {s['name']} | "
            f"{s['mean_reward']:.4f} ± {s['mean_reward_std']:.4f} | "
            f"{s['mean_final_mastery']:.3f} ± {s['mean_final_mastery_std']:.3f} | "
            f"{s['overall_accuracy']:.3f} ± {s['overall_accuracy_std']:.3f} | "
            f"{s['final_accuracy']:.3f} ± {s['final_accuracy_std']:.3f} | "
            f"{err_s} |\n"
        )
    table = header + rows
    print(table)
    return table


def plot_curves(summaries: List[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in summaries:
        ax.plot(s["accuracy_curve_mean"], label=s["name"], linewidth=2)
    ax.set_xlabel("Interaction t")
    ax.set_ylabel("Mean accuracy across students")
    ax.set_title("KT-aware policies (simulated, partial observability)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(
    config_path: str,
    n_students: Optional[int] = None,
    n_interactions: Optional[int] = None,
    n_seeds: Optional[int] = None,
    suite: str = "full",
    tag: Optional[str] = None,
    heterogeneous: Optional[bool] = None,
):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bkt = load_bkt(ckpt_dir)
    dkt, num_skills = load_dkt(ckpt_dir, device)

    # Confirm skill count matches processed data if present
    skill_path = Path(cfg["data"]["processed_dir"]) / "skill_to_idx.pkl"
    if skill_path.exists():
        with open(skill_path, "rb") as f:
            n_data = len(pickle.load(f))
        if n_data != num_skills:
            raise ValueError(f"DKT num_skills={num_skills} != data skills={n_data}")

    actions = list(cfg["policy"]["actions"])
    alpha = float(cfg["policy"]["bandit"].get("alpha", 1.0))
    low = float(cfg["policy"]["rule_thresholds"]["low_mastery"])
    high = float(cfg["policy"]["rule_thresholds"]["high_mastery"])
    # Prefer explicit constants so basis dim stays consistent with feature map.
    linear_dim = CONTEXT_DIM_LINEAR
    basis_dim = CONTEXT_DIM_BASIS
    cfg_dim = int(cfg["policy"]["bandit"]["context_dim"])
    if cfg_dim != linear_dim:
        print(
            f"WARNING: config context_dim={cfg_dim} != linear {linear_dim}; "
            f"using {linear_dim} for linear conditions."
        )

    n_students = int(n_students if n_students is not None else cfg["simulator"]["n_students"])
    n_interactions = int(
        n_interactions
        if n_interactions is not None
        else cfg["simulator"]["n_interactions_per_student"]
    )
    n_seeds = int(n_seeds if n_seeds is not None else cfg["simulator"].get("n_seeds", 5))
    seeds = [cfg["seed"] + i for i in range(n_seeds)]
    wanted = set(SUITES[suite])
    if heterogeneous is None:
        heterogeneous = bool(cfg["simulator"].get("heterogeneous", False))

    print(
        f"KT-aware policy eval | students={n_students} | T={n_interactions} | "
        f"skills={num_skills} | seeds={n_seeds} | suite={suite} | "
        f"hetero={heterogeneous} | device={device}"
    )
    print(
        "DISCLAIMER: simulated only. Policy sees estimated state "
        "(except oracle), never true mastery or latent z_i."
    )
    print(f"  feature dims: linear={linear_dim}, basis={basis_dim}, alpha={alpha}")

    def make_linucb(dim: int):
        def factory(seed):
            return LinUCB(actions, dim, alpha=alpha)

        return factory

    def rule_factory(seed):
        return RuleBasedPolicy(low_thresh=low, high_thresh=high)

    def bkt_tracker_factory():
        return BKTOnlineTracker(bkt)

    def dkt_tracker_factory():
        return DKTOnlineTracker(dkt, device)

    # (name, factory, tracker_factory, state_mode, for_rule, feature_mode)
    condition_specs = [
        ("linucb_oracle", make_linucb(linear_dim), None, "oracle", False, "linear"),
        ("linucb_oracle_linear", make_linucb(linear_dim), None, "oracle", False, "linear"),
        ("linucb_bkt", make_linucb(linear_dim), bkt_tracker_factory, "estimated", False, "linear"),
        ("linucb_dkt", make_linucb(linear_dim), dkt_tracker_factory, "estimated", False, "linear"),
        ("linucb_no_state", make_linucb(linear_dim), None, "no_state", False, "linear"),
        ("linucb_oracle_basis", make_linucb(basis_dim), None, "oracle", False, "basis"),
        ("linucb_bkt_basis", make_linucb(basis_dim), bkt_tracker_factory, "estimated", False, "basis"),
        ("linucb_dkt_basis", make_linucb(basis_dim), dkt_tracker_factory, "estimated", False, "basis"),
        ("rule_oracle", rule_factory, None, "oracle", True, "linear"),
        ("rule_bkt", rule_factory, bkt_tracker_factory, "estimated", True, "linear"),
        ("rule_dkt", rule_factory, dkt_tracker_factory, "estimated", True, "linear"),
    ]

    summaries = []
    for name, factory, tracker_factory, state_mode, for_rule, feature_mode in condition_specs:
        if name not in wanted:
            continue
        print(f"  running {name} (features={feature_mode}) ...", flush=True)
        summaries.append(
            run_condition(
                name,
                factory,
                tracker_factory=tracker_factory,
                state_mode=state_mode,
                for_rule=for_rule,
                n_students=n_students,
                n_interactions=n_interactions,
                num_skills=num_skills,
                seeds=seeds,
                heterogeneous=heterogeneous,
                feature_mode=feature_mode,
            )
        )

    table = print_table(summaries)
    stem = f"kt_policy_{tag}" if tag else "kt_policy_comparison"
    fig_path = fig_dir / f"{stem}_learning_curves.png"
    plot_curves(summaries, fig_path)
    print(f"Wrote {fig_path}")

    payload = {
        "disclaimer": (
            "Simulated KT-aware policy comparison. Estimated-state conditions "
            "never expose true mastery or latent learner type z_i to the policy. "
            "Not real student learning."
        ),
        "tag": tag,
        "suite": suite,
        "heterogeneous": heterogeneous,
        "context_dim_linear": linear_dim,
        "context_dim_basis": basis_dim,
        "alpha": alpha,
        "n_students": n_students,
        "n_interactions": n_interactions,
        "n_seeds": n_seeds,
        "num_skills": num_skills,
        "device": str(device),
        "conditions": summaries,
        "markdown_table": table,
    }
    out = results_dir / f"{stem}.json"

    def _jsonable(obj):
        if isinstance(obj, float) and obj != obj:  # NaN
            return None
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonable(v) for v in obj]
        return obj

    with open(out, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--n-students", type=int, default=None)
    parser.add_argument("--n-interactions", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument(
        "--suite",
        type=str,
        default="full",
        choices=sorted(SUITES.keys()),
        help="Which conditions to run",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Output stem suffix, e.g. robustness → kt_policy_robustness.json",
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        help="Enable latent learner types z_i (hidden from policy)",
    )
    parser.add_argument(
        "--no-heterogeneous",
        action="store_true",
        help="Force homogeneous Effect(a,m) even if config sets heterogeneous",
    )
    args = parser.parse_args()
    hetero = None
    if args.heterogeneous:
        hetero = True
    elif args.no_heterogeneous:
        hetero = False
    main(
        args.config,
        n_students=args.n_students,
        n_interactions=args.n_interactions,
        n_seeds=args.n_seeds,
        suite=args.suite,
        tag=args.tag,
        heterogeneous=hetero,
    )
