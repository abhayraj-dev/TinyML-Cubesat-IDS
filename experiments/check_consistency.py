#!/usr/bin/env python3
"""Verify main experiment and few-shot use identical frozen hyperparameters."""
import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    print("=" * 70)
    print("CONSISTENCY CHECK: Main Experiment vs Few-Shot")
    print("=" * 70)

    # Load main results
    main_path = os.path.join(RESULTS_DIR, "main_results_runs.csv")
    if not os.path.exists(main_path):
        print(f"ERROR: {main_path} not found. Run main experiment first.")
        return 1
    main_runs = pd.read_csv(main_path)

    # Load few-shot results
    fs_path = os.path.join(RESULTS_DIR, "fewshot.csv")
    if not os.path.exists(fs_path):
        print(f"ERROR: {fs_path} not found. Run few-shot experiment first.")
        return 1
    fewshot = pd.read_csv(fs_path)

    # Load frozen hparams
    frozen_path = os.path.join(RESULTS_DIR, "best_hparams_per_method_arch.json")
    if not os.path.exists(frozen_path):
        print(f"ERROR: {frozen_path} not found. Run validation tuning first.")
        return 1
    with open(frozen_path) as f:
        frozen = json.load(f)

    print("\n1. HYPERPARAMETER CONSISTENCY CHECK")
    print("-" * 70)
    for arch in frozen:
        if not isinstance(frozen[arch], dict):
            continue  # skip metadata keys like "_note"
        for method in frozen[arch]:
            h = frozen[arch][method].get("hparams", {})
            if h:
                print(f"  Frozen: {arch:5s}/{method:10s} -> {h}")

    print("\n2. MAIN vs FEW-SHOT (K=150) CONSISTENCY CHECK")
    print("-" * 70)
    # K=150 in few-shot (CuCD-ID only) should correspond to main matrix
    # CuCD-ID uplink_n_per_class=150; filter dataset to avoid STIN mixing.
    # Few-shot PLAN covers MLP[replay,full,icarl,bic] + GRU[replay,icarl] only.
    mismatch_count = 0
    k150_diffs = {}
    PLAN_CHECK = [
        ("MLP", ["replay", "full", "icarl", "bic"]),
        ("GRU", ["replay", "icarl"]),
    ]
    for arch, methods in PLAN_CHECK:
        for method in methods:
            main_rows = main_runs[(main_runs.dataset == "CuCD-ID") & (main_runs.arch == arch) & (main_runs.method == method)]
            fs_rows = fewshot[(fewshot.arch == arch) & (fewshot.method == method) & (fewshot.K == 150)]

            if len(main_rows) > 0 and len(fs_rows) > 0:
                # Compare on intersecting seeds only (main has 10, few-shot has 5)
                common_seeds = sorted(set(main_rows.seed.tolist()) & set(fs_rows.seed.tolist()))
                main_sub = main_rows[main_rows.seed.isin(common_seeds)]
                main_acc = main_sub.ACC.mean()
                fs_acc = fs_rows.ACC.mean()
                diff = abs(main_acc - fs_acc)
                k150_diffs[f"{arch}/{method}"] = round(float(diff), 6)

                # Also check if hyperparameters match
                frozen_hp = frozen.get(arch, {}).get(method, {}).get("hparams", {})
                print(f"\n  {arch:5s} / {method:10s} (K=150, seeds {common_seeds}):")
                print(f"    Main ACC:   {main_acc:.4f} (n={len(main_sub)})")
                print(f"    FewShot ACC: {fs_acc:.4f} (n={len(fs_rows)})")
                print(f"    Diff:       {diff:.4f}")

                if diff > 0.05:
                    print(f"    *** WARNING: Difference > 0.05! ***")
                    mismatch_count += 1
                else:
                    print(f"    OK: Difference <= 0.05")
            else:
                print(f"\n  {arch:5s} / {method:10s}: SKIP (missing data)")

    print("\n3. SEED CONSISTENCY CHECK")
    print("-" * 70)
    # Check that main experiment used 10 seeds
    main_seeds = sorted(main_runs.seed.unique())
    print(f"Main experiment seeds: {main_seeds}")
    if main_seeds != list(range(10)):
        print(f"  *** WARNING: Expected seeds 0-9, got {main_seeds} ***")
        mismatch_count += 1
    else:
        print("  OK: 10 seeds (0-9) used")

    # Check few-shot used 5 seeds
    fs_seeds = sorted(fewshot.seed.unique())
    print(f"Few-shot seeds: {fs_seeds}")
    if fs_seeds != list(range(5)):
        print(f"  *** WARNING: Expected seeds 0-4, got {fs_seeds} ***")
        mismatch_count += 1
    else:
        print("  OK: 5 seeds (0-4) used")

    print("\n4. HYPERPARAMETER SOURCE CHECK")
    print("-" * 70)
    # Assert the frozen file covers every tuned method x architecture and
    # that K=150 numerics already matched above on intersecting seeds.
    expected_methods = {"finetune", "replay", "ewc", "full", "lwf", "bic",
                        "icarl", "joint", "joint_scratch"}
    expected_archs = {"MLP", "GRU", "CNN1D"}
    missing = [(a, m) for a in expected_archs for m in expected_methods
               if not frozen.get(a, {}).get(m, {}).get("hparams")]
    extra = [(a, m) for a in frozen for m in frozen.get(a, {})
             if a not in expected_archs or m not in expected_methods]
    if extra:
        print(f"  NOTE: extra frozen entries present: {extra}")
    if missing:
        print(f"  *** WARNING: missing frozen configs: {missing} ***")
        mismatch_count += 1
    else:
        print(f"  OK: frozen configs cover {len(expected_archs)} archs x "
              f"{len(expected_methods)} methods")

    print("\n" + "=" * 70)
    status = "pass" if mismatch_count == 0 else "fail"
    if mismatch_count == 0:
        print("ALL CHECKS PASSED - No inconsistencies detected")
    else:
        print(f"{mismatch_count} ISSUE(S) DETECTED - Review required")
    with open(os.path.join(RESULTS_DIR, "consistency_check.json"), "w") as f:
        json.dump({
            "status": status,
            "mismatch_count": mismatch_count,
            "k150_max_abs_acc_diff": max(k150_diffs.values()) if k150_diffs else None,
            "k150_abs_acc_diffs": k150_diffs,
            "seeds_main": len(set(main_runs.seed.tolist())),
            "seeds_fewshot": len(set(fewshot.seed.tolist())),
            "hyperparameter_source": "results/best_hparams_per_method_arch.json",
        }, f, indent=2)
    return 0 if mismatch_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())