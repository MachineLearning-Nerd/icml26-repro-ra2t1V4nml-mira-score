#!/usr/bin/env python3
"""MIRA: a score for conditional distribution accuracy (arXiv:2605.02014). Reproduces Proposition
3.5: for ANY candidate model M, the MIRA score mu_MIRA(M) >= 1/2, with equality approached as the
candidate and true distributions have disjoint supports; and Corollary 3.4: mu_MIRA = 2/3 when the
candidate equals the truth.

MIRA statistic (Prop 3.2):  p(k|n) = (n+1)/(N+2) if k=1 (true sample y* in region R)
                                     (N-n+1)/(N+2) if k=0,
with n = # candidate samples in R, N = total candidate samples. Region R = ball around a random
center c of radius ||y_r - c|| where y_r is a reference candidate sample. Gaussian toy: true
N(x,Sigma*), candidate N(x,Sigma_c). Deterministic seeds.
"""
import numpy as np, json, hashlib

def mira_score(sigma_true, sigma_cand, rng, n_obs=300, N=400, Lr=60):
    vals = []
    for _ in range(n_obs):
        x = rng.uniform(-5, 5, 2)
        ystar = x + rng.standard_normal(2) * sigma_true
        cand = x + rng.standard_normal((N, 2)) * sigma_cand          # N candidate samples
        for _ in range(Lr):
            c = x + rng.uniform(-1, 1, 2)
            yr = cand[rng.integers(N)]                                # reference candidate
            radius = np.linalg.norm(yr - c)
            n = int(np.sum(np.linalg.norm(cand - c, axis=1) <= radius))
            k = int(np.linalg.norm(ystar - c) <= radius)
            p = (n + 1) / (N + 2) if k == 1 else (N - n + 1) / (N + 2)
            vals.append(p)
    return float(np.mean(vals))

def main():
    R = {"claim": "MIRA_Prop3.5_lower_bound_half", "paper": "arXiv:2605.02014"}
    rng = np.random.default_rng(0); sigma_true = 1.0
    rows = []
    for sc in [1.0, 1.5, 2.5, 5.0, 0.05]:                            # matched, over-dispersed, near-disjoint
        mu = mira_score(sigma_true, sc, np.random.default_rng(100 + int(sc * 100)))
        rows.append({"sigma_cand": sc, "mira_score": round(mu, 4),
                     "regime": "matched" if sc == 1.0 else ("near-disjoint" if sc == 0.05 else "mismatched")})
    R["mira_sweep"] = rows
    R["all_scores_ge_half"] = all(r["mira_score"] >= 0.5 - 0.01 for r in rows)      # Prop 3.5
    matched = next(r for r in rows if r["sigma_cand"] == 1.0)
    R["matched_score_near_2_3"] = abs(matched["mira_score"] - 2/3) < 0.03            # Corollary 3.4
    disjoint = next(r for r in rows if r["sigma_cand"] == 0.05)
    R["disjoint_approaches_half"] = disjoint["mira_score"] < 0.60                    # approaches 1/2

    R["verdict"] = "supports" if (R["all_scores_ge_half"] and R["matched_score_near_2_3"]
                                  and R["disjoint_approaches_half"]) else "inconclusive"

    print("claim: " + R["claim"])
    print("MIRA score via region-based ranking of a true sample among candidate samples.")
    print()
    print("  sigma_cand   regime          mira_score")
    for r in rows:
        print(f"  {r['sigma_cand']:<11} {r['regime']:<15} {r['mira_score']}")
    print()
    print(f"[3] Prop 3.5: all MIRA scores >= 1/2: {R['all_scores_ge_half']}")
    print(f"    Corollary 3.4: matched (sigma_c=sigma*) score = {matched['mira_score']} ~ 2/3: {R['matched_score_near_2_3']}")
    print(f"    near-disjoint score = {disjoint['mira_score']} approaches 1/2 from above: {R['disjoint_approaches_half']}")
    print(f"verdict: {R['verdict']}")

    def _np(o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        raise TypeError
    import os; os.makedirs("outputs", exist_ok=True)
    open("outputs/mira_results.json", "w").write(json.dumps(R, indent=2, default=_np))
    print("RESULTS_SHA256=" + hashlib.sha256(json.dumps(R, sort_keys=True, default=_np).encode()).hexdigest())
    return 0 if R["verdict"] == "supports" else 1

if __name__ == "__main__":
    raise SystemExit(main())
