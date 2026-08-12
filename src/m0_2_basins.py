"""M0.2 basin visualization: latent-space structure of PTRM parallel rollouts (tool for the E6a latent-space evidence channel)

Tracks z_H[:,0] (the puzzle-emb slot read by q-head) over K noisy rollouts of the same puzzle, quantifying:
  spread(t)   pairwise RMS distance between rollouts (normalized by ||v||) over steps -> single-basin convergence / multi-basin divergence
  q_corr      q-head final score vs rollout correctness: Spearman rho + AUC (selector effectiveness)
  cohens_d    effect size of centroid distance between correct/incorrect rollout endpoints (cluster separability)
  purity      1-NN correctness consistency of endpoints (cluster purity; chance ~ p^2+(1-p)^2)

Outputs (--out dir):
  metrics.json       per-puzzle metrics + aggregates
  basins_traj.npz    latent trajectories for all puzzles (for reuse/replotting)
  basins_pca.png     PCA plot (example puzzle trajectories + all-puzzle centroid-aligned scatter)
  basins_metrics.png spread curves + q/correctness relationship"""
import argparse
import json
import os
import time

import numpy as np
import torch

from eval_ptrm import load_model, load_test_data

matplotlib = None  # lazy import, only needed with --plot


def spearman(x, y):
    rx = torch.argsort(torch.argsort(x.float())).float()
    ry = torch.argsort(torch.argsort(y.float())).float()
    xm, ym = rx.mean(), ry.mean()
    num = ((rx - xm) * (ry - ym)).sum()
    den = ((rx - xm) ** 2).sum() * ((ry - ym) ** 2).sum()
    return (num / den.sqrt()).item() if den > 0 else float("nan")


def auc(q, y):
    """rank of positives after ascending-q (default) order -> Mann-Whitney U -> AUC (later positive = higher score = larger)"""
    order = torch.argsort(q)
    ys = y[order]
    n1 = ys.sum().item()
    n0 = len(ys) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    pos = torch.nonzero(ys).flatten().float()
    U = pos.sum().item() - n1 * (n1 - 1) / 2
    return U / (n0 * n1)


def puzzle_metrics(traj, q, ok):
    """traj [T,K,512], q [K], ok [K] -> metrics dict"""
    T, K, _ = traj.shape
    spread = []
    for t in range(T):
        vt = traj[t].double()
        d2 = torch.cdist(vt, vt) ** 2
        rms_pair = torch.sqrt(d2.sum() / (K * (K - 1)))
        rms_norm = torch.sqrt((vt ** 2).mean())
        spread.append((rms_pair / rms_norm).item())
    vf = traj[-1].double()
    good, bad = vf[ok], vf[~ok]
    cohens_d = float("nan")
    if len(good) >= 2 and len(bad) >= 2:
        cg, cb = good.mean(0), bad.mean(0)
        sp = (((good - cg) ** 2).mean() + ((bad - cb) ** 2).mean()).sqrt()
        cohens_d = ((cg - cb).norm() / sp).item()
    purity = float("nan")
    if ok.sum().item() > 0 and (~ok).sum().item() > 0:
        D = torch.cdist(vf, vf)
        D.fill_diagonal_(float("inf"))
        nn = D.min(1).indices
        purity = (ok[nn] == ok).float().mean().item()
    # intra-cluster tightness: RMS spread of correct/incorrect endpoints around their own centroid (normalized, same convention as spread)
    in_class = {"good": float("nan"), "bad": float("nan")}
    if len(good) >= 2:
        in_class["good"] = (torch.sqrt(((good - good.mean(0)) ** 2).mean())
                            / torch.sqrt((vf ** 2).mean())).item()
    if len(bad) >= 2:
        in_class["bad"] = (torch.sqrt(((bad - bad.mean(0)) ** 2).mean())
                           / torch.sqrt((vf ** 2).mean())).item()
    return {
        "spread": spread,
        "spread_ratio": spread[-1] / spread[0],
        "q_rho": spearman(q, ok),
        "q_auc": auc(q, ok),
        "cohens_d": cohens_d,
        "purity": purity,
        "in_class_spread": in_class,
        "n_good": ok.sum().item(),
        "n_total": K,
    }


def collect(model, inputs, labels, ids, K, D, sigma, save_steps):
    """K-rollout full latent trajectories: returns per-puzzle arrays"""
    N = inputs.shape[0]
    PPC = max(1, 512 // K)
    n_steps = len(save_steps)
    step_idx = {s: i for i, s in enumerate(save_steps)}
    with torch.inference_mode():
        for start in range(0, N, PPC):
            b = inputs[start:start + PPC]
            bs = b.shape[0]
            batch = {
                "inputs": b.repeat(K, 1),
                "puzzle_identifiers": ids[start:start + bs].repeat(K),
            }
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            traj = torch.zeros(n_steps, bs, K, 512, device="cuda")
            for step in range(D):
                carry.inner_carry.z_L = carry.inner_carry.z_L + \
                    sigma * torch.randn_like(carry.inner_carry.z_L)
                carry, outputs = model(carry, batch)
                if step in step_idx:
                    traj[step_idx[step]] = carry.inner_carry.z_H[:, 0].view(bs, K, -1)
            q = outputs["q_halt_logits"].view(K, bs).T
            preds = outputs["logits"].argmax(-1).view(K, bs, 81).transpose(0, 1)
            ok = (preds == labels[start:start + bs].unsqueeze(1)).all(-1)
            for pi in range(bs):
                yield start + pi, traj[:, pi].cpu().numpy(), q[pi].cpu().numpy(), ok[pi].cpu().numpy()


def pca_fit(x, n=2):
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:n].T


def make_plots(out_dir, metrics, traj_data, save_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    first_key = next(iter(traj_data))
    _, K, _ = traj_data[first_key].shape
    T = len(save_steps)

    # ---- figure 1: example-puzzle trajectories (per-puzzle PCA) + all-puzzle centroid-aligned scatter ----
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    examples = [m for m in metrics["per_puzzle"]
                if m["n_good"] not in (0, m["n_total"])][:9]
    for ax, m in zip(axes.flat, examples):
        t = traj_data[m["i"]]           # [T,K,512]
        colors = cm.viridis(np.linspace(0, 1, T))
        flat = pca_fit(t.reshape(-1, 512))
        for s in range(T):
            ax.scatter(flat[s * K:(s + 1) * K, 0], flat[s * K:(s + 1) * K, 1],
                       s=3, color=colors[s], alpha=0.6)
        ok = np.asarray(m["ok"], dtype=bool)
        vf = flat[(T - 1) * K:]
        ax.scatter(vf[ok, 0], vf[ok, 1], marker="o", s=60, facecolor="none",
                   edgecolor="lime", linewidths=1.5, label="correct")
        ax.scatter(vf[~ok, 0], vf[~ok, 1], marker="X", s=55, color="red",
                   label="wrong")
        ax.set_title(f"puzzle {m['i']}  d={m['cohens_d']:.1f} purity={m['purity']:.2f}")
        ax.axis("off")
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, save_steps[-1]))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.03)
    cbar.set_label("step")
    axes.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("M0.2 attractor basins: per-puzzle z_H[:,0] trajectories (PCA)", fontsize=13)
    fig.savefig(os.path.join(out_dir, "basins_pca.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: aggregate metrics ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    sp = np.array([m["spread"] for m in metrics["per_puzzle"]])
    mu, sd = sp.mean(0), sp.std(0)
    ax = axes[0]
    ax.errorbar(save_steps, mu, yerr=sd, capsize=3)
    ax.set_xlabel("step"); ax.set_ylabel("rollout pair RMS dist (norm)")
    ax.set_title(f"spread(t): converge {mu[0]:.3f} → {mu[-1]:.3f} "
                 f"(ratio {metrics['aggregate']['spread_ratio_mean']:.2f})")
    ax = axes[1]
    pairs = [(m["q_rho"], m["q_auc"]) for m in metrics["per_puzzle"]
             if not np.isnan(m["q_rho"]) and not np.isnan(m["q_auc"])]
    rhos, aucs = (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
    ax.scatter(rhos, aucs, alpha=0.6)
    ax.set_xlabel("Spearman ρ (q vs correct)"); ax.set_ylabel("AUC")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.axvline(0, ls="--", color="gray", lw=0.8)
    ax.set_title(f"q-head validity: mean ρ={np.mean(rhos):+.3f}, "
                 f"mean AUC={np.mean(aucs):.3f} (n={len(aucs)})")
    ax = axes[2]
    dpairs = [(m["cohens_d"], m["purity"]) for m in metrics["per_puzzle"]
              if not np.isnan(m["cohens_d"]) and not np.isnan(m["purity"])]
    ds, ps = (np.array([p[0] for p in dpairs]), np.array([p[1] for p in dpairs]))
    ax.scatter(ds, ps, alpha=0.6)
    ax.set_xlabel("Cohen's d (good vs bad centroid)")
    ax.set_ylabel("1-NN purity")
    ax.set_title(f"separability: mean d={np.mean(ds):.2f}, "
                 f"mean purity={np.mean(ps):.3f}")
    fig.suptitle("M0.2 aggregate basin metrics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out_dir, "basins_metrics.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="../data/sudoku-extreme-1k-aug-100")
    ap.add_argument("--arch-yaml", default="config/arch/trm.yaml")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--D", type=int, default=48)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", default="0,4,8,16,24,32,40,47",
                    help="steps to save latents")
    ap.add_argument("--out", default="outputs/2026-08-10/overnight/basins")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    save_steps = [int(s) for s in args.steps.split(",")]
    meta = json.load(open(os.path.join(args.data, "test", "dataset.json")))
    inputs, labels, ids, idx = load_test_data(args.data, args.n, args.seed)
    model = load_model(args.ckpt, args.arch_yaml, meta["vocab_size"], meta["seq_len"],
                       meta["num_puzzle_identifiers"])
    model.config.halt_max_steps = args.D

    npz_path = os.path.join(args.out, "basins_traj.npz")
    json_path = os.path.join(args.out, "metrics.json")
    if os.path.exists(npz_path) and os.path.exists(json_path):
        # already collected -> replot only (avoids re-running ~4min inference when changing plot params)
        print("found existing data, replot only", flush=True)
        loaded = np.load(npz_path)
        traj_data = {int(k[1:]): v for k, v in loaded.items()}
        results = json.load(open(json_path, encoding="utf-8"))
    else:
        t0 = time.time()
        per_puzzle, traj_data = [], {}
        for pi, traj, q, ok in collect(model, inputs, labels, ids, args.K,
                                       args.D, args.sigma, save_steps):
            m = puzzle_metrics(torch.tensor(traj), torch.tensor(q), torch.tensor(ok))
            m["i"] = int(pi)
            m["ok"] = ok.tolist()
            m["q"] = [round(float(v), 4) for v in q]
            per_puzzle.append(m)
            traj_data[int(pi)] = traj
        print(f"collect done: {len(per_puzzle)} puzzles, {time.time() - t0:.0f}s", flush=True)

        ag = {k: float(np.nanmean([m[k] for m in per_puzzle]))
              for k in ["q_rho", "q_auc", "cohens_d", "purity"]}
        ag["spread_ratio_mean"] = float(np.nanmean([m["spread_ratio"] for m in per_puzzle]))
        results = {
            "ckpt": args.ckpt, "n_puzzles": len(per_puzzle), "K": args.K,
            "D": args.D, "sigma": args.sigma, "steps": save_steps,
            "aggregate": ag, "per_puzzle": per_puzzle,
            "note": "spread_ratio<1=converged (single basin); q_auc>0.5=q-head selects correctly; purity>chance(p^2+(1-p)^2)=endpoints separable",
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        np.savez(npz_path, **{f"p{pi}": t for pi, t in traj_data.items()})

    make_plots(args.out, results, traj_data, save_steps)
    print(f"saved -> {args.out}/metrics.json, basins_pca.png, basins_metrics.png")


if __name__ == "__main__":
    main()
