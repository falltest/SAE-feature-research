from __future__ import annotations

import argparse
import sys

import numpy as np


MIN_ACTIVATIONS = 300      
MAX_SAMPLE = 5000          
DIP_ALPHA = 0.05          
MIN_SEPARATION = 2.0       
MIN_WEIGHT = 0.05          


def reconstruct_sample(counts, log_edges, rng, max_n=MAX_SAMPLE):
    """
    Восстанавливает псевдо-сэмпл в log10-пространстве из гистограммы.

    counts    — (n_bins,) счётчики (уже без overflow-бина)
    log_edges — (n_bins+1,) границы бинов в log10
    """
    total = counts.sum()
    if total == 0:
        return np.array([])

    if total > max_n:
        probs = counts / total
        counts = rng.multinomial(max_n, probs)

    lo = np.repeat(log_edges[:-1], counts)
    hi = np.repeat(log_edges[1:], counts)
    return rng.uniform(lo, hi)


def fit_gmm(sample):
    """GMM 1 vs 2 компоненты. Возвращает dict с результатами или None."""
    from sklearn.mixture import GaussianMixture

    X = sample.reshape(-1, 1)
    try:
        g1 = GaussianMixture(n_components=1, random_state=0, n_init=1).fit(X)
        g2 = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(X)
    except Exception:
        return None

    bic1, bic2 = g1.bic(X), g2.bic(X)
    mus = g2.means_.ravel()
    sigmas = np.sqrt(g2.covariances_.ravel())
    weights = g2.weights_.ravel()

    order = np.argsort(mus)
    mus, sigmas, weights = mus[order], sigmas[order], weights[order]

    sep = abs(mus[1] - mus[0]) / max(sigmas.mean(), 1e-9)

    return {
        "bic1": bic1,
        "bic2": bic2,
        "bic_gain": bic1 - bic2,     
        "mu1_log": mus[0],
        "mu2_log": mus[1],
        "sigma1": sigmas[0],
        "sigma2": sigmas[1],
        "w1": weights[0],
        "w2": weights[1],
        "separation": sep,
    }


def run_dip(sample):
    """dip test Хартигана. Возвращает (dip_stat, p_value) или (nan, nan)."""
    try:
        import diptest
    except ImportError:
        return np.nan, np.nan
    try:
        return diptest.diptest(sample)
    except Exception:
        return np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="путь к pass1_gemma_checkpoint.npz")
    ap.add_argument("--width", default="16k", help="какую SAE анализировать (16k / 65k)")
    ap.add_argument("--top", type=int, default=10, help="сколько кандидатов вывести")
    ap.add_argument("--out", default=None, help="куда сохранить полный CSV с метриками")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.npz)
    key = f"hist_{args.width}"
    if key not in data.files:
        print(f"Нет ключа {key}. Доступно: {data.files}", file=sys.stderr)
        sys.exit(1)

    hist = data[key]
    edges = data["bin_edges"]
    tokens_seen = int(data["tokens_seen"])
    D, n_bins = hist.shape

    print(f"{args.width}: {D} фич, {n_bins} бинов, tokens_seen={tokens_seen:,}")

    overflow_share = hist[:, -1].sum() / max(hist.sum(), 1)
    if overflow_share > 0.01:
        print(f"ВНИМАНИЕ: {overflow_share:.2%} активаций в overflow-бине. "
              f"BIN_MAX в пассе 1 занижен, распределение обрезано сверху. ")
    hist = hist[:, :-1]
    edges = edges[:-1]           

    log_edges = np.empty_like(edges)
    log_edges[0] = np.log10(edges[1]) - (np.log10(edges[2]) - np.log10(edges[1]))
    log_edges[1:] = np.log10(edges[1:])

    counts_total = hist.sum(axis=1)
    eligible = np.where(counts_total >= MIN_ACTIVATIONS)[0]
    print(f"Фич с >={MIN_ACTIVATIONS} срабатываний: {len(eligible)} / {D}")
    if len(eligible) == 0:
        print("Нечего анализировать.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    rows = []

    for n, fidx in enumerate(eligible):
        if n % 2000 == 0 and n:
            print(f"  ...{n}/{len(eligible)}")

        sample = reconstruct_sample(hist[fidx], log_edges, rng)
        if sample.size < MIN_ACTIVATIONS:
            continue

        gmm = fit_gmm(sample)
        if gmm is None:
            continue

        dip_stat, dip_p = run_dip(sample)

        rows.append({
            "feature": int(fidx),
            "n_acts": int(counts_total[fidx]),
            "density": counts_total[fidx] / max(tokens_seen, 1),
            "dip_stat": dip_stat,
            "dip_p": dip_p,
            **gmm,
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    # --- критерий бимодальности ---
    df["bimodal"] = (
        (df["dip_p"] < DIP_ALPHA)
        & (df["separation"] > MIN_SEPARATION)
        & (df[["w1", "w2"]].min(axis=1) > MIN_WEIGHT)
    )

    # --- унимодальные "хвостовые" — контрольная группа ---
    df["unimodal_tail"] = (df["dip_p"] > 0.5) & (df["separation"] < MIN_SEPARATION)

    n_bi = int(df["bimodal"].sum())
    n_uni = int(df["unimodal_tail"].sum())
    print(f"\nБимодальных (оба теста + separation + вес): {n_bi}")
    print(f"Унимодальных хвостовых (контроль):          {n_uni}")

    cols = ["feature", "n_acts", "density", "dip_p", "bic_gain",
            "separation", "mu1_log", "mu2_log", "w1", "w2"]

    print(f"\n=== ТОП-{args.top} БИМОДАЛЬНЫХ (по separation) ===")
    bi = df[df["bimodal"]].sort_values("separation", ascending=False).head(args.top)
    if len(bi):
        out = bi[cols].copy()
        out["mu1_lin"] = 10 ** out["mu1_log"]
        out["mu2_lin"] = 10 ** out["mu2_log"]
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    else:
        print("Пусто.")

    print(f"\n=== ТОП-{args.top} УНИМОДАЛЬНЫХ ХВОСТОВЫХ (контроль) ===")
    uni = df[df["unimodal_tail"]].sort_values("n_acts", ascending=False).head(args.top)
    print(uni[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}")
          if len(uni) else "Пусто.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nПолные метрики -> {args.out}")


if __name__ == "__main__":
    main()
