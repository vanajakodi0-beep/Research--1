#!/usr/bin/env python3
"""
Additional shift scenarios for the break-aware CausalImpact study.

Runs every method of the research update on the original two conditions (no shift,
5% common shift) and on new scenarios:
  A  shift only in the treated store
  B  different shift size per store (U(2%,8%), mean 5%)
  C  slow gradual shift (linear ramp 0 -> 10% over the 43 post weeks, mean ~5%)
  D  control store slightly affected by the treatment (leakage = 10% of the effect)
  D2 leakage plus a 5% common shift

Methods: standard BSTS, BSTS with control, BSTS with best control, fixed widening (k),
break-aware (lambda), fixed widening at break-aware width, conformal (scaled and
absolute), DiD (random / best control), synthetic control, point-estimate correction.

Usage:
    python run_shift_scenarios.py --data Walmart_Sales.csv --out results

NOTE: tfcausalimpact cannot be installed in every environment, so the BSTS here is a
local-level model fitted by maximum likelihood (Kalman filter), with an optional
regression on the control store. Only numpy, scipy and pandas are needed.
"""
import argparse, json, math, os, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, nnls

T0, H, T = 100, 43, 143          # pre weeks, post weeks, total weeks
NS = 45                          # stores
TRUE_EFFECT = 0.03
Z = 1.96
SCEN = ["none", "common5", "A_treated_only", "B_store_sizes", "C_gradual", "D_leak", "D2_leak_shift"]
ORIGINAL = ["none", "common5"]   # scenarios used for calibration (as in the paper)
K_GRID = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
LAM_GRID = [0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40]
N_SPLITS = 20


# ----------------------------------------------------------------------------- data
def load_real(path):
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: {path} not found. The real Walmart sales file is required; stopping.")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    wide = df.pivot_table(index="Store", columns="Date", values="Weekly_Sales").sort_index(axis=1)
    wide = wide.iloc[:, :T]
    assert wide.shape == (NS, T), wide.shape
    return wide.values / 1e0, list(wide.columns)


# ----------------------------------------------------------------------------- BSTS (local level)
def _filter(y, q):
    """Local-level Kalman filter in units of the observation variance. Exact diffuse start.
    Returns innovations v[1:], variances F[1:], next-step level a_next and variance P_next."""
    n = len(y)
    a, P = y[0], 1.0 + q
    v = np.empty(n - 1); F = np.empty(n - 1)
    for t in range(1, n):
        f = P + 1.0
        e = y[t] - a
        k = P / f
        v[t - 1] = e; F[t - 1] = f
        a = a + k * e
        P = P * (1 - k) + q
    return v, F, a, P


def fit_ll(y, x=None):
    """Concentrated ML fit of a local-level model (optionally with one regressor).
    y, x in millions. Returns dict with everything needed for cumulative forecasts."""
    def nll(lq):
        q = math.exp(lq)
        v, F, _, _ = _filter(y, q)
        if x is None:
            beta = 0.0; r = v
        else:
            vx, _, _, _ = _filter(x, q)
            beta = np.sum(vx * v / F) / np.sum(vx * vx / F)
            r = v - beta * vx
        s2 = np.sum(r * r / F) / len(v)
        # log-normal prior on the level sd, as in the default CausalImpact model (median 0.01 * sd(y), log-scale sd 0.5)
        pen = 0.5 * ((0.5 * math.log(q * s2) - math.log(0.01 * np.std(y))) / 0.5) ** 2
        return 0.5 * (len(v) * math.log(s2) + np.sum(np.log(F))) + pen
    res = minimize_scalar(nll, bounds=(-12, 4), method="bounded", options={"xatol": 1e-3})
    q = math.exp(res.x)
    v, F, a_y, P = _filter(y, q)
    out = dict(q=q, P=P, a_y=a_y, beta=0.0, var_beta=0.0, a_x=0.0)
    if x is not None:
        vx, _, a_x, _ = _filter(x, q)
        sxx = np.sum(vx * vx / F)
        beta = np.sum(vx * v / F) / sxx
        r = v - beta * vx
        out.update(beta=beta, a_x=a_x, s2=np.sum(r * r / F) / len(v))
        out["var_beta"] = out["s2"] / sxx
    else:
        out["s2"] = np.sum(v * v / F) / len(v)
    return out


_W = None
def _sum_sq():  # sum_{j=2}^{H} (H-j+1)^2
    return sum((H - j + 1) ** 2 for j in range(2, H + 1))


def cum_effect(fit, y_post_sum, x_post_sum=0.0):
    """Cumulative effect estimate and its standard error (millions) for a stored fit."""
    pred = H * fit["a_y"] + fit["beta"] * (x_post_sum - H * fit["a_x"])
    var = fit["s2"] * (H ** 2 * fit["P"] + fit["q"] * _sum_sq() + H) + fit["var_beta"] * (x_post_sum - H * fit["a_x"]) ** 2
    return y_post_sum - pred, math.sqrt(var)


# ----------------------------------------------------------------------------- control diagnostic
def yoy_z(series):
    """Year-over-year log growth; change in mean growth after the intervention / Newey-West (4 lags) SE."""
    ls = np.log(series)
    r = ls[52:] - ls[:-52]
    d = (np.arange(52, T) >= T0).astype(float)
    X = np.column_stack([np.ones_like(d), d])
    beta = np.linalg.solve(X.T @ X, X.T @ r)
    u = r - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for l in range(1, 5):
        w = 1 - l / 5.0
        G = xu[l:].T @ xu[:-l]
        S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return beta[1], beta[1] / math.sqrt(V[1, 1])


# ----------------------------------------------------------------------------- scenarios
def scenario_multipliers(name, rng):
    """Return (treated_mult, control_mult) as (NS, H) arrays applied to post-period sales."""
    ramp = np.arange(1, H + 1) / H
    shiftT = np.zeros((NS, H)); shiftC = np.zeros((NS, H)); leak = 0.0
    if name == "common5":
        shiftT[:] = 0.05; shiftC[:] = 0.05
    elif name == "A_treated_only":
        shiftT[:] = 0.05
    elif name == "B_store_sizes":
        s = rng.uniform(0.02, 0.08, NS)[:, None]
        shiftT = np.repeat(s, H, 1); shiftC = shiftT.copy()
    elif name == "C_gradual":
        shiftT = np.tile(0.10 * ramp, (NS, 1)); shiftC = shiftT.copy()
    elif name == "D_leak":
        leak = 0.1 * TRUE_EFFECT
    elif name == "D2_leak_shift":
        shiftT[:] = 0.05; shiftC[:] = 0.05; leak = 0.1 * TRUE_EFFECT
    return 1 + TRUE_EFFECT + shiftT, 1 + shiftC + leak


def conf_q(vals, alpha=0.05):
    v = np.sort(np.asarray(vals)); n = len(v)
    return v[min(n, math.ceil((n + 1) * (1 - alpha))) - 1]


# ----------------------------------------------------------------------------- main experiment
def main(Y, out, seed0=0):
    os.makedirs(out, exist_ok=True)
    Ym = Y / 1e6
    pre, post = Ym[:, :T0], Ym[:, T0:]
    post_sum = post.sum(1)
    true_eff = TRUE_EFFECT * post_sum                       # millions, per treated store
    t_start = time.time()

    # --- per-store standard BSTS fits (pre-period only, so scenario independent)
    std_fit = [fit_ll(pre[i]) for i in range(NS)]
    # --- per-pair BSTS-with-control fits
    pair_fit = {}
    for i in range(NS):
        for j in range(NS):
            if i != j:
                pair_fit[(i, j)] = fit_ll(pre[i], pre[j])
    print(f"BSTS fits done in {time.time()-t_start:.0f}s", flush=True)
    corr = np.corrcoef(pre)
    sum_pre = pre.sum(1)

    # --- per-scenario series (scenario randomness shared across splits for reproducibility)
    scen_rng = {s: np.random.default_rng(1000 + k) for k, s in enumerate(SCEN)}
    scen_data = {}
    for s in SCEN:
        tm, cm = scenario_multipliers(s, scen_rng[s])
        Yt = Ym.copy(); Yt[:, T0:] = post * tm            # series when the store is treated
        Yc = Ym.copy(); Yc[:, T0:] = post * cm            # series when the store is a control / donor
        zs = np.array([yoy_z(Yc[j] * 1e6) for j in range(NS)])   # (drift, z) of every store as a control
        # standard estimates for every treated store
        est = np.zeros(NS); se = np.zeros(NS)
        for i in range(NS):
            est[i], se[i] = cum_effect(std_fit[i], Yt[i, T0:].sum())
        scen_data[s] = dict(Yt=Yt, Yc=Yc, drift=zs[:, 0], z=zs[:, 1], est=est, se=se)
    print("scenarios built", flush=True)

    # --- DiD / SC pre-period scales and weights (scenario independent)
    def placebo_scale_did(i, j):
        pl = []
        for o in range(40, T0 - H + 1):
            ratio = pre[i, :o].sum() / pre[j, :o].sum()
            pl.append(np.sum(pre[i, o:o + H] - ratio * pre[j, o:o + H]))
        return np.quantile(np.abs(pl), 0.95)

    def sc_weights(i, donors, upto):
        D = pre[donors, :upto].T; yv = pre[i, :upto]
        sc = yv.mean(); M = 1e3
        A = np.vstack([D / sc, M * np.ones((1, len(donors)))]); b = np.concatenate([yv / sc, [M]])
        w, _ = nnls(A, b)
        return w / w.sum()

    def sc_scale(i, donors):
        pl = []
        for o in range(40, T0 - H + 1):
            w = sc_weights(i, donors, o)
            pl.append(np.sum(pre[i, o:o + H] - w @ pre[donors, o:o + H]))
        return np.quantile(np.abs(pl), 0.95)

    did_scale = {}
    for i in range(NS):
        for j in range(NS):
            if i != j:
                did_scale[(i, j)] = placebo_scale_did(i, j)
    print(f"DiD scales done {time.time()-t_start:.0f}s", flush=True)

    all_rows = []; split_summaries = []
    for sp in range(N_SPLITS):
        rng = np.random.default_rng(seed0 + sp)
        perm = rng.permutation(NS)
        cal = np.sort(perm[:22]); tst = np.sort(perm[22:])
        cal_pairs = [(i, j) for i in cal for j in cal if i != j]
        tst_pairs = [(i, j) for i in tst for j in tst if i != j]
        ci = np.array([p[0] for p in cal_pairs]); cj = np.array([p[1] for p in cal_pairs])
        ti = np.array([p[0] for p in tst_pairs]); tj = np.array([p[1] for p in tst_pairs])

        # ---------- calibration (original two scenarios only)
        def cov_std_pairs(s, mult_fn):
            d = scen_data[s]
            lo_hi = mult_fn(d)
            return lo_hi
        # fixed widening k: smallest k with cal coverage >= 95% in both original scenarios
        def cover_k(k, s):
            d = scen_data[s]; err = np.abs(d["est"][ci] - true_eff[ci]); return np.mean(err <= k * Z * d["se"][ci])
        k_sel = next((k for k in K_GRID if all(cover_k(k, s) >= .95 for s in ORIGINAL)), None)
        k_ok = k_sel is not None; k_sel = k_sel if k_ok else K_GRID[-1]
        def cover_l(l, s):
            d = scen_data[s]; err = np.abs(d["est"][ci] - true_eff[ci])
            return np.mean(err <= (Z + l * np.abs(d["z"][cj])) * d["se"][ci])
        lam_sel = next((l for l in LAM_GRID if all(cover_l(l, s) >= .95 for s in ORIGINAL)), None)
        lam_ok = lam_sel is not None; lam_sel = lam_sel if lam_ok else LAM_GRID[-1]
        # detection threshold: largest |z| of the cal control stores with no shift (finite-sample rule, n=22)
        thr = conf_q(np.abs(scen_data["none"]["z"][cal]))
        # conformal (scaled & absolute) on calibration stores, pooled over the two original scenarios
        errs = np.concatenate([np.abs(scen_data[s]["est"][cal] - true_eff[cal]) for s in ORIGINAL])
        errs_sc = np.concatenate([np.abs(scen_data[s]["est"][cal] - true_eff[cal]) / scen_data[s]["se"][cal] for s in ORIGINAL])
        q_abs = conf_q(errs); q_scaled = conf_q(errs_sc)

        # best control per treated store (highest pre correlation within its group)
        def best_ctrl(i, grp):
            others = [j for j in grp if j != i]; return others[int(np.argmax(corr[i, others]))]
        # DiD / SC multipliers from calibration stores
        def did_err(i, j, s):
            d = scen_data[s]; ratio = sum_pre[i] / sum_pre[j]
            return abs(d["Yt"][i, T0:].sum() - ratio * d["Yc"][j, T0:].sum() - true_eff[i])
        mult = {}
        v = [did_err(i, j, s) / did_scale[(i, j)] for s in ORIGINAL for (i, j) in cal_pairs]; mult["did_rand"] = conf_q(v)
        v = [did_err(i, best_ctrl(i, cal), s) / did_scale[(i, best_ctrl(i, cal))] for s in ORIGINAL for i in cal]; mult["did_best"] = conf_q(v)
        sc_cal = {}
        for i in cal:
            donors = [j for j in cal if j != i]; sc_cal[i] = (donors, sc_weights(i, donors, T0), sc_scale(i, donors))
        v = []
        for s in ORIGINAL:
            d = scen_data[s]
            for i in cal:
                donors, w, sc = sc_cal[i]
                v.append(abs(d["Yt"][i, T0:].sum() - w @ d["Yc"][donors, T0:].sum(1) - true_eff[i]) / sc)
        mult["sc"] = conf_q(v)
        sc_tst = {}
        for i in tst:
            donors = [j for j in tst if j != i]; sc_tst[i] = (donors, sc_weights(i, donors, T0), sc_scale(i, donors))
        best_t = {i: best_ctrl(i, list(tst)) for i in tst}

        rec = dict(split=sp, k=k_sel, k_reached=k_ok, lam=lam_sel, lam_reached=lam_ok, thr=thr, q_abs=q_abs, q_scaled=q_scaled, **{"m_" + a: b for a, b in mult.items()})
        split_summaries.append(rec)

        # ---------- evaluation on test pairs
        for s in SCEN:
            d = scen_data[s]
            tau_true = true_eff[ti]
            est = d["est"][ti]; se = d["se"][ti]; err = est - tau_true
            zc = np.abs(d["z"][tj])
            w_std = 2 * Z * se
            res = {}
            def add(name, half, point=None, idx=None):
                p = err if point is None else point
                res[name] = dict(cov=float(np.mean(np.abs(p) <= half)), width=float(np.mean(2 * half)),
                                 mae=float(np.mean(np.abs(p))), bias=float(np.mean(p)))
            add("standard", Z * se)
            add("fixed_k", k_sel * Z * se)
            add("break_aware", (Z + lam_sel * zc) * se)
            ba_w = np.mean(2 * (Z + lam_sel * zc) * se)
            k_eq = ba_w / np.mean(w_std)
            add("fixed_equal_width", k_eq * Z * se)
            res["fixed_equal_width"]["k_eq"] = float(k_eq)
            add("conformal_scaled", q_scaled * se)
            add("conformal_abs", np.full_like(se, q_abs))
            # BSTS with control (random control = the paired control)
            e_c = np.zeros(len(ti)); s_c = np.zeros(len(ti))
            for n, (i, j) in enumerate(tst_pairs):
                e, sd = cum_effect(pair_fit[(i, j)], d["Yt"][i, T0:].sum(), d["Yc"][j, T0:].sum())
                e_c[n] = e; s_c[n] = sd
            add("bsts_control", Z * s_c, point=e_c - tau_true)
            # BSTS with best control
            e_b = []; s_b = []; tb = []
            for i in tst:
                j = best_t[i]; e, sd = cum_effect(pair_fit[(i, j)], d["Yt"][i, T0:].sum(), d["Yc"][j, T0:].sum())
                e_b.append(e); s_b.append(sd); tb.append(true_eff[i])
            e_b = np.array(e_b); s_b = np.array(s_b); tb = np.array(tb)
            res["bsts_best_control"] = dict(cov=float(np.mean(np.abs(e_b - tb) <= Z * s_b)), width=float(np.mean(2 * Z * s_b)),
                                            mae=float(np.mean(np.abs(e_b - tb))), bias=float(np.mean(e_b - tb)))
            # DiD random control
            e_d = np.array([d["Yt"][i, T0:].sum() - (sum_pre[i] / sum_pre[j]) * d["Yc"][j, T0:].sum() for (i, j) in tst_pairs]) - tau_true
            half = mult["did_rand"] * np.array([did_scale[p] for p in tst_pairs])
            add("did_random", half, point=e_d)
            # DiD best control
            e_d = np.array([d["Yt"][i, T0:].sum() - (sum_pre[i] / sum_pre[best_t[i]]) * d["Yc"][best_t[i], T0:].sum() - true_eff[i] for i in tst])
            half = np.array([mult["did_best"] * did_scale[(i, best_t[i])] for i in tst])
            res["did_best"] = dict(cov=float(np.mean(np.abs(e_d) <= half)), width=float(np.mean(2 * half)),
                                   mae=float(np.mean(np.abs(e_d))), bias=float(np.mean(e_d)))
            # synthetic control
            e_s = []; h_s = []
            for i in tst:
                donors, w, sc = sc_tst[i]
                e_s.append(d["Yt"][i, T0:].sum() - w @ d["Yc"][donors, T0:].sum(1) - true_eff[i]); h_s.append(mult["sc"] * sc)
            e_s = np.array(e_s); h_s = np.array(h_s)
            res["synthetic"] = dict(cov=float(np.mean(np.abs(e_s) <= h_s)), width=float(np.mean(2 * h_s)),
                                    mae=float(np.mean(np.abs(e_s))), bias=float(np.mean(e_s)))
            # point-estimate correction (flag if |z| > threshold): subtract drift * expected treated sales
            flag = np.abs(d["z"][tj]) > thr
            exp_sales = np.array([Ym[i, T0 - 52:T - 52].sum() for i in ti])
            corr_pt = err - flag * d["drift"][tj] * exp_sales
            res["point_correction"] = dict(cov=float("nan"), width=float("nan"), mae=float(np.mean(np.abs(corr_pt))), bias=float(np.mean(corr_pt)))
            # diagnostic
            ctrl_z = np.abs(d["z"][tst]); base_z = np.abs(scen_data["none"]["z"][tst])
            res["diagnostic"] = dict(detect=float(np.mean(ctrl_z > thr)), mean_abs_z=float(np.mean(ctrl_z)))
            # treated-store-level coverage vectors for bootstrap (primary split only)
            if sp == 0:
                store_of_pair = ti
                cov_ba = (np.abs(err) <= (Z + lam_sel * zc) * se).astype(float)
                cov_fe = (np.abs(err) <= k_eq * Z * se).astype(float)
                boots = []
                brng = np.random.default_rng(99)
                for _ in range(2000):
                    samp = brng.choice(tst, size=len(tst), replace=True)
                    num = 0.0; den = 0.0; num2 = 0.0
                    for ssi in samp:
                        m = store_of_pair == ssi
                        num += cov_ba[m].sum(); num2 += cov_fe[m].sum(); den += m.sum()
                    boots.append(100 * (num - num2) / den)
                res["bootstrap_ba_minus_fixed_eq"] = dict(diff=float(100 * (cov_ba.mean() - cov_fe.mean())),
                                                          lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)))
            for m, r in res.items():
                row = dict(split=sp, scenario=s, method=m); row.update(r); all_rows.append(row)
        print(f"split {sp} done {time.time()-t_start:.0f}s  k={k_sel} lam={lam_sel} thr={thr:.2f}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out, "all_results.csv"), index=False)
    pd.DataFrame(split_summaries).to_csv(os.path.join(out, "split_params.csv"), index=False)
    return df, pd.DataFrame(split_summaries)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Walmart_Sales.csv", help="Walmart CSV with Store, Date, Weekly_Sales")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    Y, dates = load_real(a.data)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "sales_used.npy"), Y)
    json.dump(dict(source="real Walmart data", file=os.path.basename(a.data)), open(os.path.join(a.out, "source.json"), "w"))
    print("data source: real Walmart data", Y.shape, "mean weekly sales %.0f" % Y.mean())
    main(Y, a.out)
