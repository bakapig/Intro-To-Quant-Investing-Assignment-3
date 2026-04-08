import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import warnings
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# LOAD DATA
# ==============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "monthly_returns.csv")
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
ASSETS = df.columns.tolist()
N = len(ASSETS)

print("=" * 80)
print("HMM REGIME ANALYSIS (Data-Driven Risk Regimes)")
print("=" * 80)

# ==============================================================================
# PART 1: FIT GAUSSIAN HMM
# ==============================================================================
# We fit a 2-state Hidden Markov Model directly on the asset returns.
# 'full' covariance captures the correlation structures, which is critical for optimization.
n_states = 2
print(f"Fitting GaussianHMM with {n_states} states and 'full' covariance matrix...")

X = df.values  # N assets
model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=2000, random_state=42)
model.fit(X)

print(f"Model converged: {model.monitor_.converged}")

# Predict the historical regime states
hidden_states = model.predict(X)
# Get smoothed probabilities
smoothed_probs = model.predict_proba(X)

# Order states so that State 0 = Low Volatility (Calm), State 1 = High Volatility (Crisis)
# We can use the trace of the covariance matrices to measure total portfolio variance
variances = np.array([np.trace(model.covars_[i]) for i in range(n_states)])
sorted_idx = np.argsort(variances)
state_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_idx)}

ordered_states = np.array([state_map[s] for s in hidden_states])
# Reorder means and covariances
means = model.means_[sorted_idx]
covars = model.covars_[sorted_idx]

# Map names for better readability
state_names = {0: "Calm / Low Volatility", 1: "Crisis / High Volatility"}
regimes_series = pd.Series(ordered_states, index=df.index, name="HMM_State")

# ==============================================================================
# PART 2: ANALYZE HMM REGIMES
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("HMM REGIME PARAMETERS (Annualized)")
print("=" * 80)

regime_stats = {}
for state_idx in range(n_states):
    state_name = f"HMM State {state_idx}: {state_names[state_idx]}"
    
    # Extract parameters for this state and annualize
    mu_ann = means[state_idx] * 12
    cov_ann = covars[state_idx] * 12
    vol_ann = np.sqrt(np.diag(cov_ann))
    
    # Convert covariance back to correlation matrix for display
    outer_vols = np.outer(vol_ann, vol_ann)
    corr = cov_ann / outer_vols
    corr_df = pd.DataFrame(corr, index=ASSETS, columns=ASSETS)
    
    n_months = np.sum(ordered_states == state_idx)
    
    regime_stats[state_name] = {
        "n_months": n_months,
        "mu": mu_ann,
        "cov": cov_ann,
        "vol": vol_ann,
        "corr": corr_df,
    }
    
    print(f"\n{'-' * 80}")
    print(f"  {state_name}  ({n_months} months observed historically)")
    print(f"{'-' * 80}")
    print(f"\n  Annualized Return & Volatility:")
    for i, a in enumerate(ASSETS):
        sr = mu_ann[i] / vol_ann[i] if vol_ann[i] > 0 else 0
        print(
            f"    {a:25s}  mu={mu_ann[i]:+.4f} ({mu_ann[i]*100:+5.1f}%)  "
            f"vol={vol_ann[i]:.4f} ({vol_ann[i]*100:5.1f}%)  SR={sr:+.3f}"
        )

    print(f"\n  Key Correlations:")
    print(
        f"    Equities-Bond:  {corr_df.iloc[0,1]:+.3f}   "
        f"Equities-REITs: {corr_df.iloc[0,2]:+.3f}   "
        f"Equities-Commod: {corr_df.iloc[0,3]:+.3f}"
    )

# For comparison, add the Full Sample
full_mu_ann = df.mean().values * 12
full_cov_ann = df.cov().values * 12
full_vol_ann = np.sqrt(np.diag(full_cov_ann))
regime_stats["Full Sample (2000-2026)"] = {
    "n_months": len(df),
    "mu": full_mu_ann,
    "cov": full_cov_ann,
    "vol": full_vol_ann,
    "corr": df.corr()
}

# ==============================================================================
# PART 3: PER-REGIME OPTIMIZATION (HMM)
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("PER-REGIME OPTIMIZATION RESULTS (HMM)")
print("=" * 80)

def run_optimizations(mu_vec, cov_mat, assets):
    """Run 3 optimizations with given mu and Sigma. Returns dict of results."""
    n = len(assets)
    b = [(0.0, 1.0)] * n
    eq_con = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w_init = np.ones(n) / n

    def p_ret(w):
        return w @ mu_vec

    def p_vol(w):
        return np.sqrt(w @ cov_mat @ w)

    def p_sr(w):
        v = p_vol(w)
        return (p_ret(w)) / v if v > 0 else 0

    results = {}

    # Obj 1: Max return s.t. vol <= 10%
    res1 = minimize(
        fun=lambda w: -p_ret(w),
        x0=w_init,
        method="SLSQP",
        bounds=b,
        constraints=[eq_con, {"type": "ineq", "fun": lambda w: 0.10 - p_vol(w)}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    results["Max Ret (vol<=10%)"] = res1

    # Obj 2: Min vol s.t. return >= 4%
    res2 = minimize(
        fun=p_vol,
        x0=w_init,
        method="SLSQP",
        bounds=b,
        constraints=[eq_con, {"type": "ineq", "fun": lambda w: p_ret(w) - 0.04}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    results["Min Vol (ret>=4%)"] = res2

    # Obj 3: Max Sharpe
    res3 = minimize(
        fun=lambda w: -p_sr(w),
        x0=w_init,
        method="SLSQP",
        bounds=b,
        constraints=[eq_con],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    results["Max Sharpe"] = res3

    return results, p_ret, p_vol


all_rows = []

for regime_name, stats in regime_stats.items():
    if "Full Sample" in regime_name:
        continue
        
    mu_r = stats["mu"]
    cov_r = stats["cov"]

    print(f"\n{'-' * 80}")
    print(f"  {regime_name}")
    print(f"{'-' * 80}")

    results, p_ret, p_vol = run_optimizations(mu_r, cov_r, ASSETS)

    for obj_name, res in results.items():
        w = res.x
        ret = p_ret(w)
        vol = p_vol(w)
        sr = ret / vol if vol > 0 else np.nan
        status = "OK" if res.success else "FAIL"

        weights_str = "  ".join(f"{w[i]*100:5.1f}%" for i in range(N))
        print(
            f"  {obj_name:22s}  Ret={ret*100:+5.1f}%  "
            f"Vol={vol*100:5.1f}%  SR={sr:+.3f}  "
            f"[{status}]  w=[{weights_str}]"
        )

        row = {
            "Regime": regime_name,
            "Objective": obj_name,
            "Ann_Return": ret,
            "Ann_Vol": vol,
            "Sharpe": sr,
            "Status": status,
        }
        for i, a in enumerate(ASSETS):
            row[a] = w[i]
        all_rows.append(row)


# ==============================================================================
# PART 4: VIZUALIZING THE HMM STATES OVER TIME
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("GENERATING HMM OVERLAY CHARTS")
print("=" * 80)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Chart: Rolling Risk vs HMM States ──
rolling_vol = df.rolling(window=24).std() * np.sqrt(12)
rolling_vol = rolling_vol.dropna()

fig_hmm, ax = plt.subplots(figsize=(14, 6), dpi=150)

# Plot rolling volatility
for a in ASSETS:
    ax.plot(rolling_vol.index, rolling_vol[a], label=a.replace("_", " "), linewidth=1.5)

ax.set_title(
    "Data-Driven Risk Regimes (HMM) vs Rolling Volatility",
    fontsize=14,
    fontweight="bold",
)
ax.set_ylabel("24-Month Annualized Volatility")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.grid(True, linestyle=":", alpha=0.6)

# Overlay HMM States as red blocks
# We'll plot State 1 (Crisis) as a red background highlight
is_crisis = ordered_states == 1

# Find contiguous segments of Crisis states to draw axvspan
crisis_starts = []
crisis_ends = []

in_crisis = False
for i, state in enumerate(is_crisis):
    if state and not in_crisis:
        crisis_starts.append(df.index[i])
        in_crisis = True
    elif not state and in_crisis:
        # End at the previous day
        crisis_ends.append(df.index[i-1])
        in_crisis = False

if in_crisis:
    crisis_ends.append(df.index[-1])

for start, end in zip(crisis_starts, crisis_ends):
    ax.axvspan(start, end, color="red", alpha=0.2)

# Make a custom legend
from matplotlib.patches import Patch
import matplotlib.lines as mlines

custom_lines = [mlines.Line2D([0], [0], color='blue', lw=1.5, label='Equities'),
                mlines.Line2D([0], [0], color='orange', lw=1.5, label='Bonds'),
                mlines.Line2D([0], [0], color='green', lw=1.5, label='REITs'),
                mlines.Line2D([0], [0], color='red', lw=1.5, label='Commodities'),
                Patch(facecolor='red', alpha=0.2, label='HMM Crisis State')]
ax.legend(handles=custom_lines, loc="upper left")

fig_hmm.tight_layout()
hmm_chart_path = os.path.join(OUT_DIR, "hmm_regimes_overlay.png")
fig_hmm.savefig(hmm_chart_path)
plt.close(fig_hmm)
print(f"Saved: {hmm_chart_path}")
