"""
Exploratory Data Analysis (EDA) for monthly returns data.

Generates summary statistics, distribution plots, correlation analysis,
rolling statistics, and cumulative wealth charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "monthly_returns.csv")
OUTPUT_DIR = os.path.dirname(__file__)

df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
ASSETS = df.columns.tolist()

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Assets: {ASSETS}\n")

# ── 1. Descriptive Statistics ─────────────────────────────────────────────────
print("=" * 60)
print("1. DESCRIPTIVE STATISTICS (monthly)")
print("=" * 60)
desc = df.describe().T
desc["skewness"] = df.skew()
desc["kurtosis"] = df.kurtosis()  # excess kurtosis
print(desc.to_string())

print(f"\n{'─' * 60}")
print("Annualized Statistics")
print("─" * 60)
ann = pd.DataFrame({
    "Ann. Return":    df.mean() * 12,
    "Ann. Volatility": df.std() * np.sqrt(12),
    "Sharpe (rf=0)":  (df.mean() * 12) / (df.std() * np.sqrt(12)),
    "Max Drawdown":   ((1 + df).cumprod() / (1 + df).cumprod().cummax() - 1).min(),
})
print(ann.to_string())

# ── 2. Correlation Matrix ────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("2. CORRELATION MATRIX")
print("=" * 60)
corr = df.corr()
print(corr.to_string())

# ── 3. Plots ──────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 3a. Cumulative wealth (growth of $1)
fig, ax = plt.subplots(figsize=(12, 6))
cumulative = (1 + df).cumprod()
for i, col in enumerate(ASSETS):
    ax.plot(cumulative.index, cumulative[col], label=col, color=COLORS[i], linewidth=1.2)
ax.set_title("Growth of $1 Investment", fontsize=14)
ax.set_ylabel("Cumulative Wealth ($)")
ax.legend()
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cumulative_wealth.png"), dpi=150)
print(f"\nSaved: cumulative_wealth.png")

# 3b. Return distributions (histograms)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, (ax, col) in enumerate(zip(axes.flat, ASSETS)):
    ax.hist(df[col], bins=40, color=COLORS[i], alpha=0.75, edgecolor="white")
    ax.axvline(df[col].mean(), color="black", linestyle="--", linewidth=1, label=f"Mean={df[col].mean():.4f}")
    ax.set_title(col, fontsize=12)
    ax.set_xlabel("Monthly Return")
    ax.legend(fontsize=9)
fig.suptitle("Monthly Return Distributions", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "return_distributions.png"), dpi=150)
print(f"Saved: return_distributions.png")

# 3c. Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(ASSETS)))
ax.set_yticks(range(len(ASSETS)))
ax.set_xticklabels(ASSETS, rotation=45, ha="right")
ax.set_yticklabels(ASSETS)
for i in range(len(ASSETS)):
    for j in range(len(ASSETS)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Correlation Matrix", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
print(f"Saved: correlation_heatmap.png")

# 3d. Rolling 12-month volatility
fig, ax = plt.subplots(figsize=(12, 6))
rolling_vol = df.rolling(12).std() * np.sqrt(12)
for i, col in enumerate(ASSETS):
    ax.plot(rolling_vol.index, rolling_vol[col], label=col, color=COLORS[i], linewidth=1.2)
ax.set_title("Rolling 12-Month Annualized Volatility", fontsize=14)
ax.set_ylabel("Volatility")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rolling_volatility.png"), dpi=150)
print(f"Saved: rolling_volatility.png")

# 3e. Rolling 12-month correlation with US Equities
fig, ax = plt.subplots(figsize=(12, 6))
for i, col in enumerate(ASSETS[1:], start=1):  # skip US_Equities itself
    rolling_corr = df["US_Equities"].rolling(12).corr(df[col])
    ax.plot(rolling_corr.index, rolling_corr, label=f"Equities vs {col}", color=COLORS[i], linewidth=1.2)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Rolling 12-Month Correlation with US Equities", fontsize=14)
ax.set_ylabel("Correlation")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rolling_correlation.png"), dpi=150)
print(f"Saved: rolling_correlation.png")

# 3f. Drawdown chart
fig, ax = plt.subplots(figsize=(12, 6))
for i, col in enumerate(ASSETS):
    wealth = (1 + df[col]).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    ax.fill_between(drawdown.index, drawdown, alpha=0.3, color=COLORS[i], label=col)
ax.set_title("Drawdown from Peak", fontsize=14)
ax.set_ylabel("Drawdown")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "drawdowns.png"), dpi=150)
print(f"Saved: drawdowns.png")

plt.close("all")
print("\nEDA complete.")
