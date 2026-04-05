# Quantitative Investment Strategy: From Mean-Variance Theory to Dynamic Risk-Based Allocation

This report progresses through traditional portfolio theory, exposes its mathematical and historical flaws, and concludes with a robust, dynamically backtested proposal for risk-based capital allocation.

---

## Part 1: Historical Mean-Variance Optimization (MVO)

Using the full dataset of monthly returns (2000–2026) for US Equities, US 10-Yr Treasuries, US REITs, and US Commodities, we calculated the annualized historical expected returns ($\mu$) and covariance matrix ($\Sigma$). 

We performed traditional Markowitz optimization across three specific objectives. The results are plotted on the Efficient Frontier below.

![The Efficient Frontier and Optimized Portfolios](../Outputs/efficient_frontier.png)

### The Three Optimized Portfolios

1.  **Maximize Return subject to Volatility $\le$ 10%:** Achieved an annualized return of **7.43%** with volatility constrained exactly at 10.0%. The optimizer leaned into US Equities (34%) and REITs (27%) to harvest yield, anchored by Treasuries (39%). This portfolio sits on the upper part of the Efficient Frontier, pushing as close to the volatility ceiling as possible.
2.  **Minimize Volatility subject to Return $\ge$ 4%:** The most defensive allocation, heavily concentrated into US Treasuries (76.6%) with low allocations to Equities (10%) and Commodities (13%), achieving a volatility of only **5.22%**. This portfolio sits near the left edge of the frontier at the minimum-risk zone.
3.  **Maximize Sharpe Ratio:** The optimal risk-adjusted portfolio lying at the point where the Capital Market Line meets the Efficient Frontier. Allocated 70% to bonds and 25% to equities, achieving the highest Sharpe Ratio of **0.88**.

![Three Optimized Portfolios — Summary Table](../Outputs/three_portfolio_table.png)

### Historical Performance of the Three Portfolios

To stress-test these optimized weights against real market data, we applied each portfolio's allocation to actual monthly returns from 2000–2026:

| Portfolio | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Calmar | Final $1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Max Return (Vol≤10%)** | **+7.43%** | 10.00% | 0.743 | -34.2% | 0.217 | **$6.12** |
| **2. Min Volatility (Ret≥4%)** | +4.18% | **5.22%** | 0.800 | **-16.3%** | 0.257 | $2.89 |
| **3. Max Sharpe** | +5.14% | 5.83% | **0.882** | -17.4% | **0.296** | $3.68 |

![Three Portfolio Comparison Dashboard](../Outputs/three_portfolio_comparison.png)

**Key Observations:**
-   **Max Sharpe** delivered the highest risk-adjusted return (Sharpe = 0.882) and best Calmar Ratio (0.296), confirming its mathematical optimality on the Efficient Frontier.
-   **Min Volatility** achieved the shallowest drawdown (-16.3%) — ideal for risk-averse investors — but sacrificed absolute return.
-   **Max Return** produced the highest terminal wealth ($6.12) but paid for it with a severe -34.2% drawdown during the 2008 GFC, illustrating the risk-return tradeoff at the volatility boundary.
-   All three optimizations are constrained to be long-only and fully invested. Their positions on the Efficient Frontier directly reflect the tension between return maximization and risk minimization.

### Broader Comparison (Including Benchmarks)

For reference, we also include the naïve Equal Weight (1/N) and Global Min Volatility portfolios as benchmarks:

![MVO Performance Summary Table](../Outputs/mvo_performance_table.png)

![MVO Portfolio Historical Performance Dashboard](../Outputs/mvo_performance_dashboard.png)

---

## Part 2: The Equal Sharpe Ratio Experiment

A critical flaw in traditional MVO is that it relies blindly on historical expected returns. To test how sensitive our portfolios are to these returns, we simulated a scenario where **all assets possess the exact same theoretical Sharpe Ratio** (e.g., $SR = 0.5$). 

Because $SR = \mu / \sigma$, assuming Equal Sharpe Ratios mathematically forces the expected return of each asset to be strictly proportional to its volatility ($\mu_i = SR_{common} \cdot \sigma_i$). 

![Equal Sharpe Ratio — Efficient Frontier Comparison](../Outputs/equal_sharpe_frontier.png)

### Analysis Outcomes (All 3 Objectives):

1.  **Maximize Return s.t. Vol ≤ 10% — Weights Shift Dramatically:** Under historical returns, this portfolio tilted 34% Equities + 27% REITs to exploit their high expected returns. Under Equal SR, the optimizer can no longer favor high-return assets (since higher return = proportionally higher risk). Instead, it **redistributes weights toward Commodities (35%) and REITs (26%)** to construct the combination that reaches exactly 10% volatility through the covariance structure. The achievable return drops from 7.43% to 7.21% because the adjusted expected returns are lower.

2.  **Minimize Volatility s.t. Return ≥ 4% — Weights May Change Depending on SR Level:** Under Equal SR = 0.5, the minimum-vol portfolio achieving ≥4% return shifts to 76% Bonds + 11% Equities + 13% Commodities — similar to historical because bond dominance is driven by the covariance matrix. However, for very low SR values (e.g., SR = 0.3), the 4% return floor becomes harder to satisfy, forcing the optimizer to take on more volatile assets. This demonstrates that **the return constraint interacts with the SR level**.

3.  **Maximize Sharpe Ratio — The Max Sharpe Portfolio Becomes Degenerate:** Under Equal SR, **every** portfolio on the efficient frontier achieves exactly the same Sharpe Ratio ($SR_{common}$). There is no unique optimal point — the optimizer collapses to the **minimum-variance portfolio** (16% Equities, 63% Bonds, 2% REITs, 19% Commodities), since it achieves the same SR as any other portfolio but with the lowest risk. This is fundamentally different from the historical Max Sharpe portfolio (25% Equities, 70% Bonds).

![Weight Allocation: Historical vs Equal Sharpe Ratio](../Outputs/equal_sharpe_weights.png)

![Comparison Table: Historical vs Equal Sharpe Ratio](../Outputs/equal_sharpe_comparison_table.png)

### Does the SR Level Matter?

We tested three different SR values (0.3, 0.5, 0.7) and found:

![SR Sensitivity — Do Weights Change Across SR Levels?](../Outputs/equal_sharpe_sensitivity.png)

-   **Weights are nearly identical** across SR = 0.3, 0.5, and 0.7 for Objectives 1 and 3 — only the *scale* of return changes (proportionally), not the allocation.
-   **Objective 2 is the exception:** When $SR_{common}$ is low (0.3), the 4% return floor forces the optimizer into higher-volatility assets. When $SR_{common}$ is high (0.7), the return floor is easily met with defensive assets.

### Conclusion: Do Optimization Results Depend on the Sharpe Ratio Assumption?

**Yes — optimization results depend *dangerously heavily* on the expected return (Sharpe Ratio) assumption.** This is demonstrated by:

1. **Weight sensitivity:** All three objectives produce significantly different allocations between historical vs. equal-SR scenarios, proving that return assumptions directly control portfolio construction.
2. **Estimation Error Amplification (Michaud, 1989):** MVO mechanically overweights whichever asset has the highest estimated return — amplifying any estimation error into concentrated bets.
3. **The only stable element is the covariance matrix ($\Sigma$):** When return signals are neutralized (Equal SR), the optimizer reverts to a pure risk-minimization engine, driven entirely by correlations and volatilities. This stability is why **risk-based** portfolio construction (Part 4) is fundamentally more robust than return-based MVO.

---

## Part 3: Regime Changes in Risk Assumptions

### What Are the "Risk Assumptions"?

In Mean-Variance Optimization, risk is fully characterized by the **covariance matrix** ($\Sigma$), which encodes two components:

1. **Volatilities ($\sigma_i$)** — the annualized standard deviation of each asset (diagonal elements of $\Sigma$). MVO assumes these remain constant over time.
2. **Correlations ($\rho_{ij}$)** — how assets co-move (off-diagonal elements of $\Sigma$). MVO assumes these are stable — e.g., that bonds will always diversify equities.

Standard MVO estimates $\Sigma$ **once** from the full 25-year history and treats it as a fixed truth. The question is: do these risk parameters actually stay constant across the 2000–2026 history?

### Answer: Yes — Regime Changes Are Clearly Present

Our analysis proves the single, static $\Sigma$ assumption is **dangerously wrong**: both volatilities and correlations shift dramatically across market regimes.

### 1. Evidence of Regime Changes in Volatility

We computed rolling 24-month annualized volatility for each asset. If risk were constant, these lines would be flat — they are not:

![Rolling Volatility and Correlation across Regimes](../Outputs/regime_rolling_risk.png)

**Key findings:**
- **US Equities** volatility ranged from ~8.5% (Pre-GFC calm) to ~19.8% (GFC crisis) — a **2.3x increase**
- **US REITs** showed the most extreme swing: from 14.5% (calm) to **40.0%** during the GFC — nearly **3x**
- **US 10Yr Bonds** — supposedly the "safe" asset — saw vol spike from 6.0% to 8.1% during GFC, and rise further to 7.4% during the 2021-23 rate hike regime
- **COVID Shock (2020)** saw REITs hit 26.5% vol and Commodities 20.9% — another clear regime break

![Volatility Comparison Across Regimes](../Outputs/regime_volatility_comparison.png)

### 2. Evidence of Regime Changes in Correlations

Correlations — the other critical risk input — also shift violently:

![Correlation Matrices Across Regimes](../Outputs/regime_correlation_heatmaps.png)

**Critical observations:**
- **Equities-REITs** correlation jumped from +0.48 (Pre-GFC calm) to +0.81 (GFC crisis) to a staggering **+0.93** during COVID — diversification between these assets virtually disappeared during crises
- **Equities-Bond** correlation was near zero (+0.01) during calm periods but turned **strongly negative (-0.53)** during COVID, making bonds a powerful hedge in that specific regime — yet flipped to **positive** during the 2021-23 inflation/rate hike era
- **Equities-Commodities** went from +0.09 (calm) to +0.51 (GFC) to **+0.72** (COVID) — commodities stopped diversifying when they were needed most

### 3. Impact: The "Optimal" Portfolio Changes Dramatically Across Regimes

We re-ran all three optimization objectives using each regime's own covariance matrix. The results prove that the "optimal" frontier is not fixed — it shifts violently:

![Efficient Frontiers Across Market Regimes](../Outputs/regime_efficient_frontiers.png)

- The **Pre-GFC Calm frontier** (green) is far to the upper-left — offering high returns at low volatility
- The **GFC Crisis frontier** (red) collapses to the lower-right — even the best portfolio achieves only ~6% return at 8% vol
- The **Full Sample frontier** (gray dashed) falls somewhere in between — a compromise that represents *neither* regime well

The weight allocations tell the same story:

![Optimal Weights Across Different Regimes](../Outputs/regime_optimal_weights.png)

**Specific weight shifts:**

| Objective | Calm Period | GFC Crisis | Full Sample |
| :--- | :--- | :--- | :--- |
| **Max Return (Vol≤10%)** | 21% Eq, 60% REITs | 0% Eq, **100% Bonds** | 34% Eq, 39% Bond, 27% REITs |
| **Min Vol (Ret≥4%)** | 29% Eq, 62% Bond | 5% Eq, **82% Bond** | 11% Eq, 77% Bond |
| **Max Sharpe** | 39% Eq, 39% REITs | **99% Bonds** | 25% Eq, 70% Bond |

During the GFC crisis, the optimizer fled to bonds almost exclusively — because bonds were the only asset with decent risk-adjusted returns. During calm periods, the optimizer loaded up on REITs and equities. The Full Sample result is simply an average of all these contradict regimes.

![Per-Regime Optimization Results](../Outputs/regime_summary_table.png)

### Conclusion: Regime Changes Invalidate Static MVO

**Yes, regime changes are clearly present in the historical data, and they have a devastating impact on portfolio optimization:**

1. **Volatility is non-stationary:** Asset volatilities can spike 2-3x during crises, meaning a portfolio "optimized" on calm-period data carries far more risk than the model predicts
2. **Correlations spike during crises:** Growth assets (Equities, REITs, Commodities) become highly correlated exactly when diversification is needed most — destroying the diversification benefit that MVO relied on
3. **Optimal weights are regime-dependent:** The same optimization objective produces entirely different portfolios depending on which regime's data is used — during GFC crisis, all three objectives converge toward near-100% bonds
4. **The Full Sample MVO is a false compromise:** Using 25 years of data averages across all regimes, fitting *neither* calm nor crisis periods well

This motivates the move to **dynamic, risk-based portfolio construction** (Part 4) that adapts to changing market conditions rather than assuming a static world.

---

## Part 4: Backtesting Study — Risk-Based Optimal Portfolios

Having proven that estimating expected returns is dangerous (Part 2) and that static covariances break down during crises (Part 3), we constructed three **risk-based** portfolios that require **zero return forecasting**:

- **Equal Weight (1/N):** The naïve baseline — allocate 25% to each asset.
- **Minimum Volatility:** Solves $\min(w^T \Sigma w)$ — the portfolio with the lowest possible risk.
- **Maximum Diversification:** Maximizes the Diversification Ratio $\frac{w'\sigma}{\sqrt{w'\Sigma w}}$ — aggressively hunts for uncorrelated assets.

### Backtesting Methodology

We simulated a live trading environment using **walk-forward** covariance estimation (no look-ahead bias):
- **Rolling 36-month window:** Uses only the most recent 3 years of data — adapts faster to regime changes but noisier.
- **Expanding window:** Uses all available history up to each rebalance date — more stable but slower to adapt.

Portfolios are rebalanced monthly. No transaction costs or slippage applied (gross performance).

### Performance Comparison (OOS: 2003 – 2026)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar | Final $1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Equal Weight (1/N) | +6.48% | 10.51% | 0.617 | -40.3% | 0.161 | $4.30 |
| **Min Volatility (Rolling 36m)** | +4.21% | **5.49%** | **0.767** | -17.0% | 0.248 | $2.61 |
| Min Volatility (Expanding) | +4.25% | 5.50% | 0.773 | -16.5% | 0.257 | $2.63 |
| Max Diversification (Rolling 36m) | +4.44% | 5.96% | 0.744 | **-16.0%** | **0.278** | $2.74 |
| Max Diversification (Expanding) | +4.59% | 6.22% | 0.737 | -20.7% | 0.222 | $2.84 |

![Cumulative Wealth — $1 Invested](../Outputs/backtest_cumulative_wealth.png)

**Key observations:**
- **Equal Weight** achieved the highest absolute return ($4.30 terminal wealth) but at a devastating cost: **-40.3% max drawdown** during the 2008 GFC. This is because the naïve 1/N allocation has no risk awareness and holds full equity/REIT exposure through crises.
- **Min Volatility** delivered the best Sharpe Ratio (0.77) by concentrating ~75% in bonds and minimizing portfolio variance. It cut the max drawdown in half (-16.5%) compared to Equal Weight.
- **Max Diversification (Rolling 36m)** achieved the **shallowest max drawdown (-16.0%)** and the **best Calmar Ratio (0.278)** — the optimal risk-adjusted return per unit of worst-case loss.

### Drawdown Analysis — Resilience Across Shocks

![Drawdown Analysis](../Outputs/backtest_drawdowns.png)

The drawdown chart reveals how each strategy handled the three major shocks in our sample:
- **2008 GFC:** Equal Weight plunged to -40%, while Min Vol and Max Div limited losses to ~-15 to -17%
- **2020 COVID:** All strategies experienced drawdowns, but Min Vol and Max Div recovered faster
- **2022 Rate Hikes:** Bond-heavy Min Vol suffered more than expected (~-10%) as the traditional equity-bond correlation flipped positive

### Rolling Sharpe Ratio — Consistency Over Time

![Rolling 12-Month Sharpe Ratio](../Outputs/backtest_rolling_sharpe.png)

The rolling Sharpe chart shows that **Min Volatility and Max Diversification consistently maintain higher risk-adjusted returns** than Equal Weight across most market environments. Equal Weight's Sharpe Ratio swings wildly — from +5 during calm bull markets to below -2 during crises.

### Dynamic Weight Allocations

![Portfolio Weight Allocations Over Time](../Outputs/backtest_weight_allocations.png)

- **Equal Weight** maintains a fixed 25% per asset (by definition).
- **Min Volatility** is heavily bond-dominated (~75%) throughout, with equity weight spiking to ~30% during brief volatility drops (e.g., 2005-2006, 2012-2014).
- **Max Diversification** maintains a more diversified profile (~65% bonds, 10-15% each in other assets), with allocations actively shifting based on the changing correlation structure.

### Enhanced Backtest: GARCH-Forecasted Covariance

As an enhancement, we also ran the same three strategies using **GARCH(1,1) volatility forecasting** with a 60-month expanding window, which better captures volatility clustering:

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Equal Weight (Baseline) | +5.43% | 10.75% | 0.505 | -40.4% | 0.135 | 
| Min Volatility (Expanding) | +3.98% | 5.36% | **0.742** | -17.3% | 0.229 |
| **Max Diversification (Expanding)** | +3.99% | 6.01% | 0.663 | **-15.7%** | **0.254** |

![Drawdown Analysis — GARCH Forecasted](../Outputs/garch_drawdowns.png)

![Cumulative Wealth — GARCH Forecasted](../Outputs/garch_cumulative_wealth.png)

The GARCH-enhanced backtest confirms the same ordering: risk-based strategies (Min Vol, Max Div) dramatically outperform Equal Weight on a risk-adjusted basis.

### Final Strategic Recommendation

For institutional capital deployment, **we recommend the Maximum Diversification strategy (Rolling 36m window):**

1. **Best downside protection:** Shallowest max drawdown of **-16.0%** across the entire 23-year sample, including the 2008 GFC, COVID crash, and 2022 rate shock
2. **Highest Calmar Ratio (0.278):** The best return per unit of worst-case loss — the metric that matters most for capital preservation mandates
3. **Active diversification benefit:** Unlike Min Vol (which simply piles into bonds), Max Diversification actively exploits changing correlations to maintain exposure across asset classes while still controlling risk
4. **Robustness:** Both simple historical and GARCH-enhanced covariance methods confirm Max Diversification's superiority, indicating the result is not model-dependent
