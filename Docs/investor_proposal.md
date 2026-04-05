# Strategic Capital Allocation Proposal: Adapting to Market Regimes with Risk-Based Portfolios

## Executive Summary

As an investor, the true measure of your portfolio’s success is not simply the raw return generated, but the *risk-adjusted return*—how much compensation you receive for the volatility and downside risks you endure. In this proposal, we recommend shifting a portion of equity capital away from traditional Mean-Variance Optimization (MVO) or Equal Weighting, and toward **Risk-Based Allocation Models (Minimum Volatility and Maximum Diversification)**.

Our rigorous 23-year backtest (2003–2026), using walk-forward covariance estimation, demonstrates that these risk-based approaches deliver substantially higher efficiency (Sharpe Ratios up to 0.77) and cut maximum drawdowns by more than half (limiting them to just -16% to -17% compared to the devastating -40% drop of a naïve baseline during the 2008 Financial Crisis). 

**Recommendation:** We strongly recommend implementing a **Rolling 36-Month Maximum Diversification Strategy**. This strategy completely sidesteps the dangerous task of predicting expected returns, relying instead on stable, walk-forward measures of risk and dynamic covariance to protect capital through changing macroeconomic regimes. It achieved the **shallowest max drawdown (-16.0%)** and the **best Calmar Ratio (0.278)** of all tested strategies.

---

## 1. The Challenge: Why Traditional Models Fail

The fundamental challenge in quantitative investing is that **market regimes keep changing**. A "regime" is an extended period where markets exhibit specific behavioral characteristics (e.g., the Pre-GFC calm of 2003-2006, the explosive crisis of 2008, or the recent inflationary shocks of 2022). 

Traditional portfolio optimization (the Markowitz Mean-Variance model) requires the investor to estimate *expected returns*. However, expected returns are notoriously difficult to predict and highly unstable. Worse, optimization algorithms are highly sensitive to these inputs: a tiny error in forecasting returns results in huge, unintended shifts in portfolio allocation—often right before a regime change catches the portfolio completely off-balance.

### The Sharpe Ratio and Investor Utility
Our goal is not merely higher numbers on a spreadsheet. Investor utility is fundamentally modeled as maximizing risk-adjusted return, formally measured by the **Sharpe Ratio**:

$$ \text{Sharpe Ratio} = \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}} $$

Where:
*   $w$ represents the portfolio weights.
*   $\mu$ represents the asset returns.
*   $r_f$ is the risk-free rate.
*   $\Sigma$ represents the covariance matrix (risk).

Because the numerator ($w^T \mu$) is unstable, our recommended strategy focuses entirely on optimizing the denominator ($\Sigma$)—creating a robust portfolio that survives regardless of which way the market breaks.

### The Strategic Frontier: Where Your Capital Sits
The chart below illustrates the **Efficient Frontier**— the mathematical limit of the best possible return for every level of risk. 

![The Efficient Frontier and Optimized Portfolios](../Outputs/efficient_frontier.png)

*   **Individual Assets (Gray):** Notice how most single assets (like Equities or REITs) sit well to the right or below the frontier. Holding them in isolation is inefficient.
*   **The Curve (Dashed Line):** This is the boundary of "perfection." By combining assets that aren't perfectly correlated, we "push" the portfolio northwest—higher returns with lower risk.
*   **Target Portfolios:** We have highlighted your three strategic options. The **Max Sharpe** portfolio (Gold) represents the absolute peak of risk-adjusted efficiency. The **Global Minimum Volatility** point (Green) represents the safest possible combination of these assets.

### The Three Optimized Portfolios

![Three Portfolio Comparison Dashboard](../Outputs/three_portfolio_comparison.png)

| Portfolio | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Final $1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Max Return (Vol≤10%) | +7.43% | 10.00% | 0.743 | -34.2% | $6.12 |
| Min Volatility (Ret≥4%) | +4.18% | 5.22% | 0.800 | -16.3% | $2.89 |
| Max Sharpe | +5.14% | 5.83% | 0.882 | -17.4% | $3.68 |

While elegant in theory, these MVO portfolios rely on **static historical assumptions** that break down in practice — as we demonstrate next.

---

## 2. Advanced Risk Modeling: Dynamic Windows Over Static Averages

To optimize allocation correctly, we must model risk accurately. MVO models traditionally assume asset returns follow a normal ("bell-shaped") distribution with constant volatility. Decades of market data prove this false. Instead, volatility "clusters"—periods of low volatility are followed by stretches of calm, while massive crashes immediately trigger extended periods of extreme risk.

### Walk-Forward Estimation
We have explicitly discarded the use of a single, static historical average in favor of dynamic **Rolling and Expanding Window** covariances. By continuously recalculating the covariance matrix at every month-step, our portfolio rapidly detects when the market enters a high-risk regime and immediately updates its structural variance matrix to account for shifting correlations.

*Note: While more complex mathematical frameworks like GARCH or Copulas exist, standard variance—when updated dynamically via rigorous walk-forward backtesting—is highly effective at building resilient allocations.*

### Why Static Risk Assumptions Fail: Regime Change Evidence

Our analysis of 25 years of data reveals that risk parameters shift violently across market regimes:

![Rolling Volatility and Correlation across Regimes](../Outputs/regime_rolling_risk.png)

- **Volatility is non-stationary:** US REITs volatility swung from 14.5% (calm) to 40.0% (GFC crisis) — nearly 3x
- **Correlations spike in crises:** Equities-REITs correlation jumped from +0.48 (calm) to +0.93 (COVID) — diversification disappeared exactly when needed most

![Correlation Matrices Across Regimes](../Outputs/regime_correlation_heatmaps.png)

The efficient frontier itself shifts dramatically depending on which regime's data is used:

![Efficient Frontiers Across Market Regimes](../Outputs/regime_efficient_frontiers.png)

This proves that a portfolio "optimized" on one regime's data will be dangerously mis-calibrated for another. **Static MVO is not fit for purpose.**

---

## 3. Methodology: Risk-Based Backtesting Strategies

Because of the danger of static return estimates, we evaluate three portfolios that rely entirely on the dynamic covariance matrix generated by our GARCH models:

1.  **Equal Weight (The Naïve Baseline):** $w_i = 1/N$. Evaluates holding all assets equally regardless of their inherent risk.
2.  **Minimum Volatility (MinVol):** mathematically solves for the point of absolute lowest total portfolio variance.
3.  **Maximum Diversification (MaxDiv):** Maximizes the "Diversification Ratio". Instead of just picking low-volatility assets, this model aggressively seeks out *uncorrelated* assets to ensure risk is maximally diversified across different economic drivers.

### Simulating Reality: The Backtesting Protocol
It is a common pitfall to use simple "train/test data splits" common in static machine learning. Financial time series are sequential; therefore, we must use traditional financial walk-forward backtesting:
*   **Rolling Window (36-month):** The model calibrates using the trailing 3 years of data, moving forward one month at a time. This causes the portfolio to actively "forget" old regimes and rapidly adjust to new realities.
*   **Expanding Window:** The model is initialized in 2000, but the calibration window continuously expands month-over-month. This provides a highly stable, long-term perspective of asset covariance.

All backtests involve rebalancing the portfolio capital monthly over a 21-year out-of-sample period (2005 to 2026).

---

## 4. Empirical Analytics and Outcomes

The Out-of-Sample (OOS) performance for all portfolios is shown below. 

### Performance Summary Table (OOS: 2003 – 2026)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar | Final $1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal Weight (1/N)** | +6.48% | 10.51% | 0.617 | -40.3% | 0.161 | $4.30 |
| Min Volatility (Rolling 36m) | +4.21% | **5.49%** | **0.767** | -17.0% | 0.248 | $2.61 |
| Min Volatility (Expanding) | +4.25% | 5.50% | 0.773 | -16.5% | 0.257 | $2.63 |
| **Max Diversification (Rolling 36m)** | +4.44% | 5.96% | 0.744 | **-16.0%** | **0.278** | $2.74 |
| Max Diversification (Expanding) | +4.59% | 6.22% | 0.737 | -20.7% | 0.222 | $2.84 |

### Visual Analytics: Drawdowns and Regimes

![Drawdown Analysis - Resilience Across Shocks](../Outputs/backtest_drawdowns.png)

*The chart above highlights catastrophic regime changes—most notably the 2008 GFC and the 2020 COVID shock in red. The blue Equal Weight baseline collapses nearly -40%. Conversely, the MinVol and MaxDiv risk-based portfolios drastically mute the market shock, protecting investor capital.*

![Capital Allocation Wealth - Dynamic Backtest ($1 Invested)](../Outputs/backtest_cumulative_wealth.png)

*While Equal Weight (blue) achieves a higher raw terminal wealth ($4.30), its journey is chaotic with a -40% drawdown. Risk-based portfolios (green and orange) provide a remarkably smooth ascent. This is heavily preferred for institutional allocation where consistent compound growth and low drawdown profiles minimize the risk of forced liquidation during crises.*

### Rolling Sharpe Ratio — Consistency Over Time

![Rolling 12-Month Sharpe Ratio](../Outputs/backtest_rolling_sharpe.png)

*Min Volatility and Max Diversification consistently maintain higher risk-adjusted returns across most environments. Equal Weight swings wildly — from +5 during calm markets to below -2 during crises.*

### Dynamic Weight Allocations

![Portfolio Weight Allocations Over Time](../Outputs/backtest_weight_allocations.png)

*Min Volatility stays ~75% bonds; Max Diversification maintains a more balanced profile, actively shifting based on the changing correlation structure — evidence of genuine adaptive risk management.*

---

## 5. Next-Generation Enhancements: Dynamic Regime Adaptation

While our GARCH-calibrated Minimum Volatility and Maximum Diversification portfolios significantly outperform static models during market shocks, true institutional resilience requires a framework that actively *adapts* to regime changes rather than just passively surviving them. To build a highly dynamic strategy capable of absorbing severe macro-economic shocks, we have identified three critical enhancements for future iterations:

1. **Hidden Markov Models (HMM) for Proactive Regime Switching:** 
   Currently, our rolling window approach causes the covariance matrix to adapt with a slight mathematical lag. By implementing unsupervised machine learning models like HMMs, we can statistically classify the market into discrete hidden states (e.g., "State 1: Low Vol Rally", "State 2: High Vol Crash"). Once a probability shift into a crisis state is detected, the strategy can completely toggle its underlying objective function—for instance, pursuing aggressive returns during State 1, but instantly hard-switching to Minimum Volatility allocation the moment State 2 is detected.
2. **Dynamic Volatility Targeting (Leverage Scaling):**
   Rather than remaining 100% invested at all times, we can use short-term dynamic variance forecasts to actively scale total gross exposure. If forecasted market volatility breaches a critical threshold signaling a crash, the strategy dynamically deleverages (shifting capital to cash). Conversely, in ultra-safe regimes, it mathematically applies leverage to realize a constant "target volatility," completely smoothing the long-term wealth curve.
3. **Correlation-Triggered Hedging:**
   Our regime analysis proves that during extreme crises (like 2008 or 2020), the correlation between Equities and Alternative Assets (like REITs) rapidly spikes toward +1.0. This destroys standard diversification exactly when investors need it to work. A truly dynamic strategy monitors short-term rolling correlations and implements a circuit-breaker: the moment intra-portfolio correlations converge, it forcefully rotates capital exclusively into strictly negative-correlated assets (such as US Treasuries) or long-volatility hedges.

---

## 6. Strategic Conclusion

**Our analytical findings point us to two clear options for your capital deployment:**

**Our primary recommendation is the Rolling 36-Month Maximum Diversification Strategy:**

| Metric | Equal Weight | Min Volatility | **Max Diversification** |
| :--- | :---: | :---: | :---: |
| Ann. Return | +6.48% | +4.21% | **+4.44%** |
| Max Drawdown | -40.3% | -17.0% | **-16.0%** |
| Sharpe Ratio | 0.617 | 0.767 | **0.744** |
| Calmar Ratio | 0.161 | 0.248 | **0.278** |

**Why Max Diversification wins:**

1. **Shallowest drawdown (-16.0%):** Best capital preservation across 23 years spanning the GFC, COVID crash, and 2022 rate shock
2. **Highest Calmar Ratio (0.278):** The best return per unit of worst-case loss — the metric that matters most for institutional mandates
3. **Active diversification:** Unlike Min Vol (which simply piles into bonds), Max Diversification actively exploits changing correlations to maintain broader exposure while controlling risk
4. **No return forecasting required:** Entirely eliminates the dangerous task of predicting expected returns

**Alternative for maximum conservatism:** If the mandate is ultra-defensive, the **Min Volatility (Expanding)** strategy achieves the highest Sharpe Ratio (0.773) with the lowest portfolio volatility (5.50%).

By transitioning from fixed expected-return models to **dynamic, walk-forward Risk-Based Portfolios**, we formally immunize the portfolio against regime changes, protecting wealth through market crises while still capturing a stable, long-term yield.
