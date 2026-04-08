# Stop Guessing Returns, Start Managing Risk: A PM's Guide to Real-World Allocation

**Author:** Chin Zheng Yang | **Student Number:** A0243253M

---

## Executive Overview
For decades, the foundation of institutional asset management has been built upon traditional Mean-Variance Optimization (MVO). However, this paper systematically deconstructs that theoretical framework to expose its severe mathematical and empirical vulnerabilities during real-world market shocks. By moving beyond the dangerous assumption of static market conditions, this report introduces and extensively backtests a dynamic, risk-based capital deployment strategy designed to actually survive market crises.

---

## 1. Data Parameters & Methodology

### Asset Universe
The study covers a diversified cross-section of the US economy:
* **Equities:** SPDR S&P 500 ETF Trust (SPY)
* **Fixed Income:** US 10-Year Treasury Bonds (GS10/FRED)
* **Real Estate:** US Real Estate Investment Trusts (VGSIX)
* **Commodities:** Bloomberg Commodity Index (^BCOM)

### Study Horizon
A comprehensive **26-year period** (January 2000 – March 2026), comprising 315 monthly observations to capture multiple distinct economic cycles, from the Dot-Com bust to the post-COVID tightening cycle.

### Data Acquisition & Preparation Methodology
To ensure mathematical rigor, all market data was programmatically sourced and standardized:

- **Equities, REITs, and Commodities:** Daily price history was extracted via the Yahoo Finance API. To accurately represent total portfolio wealth generation, the extraction engine utilized dividend- and split-adjusted closing prices. These series were then resampled to the final trading day of each month to compute simple monthly percentage total returns, dropping forward-looking NAs.
- **US 10-Year Treasury Bonds:** Rather than trusting an ETF proxy, the model sourced the raw "GS10" (10-Year Constant Maturity Yield) directly from the Federal Reserve Economic Data (FRED) database. Because static yields do not equal investable returns, the model executed a mathematically exact Total Return simulation. For every individual month over the 26-year sample, the engine dynamically calculated the bond's Macaulay and Modified Duration based on the prevailing yield curve. It then computed the true Monthly Total Return by combining the accrued income (Yield/12) with the capital gain or loss resulting from the yield change acting against that dynamically calculated duration. This prevents the severe mispricing errors caused by assuming a static duration during periods of extreme rate volatility.


---

## 2. The Baseline: Historical Mean-Variance Optimization (MVO)

To establish a proper quantitative baseline, this analysis first applies the traditional Markowitz MVO framework to the full 26-year dataset. By calculating the annualized historical expected returns ($\mu$) and the standard covariance matrix ($\Sigma$) across the four distinct asset classes, the model tests three specific investment mandates. These outcomes define the traditional "Efficient Frontier"—a theoretical curve plotting the maximum possible risk-adjusted return based entirely on historical averages:

![The Efficient Frontier and Optimized Portfolios](../Outputs/efficient_frontier.png)

### Decoding the Efficient Frontier

The visual representation of the Efficient Frontier plots thousands of simulated portfolio allocations to illustrate the fundamental mathematical tradeoff between expected return ($y$-axis) and annualized risk ($x$-axis).

Breaking down the mechanics of the chart reveals several critical portfolio construction principles:

* **The Suboptimal Cloud (Scatter Plot):** The thousands of dots, color-coded by Sharpe Ratio (from dark purple to bright yellow), represent random weight allocations across the four asset classes. The vast majority of these portfolios fall inside or below the curve, meaning they are structurally inefficient—they take on unnecessary volatility for a given level of return.
* **The Optimal Boundary (Red Line):** The solid red hyperbola is the Efficient Frontier itself. Portfolios that land exactly on this line are mathematically optimized. By definition, an investor cannot achieve a higher return without taking on more risk, and cannot lower risk without sacrificing return.
* **The Capital Market Line (Dashed Line):** The straight diagonal line represents the Capital Market Line (CML). The exact point where this line tangentially intersects the Efficient Frontier mathematically dictates the **Maximum Sharpe Ratio** portfolio (Orange Cross)—the absolute most efficient use of risk capital.
* **Visualizing the Mandates:** The chart explicitly maps the strategic constraints applied to the model. The **Max Return** portfolio (Red Star) climbs as high up the curve as possible before slamming into the vertical **10% volatility** ceiling. Conversely, the **Min Volatility** portfolio (Blue Diamond) rests at the absolute far-left edge of the curve, safely satisfying the horizontal **4% minimum return** floor.
* **The Power of Covariance:** The individual asset markers (White Circles) demonstrate exactly why diversification works. Notice that US Equities and US Commodities both carry roughly **15% volatility**, yet Equities historically delivered vastly higher returns. By blending these imperfectly correlated individual assets, the optimizer mathematically bends the frontier upward and to the left, constructing portfolios that offer superior risk-reward profiles compared to holding any single asset alone.

![Three Optimized Portfolios -- Summary Table](../Outputs/three_portfolio_table.png)


### The Three Optimized Portfolios

The optimization engine was mathematically constrained to solve for three distinct strategic goals, yielding the following allocations:

1.  **Maximize Return subject to Volatility $\le$ 10%:** Pushing risk to the absolute allowable limit, this aggressive allocation achieved an annualized return of 7.43%. To harvest this yield, the optimizer heavily favored growth assets, allocating 34% to US Equities and 27% to REITs, while utilizing a 39% Treasury allocation to act as a stabilizing anchor against the 10.0% volatility ceiling.
2.  **Minimize Volatility subject to Return $\ge$ 4%:** Representing the most highly defensive posture possible within the mandate, this model concentrated the vast majority of its capital into US Treasuries (76.6%). It supplemented this with marginal allocations to Equities (10%) and Commodities (13%), successfully satisfying the 4% return floor while compressing total portfolio risk to a mere 5.22%.
3.  **Maximize Sharpe Ratio:** Representing the theoretical "tangency portfolio," this allocation achieved the highest overall risk-adjusted performance with a Sharpe Ratio of 0.88. The model found this optimal mathematical balance by ignoring commodities and REITs entirely, holding a strict 70% in bonds and 25% in equities.


### Historical Performance of the Three Portfolios


![Three Portfolio Comparison Dashboard](../Outputs/three_portfolio_comparison.png)

### Key Performance Observations

Stress-testing these three static mandates across the 26-year historical sample reveals the stark reality of the risk-return tradeoff:

* **The Efficiency Anchor (Max Sharpe):** Delivering the optimal risk-adjusted profile, this allocation generated the highest Sharpe (0.88) and Calmar (0.30) ratios. By anchoring the portfolio with a heavy **70% allocation to US Treasuries**, it mathematically maximizes return per unit of total risk and provides the smoothest equity curve over the long term.
* **Maximum Downside Protection (Min Volatility):** Designed strictly for capital preservation, this defensive mandate successfully contained the maximum drawdown to just **-16.3%** across multiple market crises. However, its massive **77% concentration in bonds** severely stunted absolute return, perfectly illustrating the long-term compounding cost of absolute volatility suppression.
* **The Cost of Yield (Max Return):** Pushing against the 10% volatility ceiling generated the highest terminal wealth ($6.12 on a $1 initial investment). However, to achieve this, the optimizer forced a combined **61% allocation into Equities and REITs**. The cost of this aggressive positioning was severe: a devastating **-34.2% drawdown** during the 2008 Global Financial Crisis, highlighting the extreme tail risk of chasing absolute returns.

> **Structural Takeaway:** Because these traditional MVO models are constrained to remain long-only and fully invested, their survival through market shocks is entirely dictated by their fixed asset weights. The massive gap in drawdown depths during the 2008 GFC explicitly confirms that return maximization and absolute capital preservation are fundamentally competing mandates.

---

### Broader Comparison (Including Benchmarks)

For reference, we also include the naive Equal Weight (1/N) and Global Min Volatility portfolios as benchmarks:

![MVO Performance Summary Table](../Outputs/mvo_performance_table.png)

![MVO Portfolio Historical Performance Dashboard](../Outputs/mvo_performance_dashboard.png)



### Benchmark Analysis & Constraint Dynamics

To fully contextualize the performance of the optimized mandates, they must be measured against two standard baselines: the naive Equal Weight (1/N) allocation and the unconstrained Global Minimum Volatility portfolio.

* **The Danger of Naive Diversification (Equal Weight 1/N):** The default benchmark for many passive allocators is an even 25% split across the four asset classes. While this naive approach yielded a respectable **+6.59% annualized return** and a terminal wealth of **$4.89**, it possessed zero risk awareness. By forcing equal capital into highly volatile assets, the portfolio suffered a catastrophic **-40.4% maximum drawdown** and the lowest Calmar Ratio (0.16) of the cohort. This proves that simple capital distribution without covariance modeling results in severe tail-risk exposure.
* **The Non-Binding Return Constraint (Global Min Vol vs. Min Vol ≥ 4%):** A direct comparison between the unconstrained Global Minimum Volatility portfolio and the constrained "Min Volatility (Ret ≥ 4%)" mandate reveals a critical structural insight: they are mathematically identical. Both portfolios arrived at the exact same allocation (**76.6% Bonds, 10.5% Equities, 12.9% Commodities**) and the same risk/return profile (**5.22% Volatility, +4.18% Return**).

> **Structural Takeaway:** Because the unconstrained Global Minimum Volatility portfolio naturally achieves a 4.18% historical return, the ≥ 4% return constraint placed on the optimizer was entirely non-binding. This data point establishes the absolute floor of systemic risk for this specific asset universe: it is mathematically impossible to achieve a portfolio volatility lower than 5.22%, and doing so requires dedicating over three-quarters of all capital directly to US Treasuries.


---

## 3. The Equal Sharpe Ratio Experiment

A critical flaw in traditional MVO is that it relies blindly on historical expected returns. To test how sensitive our portfolios are to these returns, we simulated a scenario where **all assets possess the exact same theoretical Sharpe Ratio** (e.g., $SR = 0.5$). 

Because $SR = \frac{\mu}{\sigma}$, assuming Equal Sharpe Ratios mathematically forces the expected return of each asset to be strictly proportional to its volatility ($\mu_i = SR_{common} \cdot \sigma_i$). 

![Equal Sharpe Ratio -- Efficient Frontier Comparison](../Outputs/equal_sharpe_frontier.png)

### Decoding the Equal-Sharpe Frontier: The Collapse of the Model

Comparing the Equal Sharpe Ratio frontier (solid red) to the historical frontier (dashed grey) exposes exactly how deeply expected returns manipulate portfolio construction. Neutralizing the return signal triggers three massive structural shifts on the graph:

* **Linear Asset Realignment:** Notice the individual asset markers (white circles). In the historical chart, they were scattered chaotically based on past performance. Here, because every asset is forced to an identical Sharpe Ratio of 0.5, they align on a perfectly straight trajectory where return is strictly a mathematical function of volatility ($\mu = 0.5 \cdot \sigma$). The optimizer can no longer "cheat" by overweighting assets that historically outperformed their risk profile.
* **The Downward Frontier Shift:** At the upper-right edge of the curve, the new red frontier sits distinctly below the dashed historical line. Because growth assets like Equities and REITs are no longer gifted with outsized historical return assumptions, the absolute maximum yield the portfolio can harvest at a 10% volatility budget mathematically drops.
* **The Degenerate Tangency Portfolio:** This is the most critical visual difference. In the historical model, the Capital Market Line (CML) touched the frontier high up the curve, creating a distinct Max Sharpe portfolio heavily allocated to equities. Under equalized returns, the orange cross (Max Sharpe) collapses down and to the left, converging almost exactly with the blue diamond (Min Volatility).

> **The Mathematical Reality:** When return assumptions are equalized, every portfolio on the frontier achieves the exact same theoretical Sharpe Ratio. Consequently, the optimizer abandons return-chasing and defaults purely to covariance matrix management. The "optimal" risk-adjusted portfolio degenerates directly into the lowest-variance portfolio mathematically possible.




![Weight Allocation: Historical vs Equal Sharpe Ratio](../Outputs/equal_sharpe_weights.png)

![Comparison Table: Historical vs Equal Sharpe Ratio](../Outputs/equal_sharpe_comparison_table.png)

### Analysis Outcomes (All 3 Objectives)

Comparing the historical allocations against the Equal Sharpe Ratio ($SR = 0.5$) environment exposes precisely how return assumptions dictate portfolio construction. When the return signal is neutralized, the optimizer behaves radically differently:

1. **Maximize Return (Target: $\sigma \le 10\%$) — Weights Shift Dramatically:** Under historical returns, this portfolio tilted 34% Equities + 27% REITs to exploit their high expected returns. Under Equal SR, the optimizer can no longer favor high-return assets (since higher return = proportionally higher risk). Instead, it **redistributes weights toward Commodities (35%) and REITs (26%)** to construct the combination that reaches exactly 10% volatility through the covariance structure. The achievable return drops from 7.43% to 7.21% because the adjusted expected returns are lower.
2. **Minimize Volatility (Target: Return $\ge 4\%$) — Weights May Change Depending on SR Level:** Under Equal SR = 0.5, the minimum-vol portfolio achieving $\ge 4\%$ return shifts to 76% Bonds + 11% Equities + 13% Commodities—similar to historical because bond dominance is driven by the covariance matrix. However, for very low SR values (e.g., $SR = 0.3$), the 4% return floor becomes harder to satisfy, forcing the optimizer to take on more volatile assets. This demonstrates that **the return constraint interacts with the SR level**.
3. **Maximize Sharpe Ratio — The Max Sharpe Portfolio Becomes Degenerate:** Under Equal SR, **every** portfolio on the efficient frontier achieves exactly the same Sharpe Ratio ($SR_{common}$). There is no unique optimal point—the optimizer collapses to the **minimum-variance portfolio** (16% Equities, 63% Bonds, 2% REITs, 19% Commodities), since it achieves the same SR as any other portfolio but with the lowest risk. This is fundamentally different from the historical Max Sharpe portfolio (25% Equities, 70% Bonds).


### Does the SR Level Matter?

We tested three different SR values (0.3, 0.5, 0.7) and found:

![SR Sensitivity -- Do Weights Change Across SR Levels?](../Outputs/equal_sharpe_sensitivity.png)

* **Weights are nearly identical** across SR = 0.3, 0.5, and 0.7 for Objectives 1 and 3—only the *scale* of return changes (proportionally), not the allocation.
* **Objective 2 is the exception:** When $SR_{common}$ is low (0.3), the 4% return floor forces the optimizer into higher-volatility assets. When $SR_{common}$ is high (0.7), the return floor is easily met with purely defensive assets.

---
### Conclusion: Do Optimization Results Depend on the Sharpe Ratio Assumption?

**Yes—optimization results depend *dangerously heavily* on the expected return (Sharpe Ratio) assumption.** This is demonstrated by:

1. **Weight Sensitivity:** All three objectives produce significantly different allocations between historical vs. equal-SR scenarios, proving that return assumptions directly control portfolio construction.
2. **Estimation Error Amplification (Michaud, 1989):** MVO mechanically overweights whichever asset has the highest estimated return—amplifying any estimation error into highly concentrated bets.
3. **The Only Stable Element is the Covariance Matrix ($\Sigma$):** When return signals are neutralized (Equal SR), the optimizer reverts to a pure risk-minimization engine, driven entirely by correlations and volatilities. 

> **Structural Takeaway:** This stability under the Equal SR assumption is exactly why **risk-based** portfolio construction (explored in Part 4) is fundamentally more robust than traditional return-based MVO in real-world environments.

### Absolute SR Level Sensitivity: Does the Baseline Matter?

To determine if the specific level of the equalized Sharpe Ratio impacts portfolio construction, the model stress-tested three distinct market environments: a low-return regime ($SR = 0.3$), a baseline regime ($SR = 0.5$), and a high-return regime ($SR = 0.7$).

The results expose exactly how absolute return constraints interact with market environments:

* **Structural Immunity (Relative Mandates):** For the Max Return (Vol $\le 10\%$) and Max Sharpe mandates, the absolute level of the assumed Sharpe Ratio has zero impact on the final asset weights. Whether the SR is 0.3, 0.5, or 0.7, the allocations remain completely identical. Because these specific mandates do not require hitting a fixed absolute return floor, the optimizer perfectly scales the theoretical return up or down proportionally without altering the underlying covariance-driven allocation.
* **Constraint Friction (The Min Volatility Exception):** The defensive Min Volatility (Ret $\ge 4\%$) mandate breaks this symmetry entirely. When the market-wide SR drops to a pessimistic 0.3, the fixed 4% return floor acts as a dangerous binding constraint. To hit that absolute yield target in a low-return environment, the optimizer is forced to slash its safe US Treasury allocation from 77% down to 35%, aggressively loading up on riskier assets like REITs (21%) and Commodities (31%) just to clear the hurdle. Conversely, at an SR of 0.7, the 4% floor is easily satisfied, allowing capital to flood safely back into Treasuries (77%).

### Strategic Conclusion: The Fragility of Return Forecasting

The empirical data gathered across these sensitivity tests definitively proves that traditional Mean-Variance Optimization is dangerously dependent on expected return assumptions. This fragility is characterized by three core structural flaws:

* **Illusion of Risk Management:** The massive divergence in asset weights between the historical baseline and the equal-SR simulation proves unequivocally that return assumptions—not actual risk architecture—dictate MVO portfolio construction.
* **Estimation Error Maximization (Michaud, 1989):** The MVO engine fundamentally operates as an error-amplifier. It mechanically overweights whichever asset possesses the highest estimated return, transforming standard forecasting inaccuracies into highly concentrated, fragile bets.
* **The Stability of Covariance:** The single most stable element in the entire optimization matrix is the covariance structure ($\Sigma$). When return signals are deliberately neutralized, the optimizer automatically reverts to a pure risk-minimization engine, driven entirely by asset correlations and volatilities.

> **Structural Takeaway:** This underlying mechanical reality is exactly why abandoning return forecasting in favor of dynamic, risk-based portfolio construction provides a fundamentally superior, more robust architecture for institutional capital.


---
# 4. Market Regime Changes in Risk Assumptions

### What Are the "Risk Assumptions"?

In traditional Mean-Variance Optimization, the entire concept of "risk" is reduced to a single mathematical input: the **covariance matrix** ($\Sigma$). Relying on this matrix forces the model to make two highly dangerous, embedded assumptions:

* **Constant Volatilities ($\sigma_i$):** The assumption that the annualized risk of an individual asset (the diagonal elements of $\Sigma$) remains structurally stable over time.
* **Constant Correlations ($\rho_{ij}$):** The assumption that the relationships between assets (the off-diagonal elements of $\Sigma$) never fundamentally change—for example, trusting that bonds will always act as a reliable diversifier for equities.

Standard MVO models calculate this covariance matrix exactly once using the full 26-year dataset and treat it as absolute, unchanging truth. The critical question for capital allocators is: *do these risk parameters actually remain constant in the real world?*

---
### The Reality of Market Regimes

> **Structural Takeaway:** The empirical data proves that the single, static $\Sigma$ assumption is fundamentally broken. Risk parameters do not stay constant; both asset volatilities and cross-asset correlations fracture and shift violently as the market transitions through different macroeconomic regimes.

### Historical Baseline: Rule-Based Regime Identification

To quantify this instability, the data was initially segmented using well-known historical market boundaries. While this deterministic approach inherently relies on hindsight (ex-post knowledge of precisely when a crisis began and ended), it establishes a clear baseline for observing how risk mutates across distinct economic environments.

**Defined Regimes:**

| Regime | Period | Months |
| :--- | :--- | :---: |
| **Dot-Com Bust** | 2000–2002 | 36 |
| **Pre-GFC Calm** | 2003–2006 | 48 |
| **GFC Crisis** | 2007–2009 | 36 |
| **Post-GFC QE** | 2010–2019 | 120 |
| **COVID Shock** | 2020 | 12 |
| **Inflation/Hikes** | 2021–2023 | 36 |

#### Evidence of Regime Changes in Volatility

Calculating rolling 24-month annualized volatility across the asset universe reveals severe structural instability. If the foundational assumptions of traditional MVO held true, these risk metrics would plot as relatively flat horizontal lines across the 26-year horizon. Instead, the empirical data exhibits violent, regime-dependent spikes:


![Rolling Volatility and Correlation across Regimes](../Outputs/regime_rolling_risk.png)

**Empirical Observations of Volatility Shocks:**

* **Core Equity Instability:** US Equities experienced a massive structural risk expansion, with annualized volatility surging from a calm **8.5%** during the mid-2000s credit expansion to **19.8%** at the depth of the Global Liquidity Crisis—a **2.3x absolute increase** in baseline portfolio risk.
* **Real Estate Contagion:** US REITs demonstrated the most extreme non-stationarity of the cohort. Baseline volatility expanded from **14.5%** to a staggering **40.0%** during the '08 meltdown. This represents nearly a **300% risk expansion** that static, full-sample covariance models are mathematically blind to.
* **The "Safe Haven" Illusion:** Even US 10-Year Treasuries, traditionally allocated as the ultimate risk-free anchor, proved highly vulnerable to regime shifts. Treasury volatility spiked from its **6.0%** baseline up to **8.1%** during the GFC, and experienced a secondary structural surge to **7.4%** during the recent Post-Stimulus Tightening Cycle as inflation fundamentally disrupted bond mechanics.
* **COVID Shock (2020):** The brief but violent 2020 pandemic shock forced an immediate, synchronized regime break across real assets. REIT volatility instantly re-expanded to **26.5%**, while Commodities spiked to **20.9%**, confirming that physical asset risk compresses and explodes entirely based on the prevailing macro environment.

> **Structural Takeaway:** When an optimizer relies on a 26-year average volatility for an asset like REITs (~18%), it structurally overallocates capital. When the market inevitably transitions into a crisis regime and that asset's volatility suddenly spikes to 40%, the portfolio is left carrying more than double the targeted risk budget, leading to catastrophic drawdowns.


![Volatility Comparison Across Regimes](../Outputs/regime_volatility_comparison.png)

#### Evidence of Correlation Instability during Regimes Changes

The second half of the risk equation—cross-asset correlation—exhibits the exact same structural instability as absolute volatility. Analyzing the correlation matrices across distinct market environments proves that asset relationships are not fixed; they fracture and realign violently during market shocks:

![Correlation Matrices Across Regimes](../Outputs/regime_correlation_heatmaps.png)

**Critical Observations:**

* **Equities vs. REITs:** During the low-volatility expansion of the mid-2000s, the correlation between Equities and REITs sat at a mathematically useful **+0.48**. However, during the Global Liquidity Crisis, this correlation surged to **+0.81**, and during the 2020 exogenous shock, it essentially reached unit correlation at a staggering **+0.93**. This confirms that during severe liquidity drawdowns, the diversification benefit of real estate completely disappears.
* **Equities vs. Treasuries:** The equity-bond relationship—the bedrock of the traditional 60/40 portfolio—is highly regime-dependent rather than structurally guaranteed. While bonds offered a neutral, zero-correlation anchor (**+0.01**) during calm periods and acted as a massive deflationary hedge during the COVID crash (**-0.53**), this relationship actively works against the portfolio in specific regimes. As noted in the broader analysis, this correlation flipped positive during the 2021–2023 inflationary tightening cycle, rendering the bond hedge entirely ineffective.
* **Equities vs. Commodities:** Commodities are historically marketed as uncorrelated alternative assets, yet the empirical data proves they couple with equities during systemic stress. The equity-commodity correlation drifted from an ideal **+0.09** in calm markets to **+0.51** during the GFC, ultimately peaking at **+0.72** during the pandemic shock. Exactly when equity portfolios require an uncorrelated return stream, commodities violently couple with broader risk assets.

> **Structural Takeaway:** When an optimizer relies on a "Full Sample" correlation matrix, it averages these extreme regime shifts into a dangerously misleading middle ground. The data clearly dictates that risk assets converge toward a correlation of 1.0 precisely when diversification is needed most, virtually guaranteeing that static MVO portfolios will suffer catastrophic drawdowns during contagion events.

#### Structural Impact: The Shifting Efficient Frontier

#### Structural Impact: The Shifting Efficient Frontier

Re-running all three optimization objectives using each regime's own covariance matrix proves that the theoretical Efficient Frontier does not remain stationary; it warps, shifts, and collapses entirely based on the underlying macro environment:

![Regime-Dependent Efficient Frontiers -- How the Optimal Boundary Shifts](../Outputs/regime_efficient_frontiers.png)

* **The Pre-GFC Calm Frontier (Green):** During the 2003–2006 credit expansion, the frontier shifts radically to the upper-left. In this regime, the market offers an exceptional risk-reward profile, providing high expected returns at historically low volatility levels.
* **The GFC Crisis Frontier (Red):** Conversely, during the 2007–2009 systemic meltdown, the frontier completely collapses downward and to the right. The traditional risk-reward tradeoff breaks down; in this regime, even the most efficient portfolio mathematically possible struggles to clear a **~6% return** and is forced to absorb massive baseline risk (starting at **~8% volatility**). Pushing for higher returns in this environment results in an inverted curve, effectively guaranteeing capital destruction.
* **The Full Sample Frontier (Gray Dashed):** This falls somewhere in between—a compromise that represents *neither* regime well. This visually exposes the fatal flaw of traditional MVO: the historical average is merely a mathematical middle ground. It constructs a portfolio optimized for a blended, "average" environment that never actually exists, rendering capital highly inefficient during expansions and completely unprotected during a liquidity crisis.

---

### Structural Instability in Capital Allocation

Translating the shifting Efficient Frontier into actual portfolio mechanics reveals extreme instability in the optimal capital stack. When optimization mandates are isolated by regime, the resulting asset weights flip entirely:

| Asset Class | Pre-GFC Weight | GFC Crisis Weight | Variance |
| :--- | :---: | :---: | :---: |
| **US Equities** | High Allocation | Severe Cut | Extreme |
| **US Treasuries** | Moderate | Maximum Flight | High |
| **REITs** | Growth Driver | Eliminated | Extreme |
| **Commodities** | Diversifier | Risk Source | Moderate |

> **Structural Takeaway:** The "optimal" portfolio in a calm regime is often the most dangerous portfolio in a crisis regime. Because MVO lacks a dynamic mechanism to detect these regime shifts, a portfolio built on long-term historical averages will inevitably be "fighting the last war" with the wrong defensive posture.


![Optimal Weights Across Different Regimes](../Outputs/regime_optimal_weights.png)

**Specific weight shifts:**

| Objective | Calm Period | GFC Crisis | Full Sample |
| :--- | :--- | :--- | :--- |
| **Max Return (Vol<=10%)** | 21% Eq, 60% REITs | 0% Eq, **100% Bonds** | 34% Eq, 39% Bond, 27% REITs |
| **Min Vol (Ret>=4%)** | 29% Eq, 62% Bond | 5% Eq, **82% Bond** | 11% Eq, 77% Bond |
| **Max Sharpe** | 39% Eq, 39% REITs | **99% Bonds** | 25% Eq, 70% Bond |

**Specific Weight Shifts:**

| Objective | Calm Period (2003–2006) | GFC Crisis (2007–2009) | Full Sample (26-Year) |
| :--- | :--- | :--- | :--- |
| **Max Return ($\sigma \le 10\%$)** | 21% Eq, 60% REITs | 0% Eq, **100% Bonds** | 34% Eq, 39% Bond, 27% REITs |
| **Min Vol (Ret $\ge 4\%$)** | 29% Eq, 62% Bond | 5% Eq, **82% Bond** | 11% Eq, 77% Bond |
| **Max Sharpe** | 39% Eq, 39% REITs | **99% Bonds** | 25% Eq, 70% Bond |

---

### How the Model Actually Behaves

Looking at how the model shifts its weights shows how drastically it reacts to market changes. During the 2008 financial crisis, the optimization engine realizes its growth assumptions are broken and completely bails out of risky assets. For the top-performing portfolios, it moves **almost 100% of the money into US Treasuries**, because bonds were the only safe place to hide when the rest of the market crashed.

On the flip side, during the calm pre-crisis boom, the model does the exact opposite. It aggressively pulls money out of safe bonds to chase higher returns. The "best-balanced" portfolio swings heavily into **Equities (39%) and REITs (39%)** to squeeze out as much profit as possible while the market feels safe.

> **Structural Takeaway:** Seeing these two extremes exposes the fatal flaw of using the standard 26-year average. The "Full Sample" baseline isn't a smart, all-weather strategy; it is just a blind average of a massive boom and a massive bust. If capital is allocated based on that simple average, the portfolio is guaranteed to be **too risky when a crisis hits** and **too conservative when the market is booming.**

![Per-Regime Optimization Results](../Outputs/regime_summary_table.png)

### Data-Driven Regime Identification (Hidden Markov Model)

While the rule-based approach establishes a historical baseline, it suffers from a fatal quantitative flaw: **hindsight bias**. We hard-coded dates like "GFC Crisis: 2007–2009" because *we already know* when the crisis occurred. An actual investor in 2007 would not have known when the crisis would start or end.

To eliminate this human bias, the analysis deployed a completely unsupervised **2-State Gaussian Hidden Markov Model (HMM)** trained strictly on the raw, multivariate monthly asset returns. By utilizing a full covariance structure (`covariance_type="full"`), the algorithm was programmed to hunt for not just isolated volatility spikes, but the exact moments when cross-asset correlations structurally break down.

#### How the HMM Works

Rather than relying on human-fed dates, the HMM mathematically evaluates the 4-dimensional return vector of every single month. Using established statistical engines, it learns the true, underlying parameters of the market and probabilistically assigns each month to one of two hidden states:

* **State 0 — Calm / Low Volatility:** Without any external prompting, the model independently determined that **~94%** of the historical sample (295 of 315 months) operated within a structurally stable, low-variance environment.
* **State 1 — Crisis / High Volatility:** The algorithm isolated the remaining **~6%** of the timeline (20 months) as a distinct, high-variance state characterized by extreme market stress.

The model uses the **Viterbi algorithm** to determine the most likely state sequence, and the **Baum-Welch (EM) algorithm** to learn the emission parameters (mean return vector and full covariance matrix) for each state.

#### HMM-Discovered Regime Parameters (Annualized)

When the Hidden Markov Model partitions the historical data based purely on statistical variance, the resulting state parameters quantify exactly how violently the market fractures during a regime shift.

| Parameter | Calm State (State 0) | Crisis State (State 1) |
| :--- | :---: | :---: |
| **Months Identified** | 295 | 20 |
| **Equities Return** | +11.9% | -31.0% |
| **Equities Volatility** | 13.4% | **27.8%** (2.1x) |
| **Bond Return** | +3.7% | +4.9% |
| **Bond Volatility** | 5.9% | **14.2%** (2.4x) |
| **REITs Return** | +15.7% | **-43.9%** |
| **REITs Volatility** | 15.6% | **48.7%** (3.1x) |
| **Commodities Return** | +6.1% | **-38.4%** |
| **Eq-Bond Correlation** | -0.08 | **+0.12** |
| **Eq-REITs Correlation** | +0.55 | **+0.83** |
| **Eq-Commod Correlation** | +0.22 | **+0.74** |

---

### Empirical Observations of State 1 (The Crisis Regime)

* **Severe Volatility Multipliers:** The model mathematically confirms that risk does not scale linearly during a crisis. When the market transitions from State 0 (Calm) to State 1 (Crisis), baseline equity volatility more than doubles (from 13.4% to 27.8%). Even more devastating, REIT volatility explodes by a factor of 3.1x, surging to an unmanageable **48.7%**. Even US Treasuries, the presumed safe haven, see their volatility compound by 2.4x.
* **Correlation Spikes:** The HMM’s full covariance analysis explicitly proves the diversification breakdown thesis. In State 0, Equities and REITs share a moderate +0.55 correlation. In State 1, this relationship violently tightens to **+0.83**. Similarly, the Equity-Commodity correlation surges from +0.22 to **+0.74**. The data dictates that during true systemic shocks, all real assets collapse together.
* **The Bond Flip:** Notably, the Equity-Bond correlation flips from a slightly negative diversifier (-0.08) in State 0 to a positive correlation (+0.12) in State 1, warning allocators that traditional 60/40 hedging mechanics degrade precisely when portfolio stress peaks.

#### Validation: HMM Aligns with Known Crises

The ultimate test of the unsupervised model is whether its mathematically derived "State 1" aligns with actual historical events.

Without being fed any external dates, news headlines, or macroeconomic labels, the algorithm independently identified the exact periods of structural market failure. The red vertical bands in the overlay chart represent the months the HMM probabilistically flagged as "Crisis" states. It perfectly isolated the core of the 2008 Global Liquidity Crisis and the immediate exogenous shock of the 2020 pandemic. This blind validation proves that the "regime shift" phenomenon is not a narrative construct created by hindsight bias; it is a measurable, statistical property embedded directly into the asset return data.

![HMM Regimes Overlay](../Outputs/hmm_regimes_overlay.png)
---
#### HMM Per-Regime Optimization Results

To definitively prove how these mathematical states impact capital allocation, the optimization mandates were re-run using the distinct mean and covariance structures independently generated by the HMM.

| Regime | Objective | Ann. Return | Ann. Vol | Sharpe | Weights (Eq / Bond / REIT / Comm) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Calm (HMM)** | Max Ret (Vol<=10%) | +11.6% | 10.0% | **1.16** | 26% / 18% / 46% / 10% |
| **Calm (HMM)** | Min Vol (Ret>=4%) | +5.1% | 4.9% | 1.04 | 13% / 74% / 0% / 13% |
| **Calm (HMM)** | Max Sharpe | +7.8% | 6.3% | **1.25** | 21% / 50% / 18% / 11% |
| **Crisis (HMM)** | Max Ret (Vol<=10%) | -4.7% | 12.2% | -0.39 | 0% / 78% / 0% / 22% |
| **Crisis (HMM)** | Min Vol (Ret>=4%) | +4.0% | 13.8% | 0.29 | 0% / 98% / 0% / 2% |
| **Crisis (HMM)** | Max Sharpe | +4.9% | 14.2% | 0.35 | **0% / 100% / 0% / 0%** |

---

### Key Findings from the HMM Analysis

* **The Calm State Advantage (State 0):** When operating in the 94% of the time defined as "Calm," the optimizer constructs beautifully diversified portfolios. It is able to achieve massive risk-adjusted returns (Sharpe > 1.0) by comfortably allocating across Equities, Bonds, REITs, and Commodities.
* **The Crisis State Collapse (State 1):** During the Crisis state, the optimizer is forced into near-100% bonds—the only asset maintaining a positive Sharpe ratio. Notably, under the Max Return mandate, the optimizer mathematically cannot achieve a positive return while respecting the 10% volatility constraint (yielding -4.7% at 12.2% vol). This explicitly proves that during true market stress, arbitrary risk constraints become totally infeasible.

> **Structural Takeaway:** The HMM Crisis correlation structure (Eq-REITs = +0.83, Eq-Commodities = +0.74) mathematically confirms the "diversification breakdown" thesis: growth assets converge toward unit correlation precisely when diversification is needed most, destroying traditional MVO frameworks from the inside out.

### Conclusion: Regime Changes Invalidate Static MVO

To answer the question as stated in the assignment requirement: **Yes, regime changes are clearly present in the historical data, and they have a devastating impact on portfolio optimization.** The convergence of both deterministic (rule-based) and unsupervised (HMM) analysis definitively invalidates traditional, static Mean-Variance Optimization for institutional capital deployment.

1. **Volatility is Non-Stationary:** Asset volatilities routinely compound by 2x to 3x during regime shifts, guaranteeing that portfolios optimized on "calm" data will carry vastly more real-world risk than their models predict.
2. **Correlations Spike During Crises:** Growth assets (Equities, REITs, Commodities) become highly correlated exactly when diversification is needed most—destroying the diversification benefit that traditional MVO relies on.
3. **Optimal Weights are Regime-Dependent:** The same optimization objective produces entirely different portfolios depending on which regime's data is used. During the GFC crisis, for example, all three objectives converged toward near-100% bonds.
4. **The Full Sample MVO is a False Compromise:** Constructing an allocation based on a 26-year "Full Sample" average mathematically guarantees underperformance during expansions and catastrophic exposure during crises.
5. **Data-Driven Validation (HMM):** A purely statistical model, given no date information, independently recovers the same crisis periods and confirms the correlation breakdown—proving the regime-switching phenomenon is a robust statistical property of the data, not an artifact of human date selection.

Because predicting returns is structurally flawed (Part 2) and relying on static risk assumptions guarantees failure during a crisis (Part 3), capital allocators must transition to dynamic, risk-based portfolio construction (Part 4) that adapts to changing market conditions rather than assuming a static world.

---

## 5. Backtesting Study: Dynamic Risk-Based Capital Allocation

Because historical return forecasting amplifies error (Part 2) and static risk assumptions collapse during systemic shocks (Part 3), a robust institutional framework must abandon yield prediction entirely.

To construct a truly resilient capital stack, the analysis evaluated three purely risk-based allocation models that require strictly zero return forecasting:

* **Equal Weight (1/N):** A naive distribution allocating an inflexible 25% to each asset class. It acts as the benchmark for a portfolio with zero risk awareness.
* **Minimum Volatility:** Solving mathematically for the lowest possible structural risk ($\min(w^T \Sigma w)$)—the portfolio with the absolute lowest mathematical variance.
* **Maximum Diversification:** Maximizes the Diversification Ratio ($\frac{w^T\sigma}{\sqrt{w^T\Sigma w}}$). This forces the model to aggressively hunt for uncorrelated asset streams, maximizing true portfolio breadth.

---

### Sample Covariance Backtesting

To see how these strategies would actually perform with real money, we ran a rigorous backtest. Crucially, we didn't let the model cheat by looking into the future. We rebalanced the capital every single month (gross performance, no transaction costs or slippage applied), completely blind to what was coming next, using two different memory settings:

* **Rolling 36-Month Window:** Here, the model only looks at the last three years of market data. Because it drops old data quickly, it is highly adaptable. When the market violently shifts from a calm period into a crisis, this model reacts almost immediately.
* **Expanding Window:** Here, the model looks at all the historical data up to that specific month. While this makes the portfolio very stable during long, calm bull markets, it has a fatal flaw: it is dangerously slow to react when a sudden crisis hits.


#### Performance Comparison (OOS: 2003 -- 2026)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar | Final $1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Equal Weight (1/N) | +6.45% | 10.50% | 0.614 | -40.4% | 0.160 | $4.28 |
| **Min Volatility (Rolling 36m)** | +4.16% | **5.48%** | **0.758** | -17.0% | 0.244 | $2.58 |
| Min Volatility (Expanding) | +4.13% | 5.47% | 0.756 | -16.9% | 0.244 | $2.56 |
| Max Diversification (Rolling 36m) | +4.40% | 5.95% | 0.739 | **-16.1%** | **0.272** | $2.72 |
| Max Diversification (Expanding) | +4.50% | 6.14% | 0.732 | -19.8% | 0.227 | $2.78 |

![Cumulative Wealth -- $1 Invested](../Outputs/backtest_cumulative_wealth.png)

### Key Observations

* **Equal Weight (1/N):** If you just look at the final dollar amount, the naive Equal Weight portfolio looks like the winner, turning $1 into $4.28. But look at the cost. Because this strategy has zero risk awareness, it blindly held onto its risky growth assets all the way through the 2008 financial crisis, forcing the investor to suffer a devastating **-40.4% drop**. It generates absolute return, but it completely fails at capital preservation.
* **Min Volatility:** By actively hunting for the lowest-risk combination of assets, this strategy delivered the highest overall Sharpe Ratio (0.758). It did this exactly how you would expect: by hiding heavily in bonds (about 75% of the portfolio). While it gave up some upside, it successfully cut the maximum drawdown in half compared to the baseline, only dropping **-17.0%** during the worst market shocks.
* **Max Diversification (Rolling 36m):** This is the standout strategy of the entire study. By using a fast-adapting, 36-month rolling window to aggressively hunt for true diversification, it achieved the absolute shallowest drawdown of the group (just **-16.1%**). Even more importantly, it delivered the best Calmar Ratio (0.272). This means it generated the highest return possible for every unit of worst-case risk it took on. For an institutional allocator focused on protecting wealth while still capturing growth, this is the holy grail.

---

### Drawdown Analysis: Resilience Across Shocks

The ultimate test of a portfolio isn't how much money it makes when the sun is shining; it's how much capital it protects when the market breaks. Looking at the drawdown history across the three major macroeconomic shocks of the last two decades proves exactly why risk management is non-negotiable.

![Drawdown Analysis](../Outputs/backtest_drawdowns.png)

#### How the Strategies Handled the Crises

* **The 2008 Meltdown (The Liquidity Test):** The Global Financial Crisis was a bloodbath for naive portfolios. The Equal Weight strategy, carrying its static 50% allocation to growth (Equities and REITs), was dragged down to a catastrophic **-40.4% loss**. Meanwhile, the risk-aware strategies (Min Volatility and Max Diversification) successfully deployed their defensive mechanics, capping their worst-case losses between -15% and -17%. They survived the crisis with their capital base largely intact.
* **The 2020 COVID Flash Crash (The Exogenous Shock):** While every portfolio took a hit during the immediate panic of the pandemic lockdown, the recovery paths were vastly different. Because Min Volatility and Max Diversification are built to constantly monitor shifting correlations, they were able to weather the storm and bounce back significantly faster than the rigid Equal Weight baseline.
* **The 2022 Rate Hikes (The Bond Failure):** This is where the story gets really interesting. During the aggressive central bank tightening of 2022, bonds stopped acting like safe havens because inflation fundamentally broke their pricing. The ultra-defensive Min Volatility strategy, which relies on hoarding bonds, suffered a painful **~10% drawdown** because the traditional equity-bond hedge flipped positive (they both went down together). 

> **Structural Takeaway:** The 2022 bond failure proves exactly why the Max Diversification strategy is superior. Because it isn't permanently married to bonds, it actively hunts for true un-correlation anywhere it can find it. This dynamic adaptability allows it to navigate bizarre regime shifts far more effectively than strategies that rely on static defensive hoarding.

#### Rolling Sharpe Ratio -- Consistency Over Time

![Rolling 12-Month Sharpe Ratio](../Outputs/backtest_rolling_sharpe.png)

The naive Equal Weight strategy is a complete rollercoaster. During raging bull markets, its Sharpe Ratio spikes up to an unsustainable +5. But the moment the market cracks, it violently collapses below -2. This massive swing proves that Equal Weight doesn't manage risk at all—it just blindly rides the market's momentum, leaving you completely exposed when the tide turns. By contrast, the two risk-based strategies carve out a much tighter, more stable path. Across almost every single market environment—whether a boom or a bust—they maintain significantly better, smoother risk-adjusted returns. They might not catch the absolute peak of a reckless bull market, but more importantly, they protect you from the devastating, sleep-losing lows of a crisis.

#### Dynamic Weight Allocations

![Portfolio Weight Allocations Over Time](../Outputs/backtest_weight_allocations.png)

- **Equal Weight** By definition, this strategy forces a static 25% allocation into each asset class, completely ignoring market context. The visual result is flat lines across the entire timeline. When the 2008 and 2020 crashes hit, it mechanically held onto its heavy growth allocations, driving those devastating drawdowns.
- **Min Volatility** The visual weight chart confirms that this strategy is essentially a massive bond vault with a small equity engine. It consistently anchors ~75% of the portfolio in US Treasuries. When the algorithm detects a highly stable environment (like the 2005–2006 calm or the 2012–2014 QE expansion), it tentatively creeps its equity exposure up to ~30%. But the moment volatility flashes red, it instantly drops those equities and hides back in bonds. It is effective for capital preservation, but its heavy fixed-income bias restricts absolute growth.
- **Max Diversification** This is the engine behind the superior risk-adjusted performance. Max Diversification does not just hide in bonds; it actively manages a true multi-asset allocation based on shifting correlations. It maintains a stable baseline (roughly 65% bonds, with the rest distributed across Equities, REITs, and Commodities). Crucially, notice how the bands shift and breathe over time. It dynamically dials down specific growth assets exactly when their correlations begin to spike (as proven in the HMM analysis), allowing it to side-step major structural shocks while capturing growth across a much broader base of assets than Min Volatility.

### Enhanced Backtest: GARCH-Forecasted Covariance

As an enhancement, we also ran the same three strategies using **GARCH(1,1) volatility forecasting** with a 60-month expanding window, which better captures volatility clustering:

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Equal Weight (Baseline) | +5.43% | 10.75% | 0.505 | -40.4% | 0.135 | 
| Min Volatility (Expanding) | +3.98% | 5.36% | **0.742** | -17.3% | 0.229 |
| **Max Diversification (Expanding)** | +3.99% | 6.01% | 0.663 | **-15.7%** | **0.254** |

![Drawdown Analysis -- GARCH-Forecasted Covariance](../Outputs/garch_drawdowns.png)

![Capital Allocation Wealth -- GARCH-Forecasted ($1 Invested)](../Outputs/garch_cumulative_wealth.png)

* **Equal Weight (Baseline):** The Equal Weight strategy still suffers a catastrophic **-40.4% drawdown**. No matter what forecasting model you use, holding a static 25% allocation into a crisis guarantees massive capital destruction.
* **Min Volatility:** By constantly pushing capital into the safest possible assets based on the GARCH forecast, this strategy successfully minimized absolute variance to just **5.36%**, generating the highest Sharpe Ratio (0.742).
* **Max Diversification:** Once again, the Max Diversification strategy proves it is the ultimate engine for capital preservation. Even under this entirely different forecasting model, it delivered the absolute shallowest maximum drawdown (**-15.7%**) and the highest Calmar Ratio (**0.254**).

The GARCH-enhanced backtest confirms the same ordering: risk-based strategies (Min Vol, Max Div) dramatically outperform Equal Weight on a risk-adjusted basis.

---
### Study Synthesis: The Case for Dynamic Risk Allocation

Before delivering the final verdict, here is exactly what this comprehensive 26-year study proved:

1. **Parts 1 & 2 (The Problem with Traditional Models):** Standard optimization is dangerously fragile. It blindly trusts historical expected returns, and when you systematically neutralize those return assumptions, the entire architectural model falls apart.
2. **Part 3 (The Reality of Market Crises):** Market risk is not flat. We proved—and validated using an unbiased unsupervised AI model—that when a crisis hits, volatility triples and traditional diversification completely breaks down. All risk assets crash together.
3. **Part 4 (The Solution):** If you cannot predict returns and you cannot trust static risk assumptions, you have to adapt. Dynamic, risk-based portfolios completely outperformed traditional static models across every major market shock.
---

## Final Strategic Recommendation

For the deployment of institutional capital, the verdict is clear: The **Maximum Diversification Strategy** (using a Rolling 36-month window) is the superior allocation framework. Here is exactly why this strategy wins the mandate:

1. **Best Downside Protection:** It survived the 2008 financial crisis, the 2020 pandemic flash crash, and the 2022 inflation shock better than anything else we tested. It successfully capped its worst-case losses at just -16.1% over a brutal two-decade span.
2. **Highest Calmar Ratio (0.272):** It delivered the best return per unit of worst-case loss—the metric that matters most for capital preservation mandates.
3. **Active Diversification Benefit:** Unlike the Minimum Volatility strategy, which lazily hides in bonds, Maximum Diversification actively works for you. It constantly hunts for assets that aren't moving together, allowing it to capture growth during calm periods while naturally retreating to safety when correlations start to spike.
4. **Robustness:** Both simple historical and GARCH-enhanced covariance methods confirm Max Diversification's superiority, indicating the result is not model-dependent.
5. **HMM-Validated Regime Awareness:** Our data-driven HMM analysis confirms that the correlation-based diversification framework correctly adapts to the two distinct market states—maintaining balanced allocations during calm periods while naturally gravitating toward defensive assets during crises.

---

### Summary of All Methods & Results

| Part | Method | Key Insight |
| :--- | :--- | :--- |
| **1. MVO** | Full-sample Markowitz | Max Sharpe (SR=0.88) is mathematically optimal but dangerously relies on accurate return estimates. |
| **2. Equal SR** | Return sensitivity test | Optimal weights change dramatically when return assumptions change—proving MVO is fragile. |
| **3A. Rule-Based Regimes** | Hardcoded date ranges | Volatility spikes 2-3x during crises; correlations break down; "optimal" weights flip entirely. |
| **3B. HMM Regimes** | Unsupervised 2-state Gaussian HMM | Data-driven model independently discovers GFC/COVID crises; confirms correlation breakdown. |
| **4A. Backtest (Sample Cov)** | Rolling/Expanding 36m window | Max Div: best Calmar (0.272), shallowest drawdown (-16.1%). |
| **4B. Backtest (GARCH)** | GARCH(1,1) forecasted covariance | Confirms same ranking: Max Div > Min Vol >> Equal Weight on risk-adjusted basis. |



In a nutshell, Chasing past returns destroys wealth during a crisis. By transitioning to a dynamic, Maximum Diversification framework, allocators can stop guessing what the market will do next and start mathematically protecting capital against whatever comes.

---

## Appendix: Quantitative Engineering & Methodologies

The entire analysis in this report was programmatically generated using a robust Python architecture. It relies heavily on established quantitative libraries to ensure mathematical precision, automation, and fully reproducible results.

---

### 02: Portfolio Optimization Framework

The baseline Mean-Variance optimization engine was constructed using `scipy.optimize.minimize`, implementing the **Sequential Least Squares Programming (SLSQP)** algorithm. The model enforces strict long-only boundaries ($0 \le w_i \le 1$) and a hard capital constraint ($\sum w_i = 1$) at every iteration. 

For the "Equal Sharpe" stress tests, the traditional expected return vector ($\mu$) was programmatically overridden alongside the historical covariance matrix to simulate synthetic risk-return environments, precisely isolating the optimizer's behavioral mechanics when deterministic forecasting is removed.

### 03: Regime Change Analysis & Unsupervised Learning

The original deterministic analysis computed trailing 24-month volatilities and cross-asset correlation distributions using fast `pandas` vectorized rolling windows. 

To completely eliminate human hindsight bias in the secondary study, an unsupervised **Gaussian Hidden Markov Model (HMM)** was engineered via the `hmmlearn` library. Configured with two latent states (`n_components=2`) and `covariance_type="full"`, the model probabilistically clustered return profiles into "Calm" and "Crisis" distributions. The Viterbi algorithm was then deployed to decode the most likely historical sequence of these hidden states strictly from the raw multivariate return vectors, without any hardcoded dates or external macroeconomic data.

### 04: Advanced Backtesting & GARCH Variance Forecasting

The dynamic backtesting harness was built from scratch to simulate genuine out-of-sample institutional capital deployment. The engine systematically steps forward month-by-month, purging forward-looking data and recalculating the complex non-linear Min Volatility and Max Diversification targets over distinct memory structures (e.g., 36-month moving windows).

To account for volatility clustering, the enhanced covariance matrix deployed an **ARIMA-GARCH(1,1)** mathematical formulation via the `arch` library. Instead of naively averaging past historical standard deviations, the engine iteratively re-fit the GARCH model across the expanding memory window. It then computationally projected the 1-step-ahead ($t+1$) forward expected variance for each asset, ensuring that the backtesting allocations correctly anticipated compounded volatility shocks.