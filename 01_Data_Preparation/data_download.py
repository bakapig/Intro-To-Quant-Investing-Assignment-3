"""
Download monthly returns data for portfolio optimization.

Asset Classes & Data Sources:
  - US Equities:       SPY    (SPDR S&P 500 ETF, via Yahoo Finance)
  - US 10Yr Treasury:  GS10   (10-Year Constant Maturity Yield, via FRED)
                       -> Total return approximated using:
                          R_t ≈ Y_{t-1}/12 - D × (Y_t - Y_{t-1})
                          where D = 8.5 (modified duration)
  - US REITs:          VGSIX  (Vanguard REIT Index Fund, via Yahoo Finance)
  - US Commodities:    ^BCOM  (Bloomberg Commodity Index, via Yahoo Finance)

Data range: 2000-01-01 to present (all assets have data from 2000)
"""

import yfinance as yf
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
YF_TICKERS = {
    "US_Equities": "SPY",  # S&P 500
    "US_REITs": "VGSIX",  # Vanguard REIT Index Fund (May 1996)
    "US_Commodities": "^BCOM",  # Bloomberg Commodity Index (broad, diversified)
}

FRED_YIELD_SERIES = "GS10"  # 10-Year Constant Maturity Treasury Yield

START_DATE = "1999-12-01"  # Fetch from Dec 1999 so first return = Jan 2000
END_DATE = None  # None = today
import os
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "Data", "monthly_returns.csv")


# ── Download YF prices & compute returns ──────────────────────────────────────
def download_yf_returns(
    tickers: dict, start: str, end: str | None = None
) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance, resample to month-end, compute returns."""
    ticker_list = list(tickers.values())
    print(f"Downloading daily prices for: {ticker_list}")

    data = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=True,
    )

    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = ticker_list

    # Rename columns from tickers to descriptive names
    reverse_map = {v: k for k, v in tickers.items()}
    prices.rename(columns=reverse_map, inplace=True)

    # Resample daily -> month-end (last trading day of each month)
    prices_monthly = prices.resample("ME").last()

    # Compute simple monthly returns
    returns = prices_monthly.pct_change().dropna(how="all")
    return returns


# ── Download FRED yield & compute 10Yr Treasury returns ──────────────────────
def download_treasury_returns(
    fred_series: str,
    start: str,
    end: str | None = None,
) -> pd.Series:
    """
    Download 10-Year Treasury yield from FRED and calculate exact monthly total
    returns by dynamically computing Macaulay and Modified Duration at each step.

    WHY DYNAMIC DURATION?
    Instead of using a hardcoded average duration (like D=8.5), we precisely 
    calculate the Macaulay Duration for a 10-year par bond at the prevailing 
    market yield every single month. As yields fall, duration mathematically 
    extends; as yields rise, it contracts. This ensures our price-shock 
    simulations are 100% mathematically rigorously aligned with the environment.

    Total Monthly Return:
        R_t ≈ (Y_{t-1} / 12)  -  D^{mod}_{t-1} × (Y_t - Y_{t-1})
    """
    import numpy as np

    print(f"Downloading FRED series: {fred_series}")
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={fred_series}&cosd={start}"
    )
    if end:
        url += f"&coed={end}"
    yields_df = pd.read_csv(
        url, parse_dates=["observation_date"], index_col="observation_date"
    )
    yields_df.columns = [fred_series]
    yields_df[fred_series] = pd.to_numeric(yields_df[fred_series], errors="coerce")
    yields_monthly = yields_df.resample("ME").last().dropna()
    y = yields_monthly[fred_series] / 100  # Convert percentage to decimal

    # --- DYNAMIC DURATION CALCULATION ---
    # Assumption: The generic 10-Year Treasury Yield series represents a newly 
    # issued par bond. For a par bond, Coupon Rate = Yield to Maturity.
    m = 2  # Semi-annual payments (standard for US Treasuries)
    periods = 10 * m
    d_mods = []

    for yield_val in y:
        if pd.isna(yield_val) or yield_val <= 0:
            d_mods.append(np.nan)
            continue
            
        y_s = yield_val / m  # Semi-annual yield
        coupon = (yield_val / m) * 100  # Semi-annual coupon cash flow (assuming 100 par)
        
        # Calculate Macaulay Duration numerator: sum of (t * CF_t) / (1+r)^t
        mac_dur_numerator = 0
        for i in range(1, periods + 1):
            t_i = i / m  # Time in years
            cf = coupon if i < periods else (100 + coupon)  # Principal returned at end
            mac_dur_numerator += (t_i * cf) / ((1 + y_s)**i)
            
        # Since it is a par bond, the Present Value (denominator) is exactly 100
        mac_duration = mac_dur_numerator / 100.0
        
        # Convert to Modified Duration
        mod_duration = mac_duration / (1 + y_s)
        d_mods.append(mod_duration)
        
    d_mod_series = pd.Series(d_mods, index=y.index)

    # Calculate approximate Total Return:
    # R_t = (Income from Yield) + (Capital Gain/Loss using dynamic duration)
    y_prev = y.shift(1)
    d_mod_prev = d_mod_series.shift(1)
    
    returns = y_prev / 12 - d_mod_prev * (y - y_prev)
    returns = returns.dropna()
    returns.name = "US_10Yr_Bond"
    # Normalize index to month-end for alignment with YF data
    returns.index = returns.index.to_period("M").to_timestamp("M")

    print(
        f"  10Yr Treasury returns: {returns.index.min().date()} to {returns.index.max().date()}"
    )
    return returns


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Download Yahoo Finance returns (equities, REITs, gold)
    yf_returns = download_yf_returns(YF_TICKERS, START_DATE, END_DATE)

    # 2. Download 10-Year Treasury returns from FRED
    treasury_returns = download_treasury_returns(
        FRED_YIELD_SERIES, START_DATE, END_DATE
    )

    # 3. Merge all returns into one DataFrame
    monthly_returns = yf_returns.join(treasury_returns, how="inner")

    # Reorder columns for clarity
    col_order = ["US_Equities", "US_10Yr_Bond", "US_REITs", "US_Commodities"]
    monthly_returns = monthly_returns[
        [c for c in col_order if c in monthly_returns.columns]
    ]
    monthly_returns.dropna(inplace=True)

    # Trim to start from Jan 2000
    monthly_returns = monthly_returns.loc["2000-01-01":]

    print(f"\nMonthly returns shape: {monthly_returns.shape}")
    print(
        f"Date range: {monthly_returns.index.min().date()} to {monthly_returns.index.max().date()}"
    )
    print(f"\n── Monthly Returns (first 5 rows) ──")
    print(monthly_returns.head().to_string())
    print(f"\n── Summary Statistics (annualized) ──")
    summary = pd.DataFrame(
        {
            "Ann. Mean Return": monthly_returns.mean() * 12,
            "Ann. Volatility": monthly_returns.std() * (12**0.5),
            "Sharpe (rf=0)": (monthly_returns.mean() * 12)
            / (monthly_returns.std() * (12**0.5)),
        }
    )
    print(summary.to_string())

    # 4. Save to CSV
    monthly_returns.index.name = "Date"
    monthly_returns.to_csv(OUTPUT_CSV)
    print(f"\nSaved monthly returns to: {OUTPUT_CSV}")

    # 5. Compute MVO inputs: expected returns (μ), covariance matrix (Σ)
    import numpy as np

    mu_monthly = monthly_returns.mean()
    cov_monthly = monthly_returns.cov()

    # Annualize
    mu_annual = mu_monthly * 12
    cov_annual = cov_monthly * 12
    vol_annual = np.sqrt(np.diag(cov_annual))

    print(f"\n{'=' * 60}")
    print("MVO INPUTS (annualized, from full historical data)")
    print(f"{'=' * 60}")

    print(f"\n── Expected Returns (μ) ──")
    for asset, ret in mu_annual.items():
        print(f"  {asset:25s}  {ret:+.4f}  ({ret*100:+.2f}%)")

    print(f"\n── Annualized Volatility (σ) ──")
    for asset, v in zip(mu_annual.index, vol_annual):
        print(f"  {asset:25s}  {v:.4f}  ({v*100:.2f}%)")

    print(f"\n── Covariance Matrix (Σ, annualized) ──")
    print(cov_annual.to_string())

    corr = monthly_returns.corr()
    print(f"\n── Correlation Matrix ──")
    print(corr.to_string())

    # Save MVO inputs
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "Outputs")
    mu_annual.to_csv(os.path.join(out_dir, "expected_returns.csv"), header=["Ann_Expected_Return"])
    cov_annual.to_csv(os.path.join(out_dir, "covariance_matrix.csv"))
    print(f"\nSaved: expected_returns.csv, covariance_matrix.csv")

    return monthly_returns


if __name__ == "__main__":
    monthly_returns = main()
