# Research memo

## Recommendation

Recommendation: against trading.

No signal survived every kill stage, so there is no strategy to trade. Each signal's outcome, and the numbers behind it, follow.

## Signals

The candidates were fixed before any stage ran on market data (ADR 0018), with the papers that motivate each. Each is one series tested against every asset's return.

| Signal | Definition | Motivated by | Outcome over the kill stages |
|---|---|---|---|
| sig_default_spread | Moody's Baa minus Aaa corporate bond yield (FRED DBAA - DAAA), the newest values usable at t | Keim & Stambaugh 1986; Fama & French 1989 | sig_default_spread was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |
| sig_inflation | CPI inflation (ALFRED CPIAUCSL): log change from month o-12 to o, both from the vintage usable at t | Fama & Schwert 1977 | sig_inflation was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |
| sig_ip_growth | industrial production (ALFRED INDPRO): log change from month o-1 to o, both from the vintage usable at t | Fama 1990 | sig_ip_growth was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |
| sig_payrolls_growth | nonfarm payrolls (ALFRED PAYEMS): log change from month o-1 to o, both from the vintage usable at t | Flannery & Protopapadakis 2002 | sig_payrolls_growth was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |
| sig_rel_bill | 3-month Treasury bill rate (FRED DTB3) minus its mean over the prior 252 sessions, each value as usable at its session | Ang & Bekaert 2007 | sig_rel_bill was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |
| sig_rv_spy | half the log of annualised SPY realised variance over sessions t-20 to t: 0.5 log((252/21) sum r^2) | French et al. 1987; Moreira & Muir 2017 | sig_rv_spy was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |
| sig_term_spread | 10-year minus 3-month Treasury constant-maturity yield (FRED T10Y3M), the newest value usable at t | Campbell 1987; Fama & French 1989 | sig_term_spread was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |
| sig_tsmom_spy | sum of SPY daily log returns over sessions t-251 to t-21 (12-1 month time-series momentum) | Moskowitz et al. 2012 | sig_tsmom_spy was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |
| sig_unrate_change | unemployment rate (ALFRED UNRATE): change from month o-1 to o, both from the vintage usable at t | Boyd et al. 2005 | sig_unrate_change was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |
| sig_vrp | variance risk premium: (VIX/100)^2/12 minus the sum of squared SPY daily log returns over sessions t-20 to t | Bollerslev et al. 2009 | sig_vrp was killed at Stage 1 (causal screen) by its test (causal.no_significant_link) |

## Causal screen

Each signal's lagged link to the asset returns, given the observed variables. The verdict uses the BH q-value at the 0.05 level.

| Signal | Family p | BH q | Best link | Partial correlation |
|---|---|---|---|---|
| sig_default_spread | 1.000 | 1.000 | GLD, lag 5 | 0.019 [-0.004, 0.043], 95% interval |
| sig_inflation | 0.652 | 1.000 | XLV, lag 1 | -0.031 [-0.054, -0.008], 95% interval |
| sig_ip_growth | 1.000 | 1.000 | XLP, lag 3 | 0.039 [0.009, 0.068], 95% interval |
| sig_payrolls_growth | 0.072 | 0.716 | XLI, lag 1 | -0.048 [-0.069, -0.007], 95% interval |
| sig_rel_bill | 1.000 | 1.000 | XLF, lag 3 | 0.037 [0.001, 0.069], 95% interval |
| sig_rv_spy | 1.000 | 1.000 | XLB, lag 5 | 0.027 [-0.003, 0.061], 95% interval |
| sig_term_spread | 1.000 | 1.000 | GLD, lag 3 | 0.044 [0.011, 0.074], 95% interval |
| sig_tsmom_spy | 1.000 | 1.000 | XLU, lag 2 | 0.014 [-0.023, 0.045], 95% interval |
| sig_unrate_change | 0.467 | 1.000 | XLI, lag 1 | -0.037 [-0.061, -0.014], 95% interval |
| sig_vrp | 0.507 | 1.000 | IEF, lag 5 | -0.050 [-0.090, -0.009], 95% interval |

## Limits that apply to this run

This run used market data, with known limits (ADR 0004): the ETF universe has survivorship and hindsight bias, and price levels are not point-in-time, so only returns are used. Daily FRED market series are used as FRED first posted them once their archive begins; before it, each value is assumed usable a fixed number of sessions after its date, set from the archive's regular posting delays, and revisions before the archive cannot be seen (ADR 0019). Macro series are the vintages known at each date, and their changes are computed within one vintage (ADR 0019). Stage 1 (causal screen) tests lags starting one session back, which includes a link a trade entered at the next session's close could not use; Stage 3 (portfolio construction) trades at that close, so it alone tests what could have been traded (ADR 0018). Every rate this memo quotes from the planted-signal benchmark is a benchmark validation rate, measured on synthetic data with a known answer. It describes how the pipeline's tests behave, and is not a property of this strategy.

A kill at Stage 1 (causal screen) means the edge was not detectable at this sample size, not that there is no effect. The test has little power against weak edges. On the planted-signal benchmark (realistic preset, ADR 0010), a true edge of realistic strength passed Stage 1 in 0.140 [0.112, 0.173] of runs (95% interval). On the planted-signal benchmark (base preset, ADR 0010), a true edge of the default strength passed Stage 1 in 0.844 [0.810, 0.873] of runs (95% interval).

At Stage 1 (causal screen), the best lag is where the evidence was strongest, not an identified horizon, and the partial-correlation intervals are descriptive: they are not adjusted for the selection of the link and ran narrow on the benchmark (ADR 0010).

Stage 4 (adversarial regime certification) was evaluated and not adopted as a kill stage (ADR 0016). At about twenty-four years of daily data, the plausible variation in a strategy's exposure to high-volatility periods is too small to overturn a strategy that is profitable on average, and a rule that killed for losing money in stress would penalise ordinary market exposure rather than fragility. In the Stage 4 feasibility check on the planted-signal benchmark (volfrag preset, population moments, ADR 0016), a signal whose edge reverses in high volatility still had a worst plausible horizon Sharpe ratio of 0.90 (median over windows; range [0.82, 0.98]).

## Methods

Stage 1 (causal screen): PCMCI (Runge et al. 2019) with an HC3 partial-correlation test (MacKinnon & White 1985), Bonferroni over each signal's lags and assets, and the Benjamini-Hochberg procedure across signals (Benjamini & Hochberg 1995) (ADR 0009, ADR 0010). Persistent predictors distort such tests in small samples (Stambaugh 1999), which is why the stage requires a minimum sample.

Benchmark validation rates carry Wilson intervals (Wilson 1927).

## Reproducibility

run id = 75f1372b65ff89f7; data snapshot = 930504ace5464752; code commit = ae657813cc840a9792b51c20029d8bf41dcd2c20; run seed = 20261112; panel rows = 5282, from 2005-01-04 to 2025-12-31. A run is keyed by a hash of its data snapshot, config, code commit and seed (ADR 0002); every decision is in its hash-chained kill ledger (ADR 0007).

## References

Ang, Bekaert (2007). Stock return predictability: Is it there?. Review of Financial Studies 20(3). https://doi.org/10.1093/rfs/hhl021 Summary: Finds that the short interest rate is the most robust short-horizon predictor of excess equity returns, and dividend yield predictability weak.

Benjamini, Hochberg (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society Series B 57(1). https://doi.org/10.1111/j.2517-6161.1995.tb02031.x Summary: The Benjamini-Hochberg step-up procedure, which controls the false discovery rate for independent or PRDS tests.

Bollerslev, Tauchen, Zhou (2009). Expected stock returns and variance risk premia. Review of Financial Studies 22(11). https://doi.org/10.1093/rfs/hhp008 Summary: Finds that the gap between implied and realized variance (the variance risk premium) predicts quarterly aggregate stock returns.

Boyd, Hu, Jagannathan (2005). The stock market's reaction to unemployment news: Why bad news is usually good for stocks. Journal of Finance 60(2). https://doi.org/10.1111/j.1540-6261.2005.00742.x Summary: Finds that the stock market's response to rising unemployment depends on the business cycle, through interest rates and growth expectations.

Campbell (1987). Stock returns and the term structure. Journal of Financial Economics 18(2). https://doi.org/10.1016/0304-405X(87)90045-6 Summary: Shows that variables from the term structure of interest rates predict excess returns on stocks and bonds.

Fama, Schwert (1977). Asset returns and inflation. Journal of Financial Economics 5(2). https://doi.org/10.1016/0304-405X(77)90014-9 Summary: Finds that common stock returns were negatively related to expected and unexpected inflation.

Fama, French (1989). Business conditions and expected returns on stocks and bonds. Journal of Financial Economics 25(1). https://doi.org/10.1016/0304-405X(89)90095-0 Summary: Finds that the default spread and term spread track business conditions and predict expected returns on stocks and bonds.

Fama (1990). Stock returns, expected returns, and real activity. Journal of Finance 45(4). https://doi.org/10.1111/j.1540-6261.1990.tb02428.x Summary: Relates stock returns to future growth in industrial production and to shocks to expected returns.

Flannery, Protopapadakis (2002). Macroeconomic factors do influence aggregate stock returns. Review of Financial Studies 15(3). https://doi.org/10.1093/rfs/15.3.751 Summary: Finds that announcements of a few macro series, including the employment report, move aggregate stock returns or their volatility.

French, Schwert, Stambaugh (1987). Expected stock returns and volatility. Journal of Financial Economics 19(1). https://doi.org/10.1016/0304-405X(87)90026-2 Summary: Relates the expected market risk premium to predictable stock market volatility, and unexpected volatility to contemporaneous returns.

Keim, Stambaugh (1986). Predicting returns in the stock and bond markets. Journal of Financial Economics 17(2). https://doi.org/10.1016/0304-405X(86)90070-X Summary: Shows that ex-ante observable variables, including the spread between low-grade bond and bill yields, predict stock and bond risk premia.

MacKinnon, White (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. Journal of Econometrics 29(3). https://doi.org/10.1016/0304-4076(85)90158-7 Summary: Proposes the HC2 and HC3 heteroskedasticity-consistent covariance estimators; HC3 has better small-sample size.

Moreira, Muir (2017). Volatility-managed portfolios. Journal of Finance 72(4). https://doi.org/10.1111/jofi.12513 Summary: Shows that scaling factor exposure down when recent realized variance is high raised Sharpe ratios, as volatility rose more than expected returns.

Moskowitz, Ooi, Pedersen (2012). Time series momentum. Journal of Financial Economics 104(2). https://doi.org/10.1016/j.jfineco.2011.11.003 Summary: Documents that an asset's own past 12-month excess return predicts its future return across futures markets (time-series momentum).

Runge, Nowack, Kretschmer, Flaxman, Sejdinovic (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. Science Advances 5(11). https://doi.org/10.1126/sciadv.aau4996 Summary: Introduces PCMCI, a two-step causal discovery method for time series (PC1 parent selection, then momentary conditional independence tests).

Stambaugh (1999). Predictive regressions. Journal of Financial Economics 54(3). https://doi.org/10.1016/S0304-405X(99)00041-0 Summary: Shows that persistent predictors bias predictive-regression slopes and distort their tests in finite samples.

Wilson (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association 22(158). https://doi.org/10.1080/01621459.1927.10502953 Summary: The Wilson score interval for a binomial proportion.
