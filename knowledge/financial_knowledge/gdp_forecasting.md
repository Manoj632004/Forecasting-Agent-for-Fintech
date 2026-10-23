# GDP dataset forecasting

## GDP forecasting task
Distinguish forecasting a future quarter's GDP from nowcasting an ongoing or completed but unpublished quarter. Establish forecast origin, horizon in GDP periods, country and target convention. GDP level, ordinary quarterly growth, annualized quarterly growth and year-on-year growth require different output labels. A generic column name does not establish the units.

## GDP history and candidate models
Quarterly GDP often offers a modest number of observations relative to potential indicators. Count usable quarters after target derivation and lags rather than assuming a fixed history length. Start with a small GDP lag set and avoid fitting many parameters to a short effective sample.
For GDP levels, consider a last-observed-level baseline and, if supported and justified, drift. For GDP growth, compare persistence and a training-history mean; a zero-growth baseline can also be informative. A seasonal baseline is a candidate when the GDP series has meaningful seasonality, not an automatic requirement for seasonally adjusted GDP.
An autoregressive GDP-growth model can provide a compact benchmark. If the registered model is XGBoost, use available GDP lag and release-safe indicator features; do not claim a statistical model was run unless the implementation supports it. GDP expenditure components, industrial production or unemployment are candidate covariates, not guaranteed inputs in every upload.

## GDP-specific tuning considerations
When validation search is implemented, evaluate a small set of GDP lag lengths (for example 1, 2 and 4 quarters) and model settings with expanding or rolling chronological windows. Choose validation horizons matching the requested one-quarter or multi-quarter GDP forecast. Overlapping year-on-year GDP-growth targets share underlying periods; inspect horizon and information overlap when defining folds.
For short quarterly GDP samples, shallow XGBoost trees, larger minimum child weights and stronger regularization are candidates to evaluate, not fixed economic truths. Tune learning rate jointly with tree count. Do not copy numeric settings from a different country's GDP dataset or from its in-sample R-squared. If the tool only fits one model on a fixed split, report that no tuning search was performed.

## GDP data vintages and forecast covariates
First-release GDP and subsequently revised GDP are distinct evaluation targets. State which vintage supplies truth. A model trained and evaluated on today's revised history is a latest-vintage experiment, not proof of real-time nowcasting skill. UK GDP revision triangles provide growth/revision evidence: https://www.ons.gov.uk/economy/grossdomesticproductgdp/datasets/revisionstrianglesforukgdpabmi
For each forecast origin, GDP lags and indicators must have been published by that time. A nominal one-quarter lag may still be unpublished. Do not use observed future employment, trade or GDP components for a future multi-quarter path without explicitly labeling a conditional scenario. A backtest that uses actual newly observed GDP history at every step is a rolling backtest, not a single-origin future forecast.

## GDP output scale and interpretation
For percentage GDP-growth targets, MAE/RMSE are in percentage points. For fractional growth targets, convert explicitly before labeling errors in percentage points. For GDP levels, report the documented currency/index units. MAPE is unreliable for GDP growth around zero or below zero.
To restore ordinary quarterly growth g[t] in percent to levels, use Y[t] = Y[t-1] * (1 + g[t]/100). For annualized quarterly growth a[t], use Y[t] = Y[t-1] * (1 + a[t]/100)**(1/4), within the valid domain. Year-on-year growth requires the corresponding year-earlier level rather than the immediately preceding quarter. Preserve anchors and propagate predictions recursively when required; never use future actual GDP as inversion anchors for a single-origin forecast.
Compare performance on ordinary quarters and exceptional GDP shocks without silently deleting the shocks or selecting models using the test period. Do not describe a GDP-growth forecast as a recession probability or an investment recommendation. Save the GDP definition, horizon, vintage assumption and evaluation type through supported result metadata.
