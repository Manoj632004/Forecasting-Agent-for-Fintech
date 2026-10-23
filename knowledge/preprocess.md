# Preprocessing role
Purpose: execute supported preparation tools and produce chronological splits and a reproducible manifest. Inputs: dataset and discovery brief. Outputs: feature/target splits, dates, transformation metadata and operation log in the existing format.

## Decisions
Validate column roles and dates against tool results. Keep timestamps aligned with labels and features. Establish split boundaries before estimating any data-dependent transformation; recompute boundaries if row removal changes the effective training population. Do not fill unknown targets to fabricate training labels.
Forward fill only when carrying the last observed value is meaningful; never backward fill from future observations. Mean/median/mode imputation and outlier thresholds must use training observations only. Missing calendar periods and missing values are different problems. Resampling must respect units: flows may sum, stocks often use period-end values, rates need an explicit convention. Never silently aggregate entities.
Create target lags using positive shifts. For predicting y[t] from past observations, rolling means must exclude y[t], for example shift(1).rolling(window). Reset histories by entity. Drop unavoidable leading missing rows and record losses. Never use an unavailable same-period feature merely because it exists in historical data.
Avoid automatic differencing or outlier deletion. Financial shocks can be real signals. Trees usually do not require scaling; linear and distance-based models may. Preserve inversion information for transformed targets, including anchors and periods. Never report transformed-scale results as original-scale forecasts.

## Boundaries and examples
Follow supported tool signatures and output schemas. Required leakage rules override a brief suggesting unsafe operations; report unsupported safe alternatives through existing error/notes paths. Example: filling a training gap with the next month's observed target leaks. A last-value fill uses only history but still requires domain justification.
