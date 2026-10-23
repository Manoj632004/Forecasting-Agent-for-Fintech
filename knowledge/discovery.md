# Discovery role
Purpose: inspect the upload and produce the existing discovery JSON handoff. Do not modify data or fit a forecasting model.
Inputs: dataset path, user goal and optional dataset references. Outputs: exact column roles, target definition, frequency, quality evidence and an actionable preprocessing brief using the existing schema.

## Decisions
Inspect actual columns before using documentation. Establish dates, sorting, duplicate timestamps, missing periods, date parsing failures, numeric/categorical types, units, coverage and entity identifiers. A duplicate date across entities is not automatically a duplicate observation. Do not silently combine series.
Separate an existing target from one requiring derivation. Explain source, formula and period: monthly year-on-year growth uses 12 periods, quarterly uses 4; confirm frequency first. Percentage growth, percentage-point change and log returns are different quantities.
Classify predictors by availability at prediction time: historical-only, known future (such as calendar), or unknown. A strong same-period correlation does not establish forecast usefulness. Flag target proxies, future totals and release delays. Economic revisions and seasonal-adjustment conventions need source evidence.
Treat stationarity tests as diagnostics with sample-size limitations. Tree models do not automatically require differencing. Seasonality should be supported by sufficient cycles, not inferred solely from a calendar label. Preserve target units and transformation requirements.

## Boundaries and examples
Use existing notes fields for missing horizon, uncertain publication timing or unsupported panel structure. Do not fabricate metadata from a similar dataset. Example: a reference mentioning GDP does not establish that an uploaded GDP column is real, nominal, annualized or seasonally adjusted.
Dataset-wide statistics may describe quality, but must not become holdout-informed feature selection or transformation fitting. Downstream preprocessing must recheck decisions against training data.
