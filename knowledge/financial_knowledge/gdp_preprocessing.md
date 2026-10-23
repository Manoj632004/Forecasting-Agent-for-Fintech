# GDP dataset preprocessing

## Expected GDP input
Use the discovery handoff to identify the uploaded GDP series and its requested target: nominal/real level, volume index, per-capita value or growth rate. Preserve country, currency, multiplier (units/millions/billions), price basis, seasonal adjustment, frequency and vintage where documented. Never assign missing metadata from a different GDP dataset.

## GDP dates and coverage
Parse quarter labels such as 2020 Q1 consistently and sort by economic period. A quarter-start timestamp and a quarter-end timestamp can identify the same quarter; normalize deliberately before joining. Keep country-specific histories separate. Distinguish annual summary rows from quarterly observations. Check complete calendar spacing before interpreting a shift of four rows as one year.
For repeated country-quarter rows, determine whether they are duplicates, GDP variants or publication vintages. Do not average real and nominal GDP or distinct revisions. A missing GDP quarter is an unknown observation, not zero production. Do not interpolate target GDP using a later quarter or forward-fill GDP targets to create artificial labels.

## GDP target preparation
If the request is growth from a level series, compute 100 * (Y[t] / Y[t-k] - 1): k=1 for ordinary quarterly growth, k=4 for quarterly year-on-year growth, k=12 for monthly year-on-year growth. These assume complete, regular periods. Existing GDP growth columns must not undergo percentage change again. Fractional growth 0.02 corresponds to 2 percent; verify units before multiplying.
Annualized quarterly growth is 100 * ((Y[t] / Y[t-1])**4 - 1), not exactly four times ordinary growth. Do not divide GDP levels by four merely because a growth series is annualized. Annual-rate levels and annualized growth rates are different metadata concepts. BEA describes its annual-rate convention here: https://bea.gov/index.php/news/gdp-release-additional-information
Check zero denominators, unavailable preceding periods and nonpositive values before logs. Keep the target definition and scale in transformation metadata. Remove the unavoidable first k target rows only after aligning dates and predictors. Retain historical anchors if forecasts will be converted back to GDP levels.

## GDP predictors and mixed frequencies
For monthly indicators joined to quarterly GDP, define the economic aggregation: quarterly mean may suit a monthly rate or index, a sum may suit a flow, and quarter-end may suit a stock. Confirm each source's units and frequency first. Quarterly annual-rate flows must not be treated as ordinary quarterly totals without conversion justified by metadata.
Join indicators by information available at the forecast origin, including publication delays. A complete three-month quarterly average cannot be used before those months and releases are observed. Incomplete-quarter nowcasting needs explicit handling; do not silently fill remaining months from future data. Unavailable release metadata should remain a warning.
Possible engineered features are quarterly GDP-growth lags 1, 2, 3, 4 and four/eight-quarter summaries excluding the target quarter. Use only histories supported by the dataset. The level Y[t] used to derive current GDP growth leaks that growth when supplied as a current predictor; Y[t-1] may be usable if already published.

## GDP shocks, seasonal adjustment and handoff
Pandemic contractions and rebounds can be genuine GDP events; do not automatically clip or delete them as data errors. Verify discontinuities against units, source definitions and revisions. Keep documented seasonally adjusted GDP as such; do not apply a second seasonal adjustment blindly. ONS publishes GDP variants and revisions: https://www.ons.gov.uk/economy/grossdomesticproductgdp/
Produce aligned chronological GDP features, targets and dates. Record growth periods, unit conversion, retained historical anchors, aggregation conventions and vintage assumptions in supported metadata fields. If the existing tools cannot perform a required GDP conversion safely, report the limitation rather than substituting a different target.
