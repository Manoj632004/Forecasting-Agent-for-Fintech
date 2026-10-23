# GDP dataset discovery

## Dataset identity
GDP (gross domestic product) measures economic production over a period. GDP uploads may contain currency levels, volume indices, growth rates or per-capita measures. Common search terms include GDP, gross domestic product, economic growth, real GDP and nominal GDP. Names such as GDP, gdp_growth and UKNGDP are hints, not proof of definition.

For the actual uploaded series, establish country, publisher, series code, units, frequency, seasonal adjustment, price basis and growth convention. Record unknown metadata explicitly. Do not infer that UKNGDP is a particular official series from its name alone.

## GDP series definitions
Nominal GDP uses current prices; real GDP removes price effects. A real-GDP growth question cannot be answered by simply relabeling nominal growth. Per-capita GDP and aggregate GDP are different targets. A chained-volume index is not a currency amount.
UK GDP releases include monthly, quarterly and annual estimates, with current-price and chained-volume measures. Verify which series is uploaded rather than assigning quarterly frequency to every GDP dataset. Source: https://www.ons.gov.uk/economy/grossdomesticproductgdp/

## Target derivation from GDP levels
Preserve an existing growth target if its definition matches the request. If the user requests GDP levels, keep levels as the target; a transformation is a modeling choice requiring inversion, not an automatic change of goal.
For ordered, regularly spaced GDP levels Y[t], derive growth only when requested:
- Previous-period percentage growth: 100 * (Y[t] / Y[t-1] - 1).
- Quarterly year-on-year percentage growth: 100 * (Y[t] / Y[t-4] - 1).
- Monthly year-on-year percentage growth: 100 * (Y[t] / Y[t-12] - 1).
- Annualized quarter-on-quarter percentage growth: 100 * ((Y[t] / Y[t-1])**4 - 1).
The quarterly annualization exponent does not apply to annual data. Ordinary quarterly growth, annualized quarterly growth and year-on-year growth are distinct labels. US BEA featured quarterly growth is annualized; check the uploaded series metadata. Source: https://bea.gov/index.php/news/gdp-release-additional-information

## Dataset checks and candidate predictors
Confirm the date represents the economic period, not publication day. Check missing quarters, duplicate country-period records, mixed annual/quarterly rows and breaks in units. Different GDP vintages for the same quarter are revisions, not necessarily duplicate errors.
Potential GDP predictors include industrial production, employment, unemployment, trade, consumption, surveys and interest rates. Confirm geography, period, definition and publication availability before recommending them. GDP expenditure components from the target quarter may be unavailable at the forecast origin; they are not automatically valid predictors.
Candidate quarterly histories include GDP-growth lags 1 through 4 and past-only four/eight-quarter summaries, subject to sample length and validation. The contemporaneous GDP level used to derive growth contains the current target information; past GDP levels may be valid when available.

## Revisions and discovery handoff
GDP is revised after first publication. A latest-vintage historical file does not reproduce information available in real time. Identify vintage/release-date columns when present; otherwise flag that limitation. UK revision triangles explicitly track GDP growth and revisions. Source: https://www.ons.gov.uk/economy/grossdomesticproductgdp/datasets/revisionstrianglesforukgdpabmi
Pass the exact target column, any derivation formula and periods, date convention, candidate features and unresolved GDP metadata through the existing discovery schema. Example: quarterly real GDP growth in percentage points must not be confused with a fractional growth series or a nominal currency level.
