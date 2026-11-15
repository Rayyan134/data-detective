# Investigator baseline

> **Pre-normalisation run.** This is the raw `gemini-3.6-flash` scoring run from
> 2026-11-15, captured **before** the parameter-normalisation fix in
> `src/investigator.py`. All six failures are the same defect (model reasoning
> text leaking into the `value_column` string). See `eval/baseline.md` for the
> diagnosis and the fix. Kept for the record; a post-fix run will overwrite the
> unsuffixed `eval/baseline-gemini-3.6-flash.md`.

- Date: 2026-11-15
- Model: `gemini-3.6-flash`
- Dataset: `data/sample_sales.csv`
- Mode: planning only (tool selection and parameters scored)
- Score: **9 / 15**

| ID | Expected tool | Chosen tool | Verdict | Seconds | Note |
|---|---|---|---|---|---|
| Q01 | group_comparison | (none) | fail | 11.7 | The model's plan was rejected: Step 1 (group_comparison) parameter 'value_column' refers to unknown column 'revenue shelter_or_similar_placeholder_if_needed_no_just_revenue_and_metric_mean_is_default_or_explicitly_set_to_mean'. Available columns: order_id, order_date, region, product, channel, units, unit_price, discount_pct, revenue, satisfaction, returned. |
| Q02 | group_comparison | (none) | fail | 28.9 | The model's plan was rejected: Step 1 (group_comparison) parameter 'value_column' refers to unknown column 'revenue oppose metric sum'. Available columns: order_id, order_date, region, product, channel, units, unit_price, discount_pct, revenue, satisfaction, returned. |
| Q03 | correlation | correlation | pass | 9.8 |  |
| Q04 | trend_over_time | (none) | fail | 89.0 | The model's plan was rejected: Step 1 (trend_over_time) parameter 'value_column' refers to unknown column 'revenue নদী revenue column name check: revenue is numeric column in schema. order_date is date-like column in schema. freq: ME for monthly trends. Wait, let me fix the string value for value_column: 'revenue'. me parameter is 'ME'. me: ME. Let me ensure exact match with schema parameters. date_column: order_date, value_column: revenue, freq: ME. perfect. double quote string strictly. no typos. standard json format strictly no comments. Wait, inside reasoning keep concise. steps: 1 step. simple plan. json valid strictly. JSON validate double quotes. done. fine string for value_column: revenue, freq: ME, date_column: order_date. purpose text. complete. clean answer string below. Wait, let me check tools parameter defaults: freq default is ME so 'ME' is fine. purpose: Calculate monthly totals of revenue over order_date to analyze change over time. perfect. format string completely. done. simple simple simple. schema check logic. valid logic. strictly output single parseable json array or object. user context schema object. done. valid parseable json only. format check string: status plan, reasoning string, steps array with tools, params, purpose. correct.'. Available columns: order_id, order_date, region, product, channel, units, unit_price, discount_pct, revenue, satisfaction, returned. |
| Q05 | find_outliers | find_outliers | pass | 9.7 |  |
| Q06 | value_counts_table | value_counts_table | pass | 15.2 |  |
| Q07 | value_counts_table | value_counts_table | pass | 7.0 |  |
| Q08 | missing_by_group | missing_by_group | pass | 24.5 |  |
| Q09 | correlation | correlation | pass | 23.7 |  |
| Q10 | group_comparison | (none) | fail | 5.4 | The model's plan was rejected: Step 1 (group_comparison) parameter 'value_column' refers to unknown column 'satisfaction MART_METRIC_MEAN_PLACEHOLDER_MARKER_FOR_METRIC_VALUE_IF_NEEDED_BUT_EXPLICIT_ENUM_IS_mean'. Available columns: order_id, order_date, region, product, channel, units, unit_price, discount_pct, revenue, satisfaction, returned. |
| Q11 | trend_over_time | (none) | fail | 4.0 | The model's plan was rejected: Step 1 (trend_over_time) parameter 'value_column' refers to unknown column 'revenue নদী_freq='. Available columns: order_id, order_date, region, product, channel, units, unit_price, discount_pct, revenue, satisfaction, returned. |
| Q12 | correlation | correlation | pass | 3.6 |  |
| Q13 | group_comparison | (none) | fail | 3.9 | The model's plan was rejected: Step 1 (group_comparison) parameter 'value_column' refers to unknown column 'revenuepopular_metric_or_whatever_it_is_called_in_schema_check_enum_metric_is_median_or_mean_or_sum_metric_param_name_is_metric_so:metric_median_let_us_verify_params_schema_below:metric:median'. Available columns: order_id, order_date, region, product, channel, units, unit_price, discount_pct, revenue, satisfaction, returned. |
| Q14 | (refuse) | (unsupported) | pass | 5.6 |  |
| Q15 | (refuse) | (unsupported) | pass | 6.3 |  |

## Chosen parameters

- **Q01** `None` {}
- **Q02** `None` {}
- **Q03** `correlation` {"columns": ["discount_pct", "revenue"], "method": "pearson"}
- **Q04** `None` {}
- **Q05** `find_outliers` {"column": "revenue", "multiplier": 1.5}
- **Q06** `value_counts_table` {"column": "channel"}
- **Q07** `value_counts_table` {"column": "product"}
- **Q08** `missing_by_group` {"group_column": "region", "target_column": "satisfaction"}
- **Q09** `correlation` {"columns": ["discount_pct", "units"], "method": "pearson"}
- **Q10** `None` {}
- **Q11** `None` {}
- **Q12** `correlation` {"columns": ["units", "revenue"], "method": "pearson"}
- **Q13** `None` {}
- **Q14** `(unsupported)` {}
- **Q15** `(unsupported)` {}
