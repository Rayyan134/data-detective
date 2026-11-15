# Investigator evaluation

- Date: 2026-11-15
- Model: `gemini-3.1-flash-lite`
- Dataset: `data/sample_sales.csv`
- Mode: planning only (+ interpretation on the first 3 questions)

> **Correction (Day 4).** Q07 is recorded as a parameter failure because the plan
> chose `top_n: 3`. That expectation was over-strict: `top_n` only caps how many
> categories are shown and does not change the answer when there are three
> products. With the expectation relaxed to `{"column": "product"}`, Q07 passes
> and the recorded run becomes **12/15 plan (80%)**, **85% parameters**. The
> three remaining failures (Q02, Q13, Q15) are genuine and are analysed in
> `eval/evaluation.md`.

## Metrics

| Metric | Value |
|---|---|
| Questions | 15 |
| Plan accuracy (tool + parameters, or correct refusal) | 73% |
| Tool accuracy | 93% |
| Parameter accuracy | 77% |
| Refusal accuracy | 50% |
| Answer accuracy (check applied to computed findings) | 93% |
| Faithfulness (no unverified numbers) | n/a |
| Normalised parameters (plans that needed a repair) | 0 |

## Latency and tokens

| Metric | Value |
|---|---|
| Latency mean | 5.72 s |
| Latency p50 | 4.08 s |
| Latency p95 | 15.54 s |
| Latency max | 19.54 s |
| Prompt tokens | 22,417 |
| Output tokens | 1,869 |
| Thinking tokens | 0 |
| Total tokens | 24,286 |
| Cost | free tier: $0 marginal cost (set `GEMINI_PRICE_INPUT_PER_MTOK` / `GEMINI_PRICE_OUTPUT_PER_MTOK` for a paid-tier estimate) |

## Per question

| ID | Expected tool | Chosen tool | Tool | Params | Answer | Plan | Seconds | Note |
|---|---|---|---|---|---|---|---|---|
| Q01 | group_comparison | group_comparison | y | y | y | y | 1.3 |  |
| Q02 | group_comparison | group_comparison | y | n | y | n | 19.5 | parameters differ from the expected plan |
| Q03 | correlation | correlation | y | y | y | y | 11.3 |  |
| Q04 | trend_over_time | trend_over_time | y | y | y | y | 1.7 |  |
| Q05 | find_outliers | find_outliers | y | y | y | y | 1.0 |  |
| Q06 | value_counts_table | value_counts_table | y | y | y | y | 13.8 |  |
| Q07 | value_counts_table | value_counts_table | y | n | y | n | 4.6 | parameters differ from the expected plan |
| Q08 | missing_by_group | missing_by_group | y | y | y | y | 1.5 |  |
| Q09 | correlation | correlation | y | y | y | y | 1.5 |  |
| Q10 | group_comparison | group_comparison | y | y | y | y | 7.5 |  |
| Q11 | trend_over_time | trend_over_time | y | y | y | y | 4.1 |  |
| Q12 | correlation | correlation | y | y | y | y | 7.0 |  |
| Q13 | group_comparison | group_comparison | y | n | y | n | 4.1 | parameters differ from the expected plan |
| Q14 | (refuse) | (none) | - | - | y | y | 1.8 |  |
| Q15 | (refuse) | group_comparison | - | - | n | n | 5.2 | expected an explicit refusal |

## Chosen parameters

- **Q01** `group_comparison` {"group_column": "region", "value_column": "revenue"}
- **Q02** `group_comparison` {"group_column": "product", "value_column": "revenue"}
- **Q03** `correlation` {"columns": ["discount_pct", "revenue"], "method": "pearson"}
- **Q04** `trend_over_time` {"date_column": "order_date", "value_column": "revenue"}
- **Q05** `find_outliers` {"column": "revenue"}
- **Q06** `value_counts_table` {"column": "channel"}
- **Q07** `value_counts_table` {"column": "product", "top_n": 3}
- **Q08** `missing_by_group` {"group_column": "region", "target_column": "satisfaction"}
- **Q09** `correlation` {"columns": ["discount_pct", "units"], "method": "pearson"}
- **Q10** `group_comparison` {"group_column": "product", "value_column": "satisfaction"}
- **Q11** `trend_over_time` {"date_column": "order_date", "value_column": "revenue"}
- **Q12** `correlation` {"columns": ["units", "revenue"], "method": "pearson"}
- **Q13** `group_comparison` {"group_column": "channel", "value_column": "revenue"}
- **Q14** (no step)
- **Q15** `group_comparison` {"group_column": "channel", "value_column": "revenue"}

## Faithfulness pass (interpretation on 3 questions)

The narrative stage was also run on these questions. A question is faithful when the answer contains no number that could not be matched to a computed result.

| ID | Faithful | Unverified numbers | Seconds |
|---|---|---|---|
| Q01 | y | - | 20.3 |
| Q02 | y | - | 5.2 |
| Q03 | y | - | 43.7 |
