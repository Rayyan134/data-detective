# Investigator baseline

- Date: 2026-11-13
- Model: `gemini-3.1-flash-lite`
- Dataset: `data/sample_sales.csv`
- Mode: planning only (tool selection and parameters scored)
- Score: **11 / 15**

| ID | Expected tool | Chosen tool | Verdict | Seconds | Note |
|---|---|---|---|---|---|
| Q01 | group_comparison | group_comparison | pass | 6.6 |  |
| Q02 | group_comparison | group_comparison | fail | 4.2 | parameters differ from {'group_column': 'product', 'value_column': 'revenue', 'metric': 'sum'} |
| Q03 | correlation | correlation | pass | 4.8 |  |
| Q04 | trend_over_time | trend_over_time | pass | 1.7 |  |
| Q05 | find_outliers | find_outliers | pass | 1.2 |  |
| Q06 | value_counts_table | value_counts_table | pass | 5.5 |  |
| Q07 | value_counts_table | value_counts_table | fail | 4.8 | parameters differ from {'column': 'product', 'top_n': 10} |
| Q08 | missing_by_group | missing_by_group | pass | 1.3 |  |
| Q09 | correlation | correlation | pass | 2.4 |  |
| Q10 | group_comparison | group_comparison | pass | 2.7 |  |
| Q11 | trend_over_time | trend_over_time | pass | 1.3 |  |
| Q12 | correlation | correlation | pass | 1.6 |  |
| Q13 | group_comparison | group_comparison | fail | 2.8 | parameters differ from {'group_column': 'channel', 'value_column': 'revenue', 'metric': 'median'} |
| Q14 | (refuse) | (unsupported) | pass | 4.3 |  |
| Q15 | (refuse) | (none) | fail | 6.1 | expected an explicit refusal |

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
- **Q14** `(unsupported)` {}
- **Q15** `None` {}
