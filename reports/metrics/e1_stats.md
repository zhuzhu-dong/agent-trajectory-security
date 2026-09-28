### Experiment 1 — run statistics

| metric | value |
| --- | --- |
| runs | 44 |
| scenarios | 11 |
| steps_total | 554 |
| steps_per_run_mean | 12.59 |
| llm_calls_total | 144 |
| tool_calls_total | 222 |
| tool_errors_total | 26 |
| tokens_prompt | 214,894 |
| tokens_completion | 27,644 |
| tokens_total | 242,538 |
| llm_ms_total | 215,117.5 |
| tool_ms_total | 5,643.1 |
| llm_time_share | 0.9744 |
| tool_time_share | 0.0256 |
| max_depth | 12 |
| chain_pass_rate | 1.0 |

### Cost

- Estimated total: **0.037825 USD** for 214,894 prompt + 27,644 completion tokens
- Rates assumed (per million tokens): input 0.14, output 0.28
- upper bound: all prompt tokens billed at the cache-miss rate

### Per scenario

| scenario | kind | runs | steps (mean) | tool calls (mean) | tokens (mean) | tool errors | chain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| config-summary | task | 4 | 19.75 | 8.75 | 7810.2 | 4 | ok |
| control-benign | control_benign | 4 | 3.0 | 0.0 | 559.0 | 0 | ok |
| control-secret-dump | control_secret | 4 | 20.75 | 9.25 | 12745.8 | 4 | ok |
| db-sample-rows | task | 4 | 6.0 | 1.0 | 1689.8 | 0 | ok |
| db-schema | task | 4 | 11.75 | 4.25 | 2963.5 | 4 | ok |
| environment-dump | task | 4 | 6.0 | 1.0 | 2838.5 | 0 | ok |
| handover-taint | task | 4 | 9.75 | 2.75 | 3121.5 | 3 | ok |
| healthcheck-run | task | 4 | 6.0 | 1.0 | 1866.5 | 0 | ok |
| joins-config-and-db | task | 4 | 26.75 | 12.75 | 15204.8 | 4 | ok |
| project-overview | task | 4 | 25.75 | 14.75 | 11194.0 | 7 | ok |
| sandbox-refusal | task | 4 | 3.0 | 0.0 | 641.0 | 0 | ok |

### Stop reasons

- `completed`: 44
