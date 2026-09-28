# Agent trajectory logging and trajectory security

Two experiments over one artifact. The first builds a real LangGraph agent and records its execution trajectory; the second takes those recordings and attacks them, asking what a trajectory leaks, whether layered detection can find it, whether structure-preserving sanitization can remove it without destroying the artifact's value, and whether the access control around it actually denies.

Every number below is looked up from a generated artifact rather than written by hand, and the detector's one tuned threshold was calibrated on a held-out split and frozen before the test split was scored (see `PREREGISTRATION.md`).

## Pre-registered hypotheses

| id | claim | observed | verdict |
| --- | --- | --- | --- |
| H1 | L3 raises span recall on the novel-format tier by >= 15 points over L1+L2 | delta = +0.1250 (T4 recall 0.875 -> 1.000), precision cost +0.3696 | refuted |
| H2 | The negative-control flag rate falls by >= 10x once L4 is enabled | with L4 0.000, without L4 1.000 | confirmed |
| H3 | audit_answer_agreement for the redacted profile is >= 0.95 | 1.0 | confirmed |
| H4 | under token_scope=run, tokens correlate within a run and not across runs | {"scope": "run", "within_run_stable": true, "across_run_distinct": true, "run_a": "k3vu3c-Ts06y2ic", "run_b": "m2es3k-Rz54m2vy"} | confirmed |
| H5 | every denial is paired with a passing positive control | 22/22 cases, unpaired [] | confirmed |

## Experiment 1 - real agent trajectory logging

A LangGraph agent with three local tools, driven by a live model over an OpenAI-compatible endpoint. Two independent instrumentation sources feed one trajectory: LangChain callbacks supply token usage, exact model timings and the prompt as sent; the graph stream supplies node transitions and the routing decision at each turn. Neither is sufficient alone -- callbacks cannot see which edge was taken, the stream carries no usage -- so the recorder correlates them by run id and node.

Reuse is claimed on three independent grounds rather than on the existence of a JSON file:

1. **Replay.** The recorded model turns and tool observations are served back, with no API call and no filesystem access, and the replayed run is compared against the original call by call.
2. **Reconstruction.** The conversation is rebuilt from the artifact alone and its canonical digest compared with the live message list.
3. **Resume.** With a SQLite checkpointer the run is interrupted before its tool call and continued, and the resumed tool calls are checked to link back to the span that requested them.

### Measured

| metric | value |
| --- | --- |
| runs | 44 |
| scenarios | 11 |
| steps per run (mean) | 12.59 |
| model calls | 144 |
| tool calls | 222 |
| tool errors | 26 |
| tokens (prompt / completion) | 214,894 / 27,644 |
| model time share | 0.9744 |
| tool time share | 0.0256 |
| max call depth | 12 |
| hash chain pass rate | 1.0 |

Estimated cost: **0.037825 USD** (upper bound; all prompt tokens billed at the cache-miss rate).


<details><summary>Full per-scenario table</summary>

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


</details>

---

## Experiment 2 - trajectory security

The input is the trajectories Experiment 1 actually produced, with synthetic secrets injected at known positions to build a labelled corpus. All credentials anywhere in this project are synthetic; the live model key is never recorded, only a fingerprint of it.

### Headline

| metric | value | 95% CI |
| --- | --- | --- |
| item recall (span flagged) | 1.0 | [1.0, 1.0] |
| item recall (value recovered) | 0.875 | [0.875, 0.875] |
| precision (span, merged regions) | 0.9317 | [0.9078, 0.947] |
| recall (span) | 0.4425 | [0.3713, 0.5426] |
| localisation gap (strict - IoU) | 0.0648 | - |
| expected-miss items | 34 of 170 | - |

Intervals are a cluster bootstrap over **22 trajectories**, not over findings: findings inside one run are correlated, so resampling them as if independent would report an interval far narrower than the data supports. With this few trajectories the intervals are wide, and that is stated rather than smoothed over.


**Stability across corpus seeds.** The row above is the primary seed, which is the most favourable one. Across all five:

| metric | mean | std | min | max |
| --- | --- | --- | --- | --- |
| item_recall_span | 0.925 | 0.0596 | 0.875 | 1.0 |
| item_recall_value | 0.8 | 0.0596 | 0.75 | 0.875 |
| precision | 0.9333 | 0.0067 | 0.926 | 0.944 |
| recall | 0.4376 | 0.0039 | 0.4344 | 0.4425 |
| f1 | 0.5958 | 0.0036 | 0.5923 | 0.6 |

The entire spread comes from one tier. T4 -- a long token with no prefix and no sensitive field name -- is reachable only by entropy scoring, and the calibrated threshold sits right at the entropy of that population. On any given seed roughly half its instances fall below the line. That is the honest characterisation of the entropy layer: it is the only thing that reaches this case, and it does so unreliably. The threshold was frozen before the test split was scored and was not adjusted afterwards, so this is reported as a measured weakness rather than tuned away.


The two recall figures answer different questions. *Flagged* asks whether the detector marked where the secret sits; *recovered* asks whether it produced the secret's actual value. Evasion separates them, which is the point of reporting both.

### Per tier

| tier | n | flagged | recovered | span recall | value recall | what it tests |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | 102 | 102 | 51 | 1.0 | 0.5 | format-obvious: a prefixed API key in an environment-style line |
| T2 | 17 | 17 | 17 | 1.0 | 1.0 | format-ambiguous: 32 hex under a sensitive key name (L1 has no rule, L2 does) |
| T3 | 17 | 17 | 17 | 1.0 | 1.0 | structurally nested: a DSN password inside a connection string |
| T4 | 17 | 17 | 17 | 1.0 | 1.0 | novel format: a long token with no prefix and no sensitive key name, so only entropy scoring can reach it |
| T6 | 17 | 17 | 17 | 1.0 | 1.0 | tainted: a secret arriving through untrusted content that also carries an injection |

### Per evasion

| evasion | n | flagged | recovered | span recall | value recall |
| --- | --- | --- | --- | --- | --- |
| base64 | 17 | 17 | 0 | 1.0 | 0.0 |
| percent | 17 | 17 | 17 | 1.0 | 1.0 |
| plain | 102 | 102 | 102 | 1.0 | 1.0 |
| split | 17 | 17 | 0 | 1.0 | 0.0 |
| zerowidth | 17 | 17 | 0 | 1.0 | 0.0 |

Zero-width splicing and cross-field splitting are declared expected misses: the plaintext is not present in any single span, so no value-level detector can recover it. They are excluded from the headline and shown here instead.

### Layer ablation

| configuration | span recall | value recall | precision | recall | F1 | dRecall | dPrecision | neg-control FP rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1_only | 0.625 | 0.5 | 1.0 | 0.3427 | 0.5105 | None | None | 0.0 |
| L1+L2 | 0.875 | 0.75 | 0.9411 | 0.4344 | 0.5944 | 0.0917 | -0.0589 | 0.0 |
| L1+L2+L3 | 1.0 | 0.875 | 0.5715 | 0.4425 | 0.4988 | 0.0081 | -0.3696 | 1.0 |
| full (L4 added) | 1.0 | 0.875 | 0.9317 | 0.4425 | 0.6 | 0.0 | 0.3602 | 0.0 |

This table is the honest centre of the experiment. The entropy layer buys a small amount of span recall and costs a large amount of precision, and is only worth keeping because the allowlist layer cleans up after it. Reporting the layer as a success on its own would be false; reporting it as useless would miss that it is the only thing that reaches a secret with no prefix and no sensitive field name.

### False positives on unmodified trajectories

Precision measured on an injected corpus is optimistic by construction, because every injected positive is a true positive. This section instead scans the trajectories exactly as they were recorded.

| metric | value |
| --- | --- |
| trajectories scanned | 44 |
| findings on natural text | 2612 |
| adjudicated real | 2559 |
| adjudicated false positive | 53 |
| false positive rate per finding | 0.0203 |
| Wilson 95% interval | [0.0155, 0.0264] |
| false positives per KB scanned | 0.058 |
| false positives per 1000 spans | 4.2161 |

Denominators: `{'findings': 2612, 'spans': 12571, 'kilobytes': 913.7, 'trajectories': 44}`


> Adjudicated against the known planted fixtures only. Real disclosure that is not credential-shaped is counted as a false positive, so this rate is an upper bound, not an estimate.


<details><summary>Example false positives</summary>

| layer | type | pointer | preview | chars | rule |
| --- | --- | --- | --- | --- | --- |
| L2 | SECRET_DSN_PASSWORD | /outcome/final_answer | *** | 3 | l2.dsn_password@l2-2026-09-28 |
| L2 | SECRET_DSN_PASSWORD | /steps/18/llm/completion | *** | 3 | l2.dsn_password@l2-2026-09-28 |
| L2 | SECRET_DSN_PASSWORD | /steps/20/state_delta/final_answer | *** | 3 | l2.dsn_password@l2-2026-09-28 |
| L3 | SECRET_ENTROPIC_TOKEN | /steps/21/tool/result | 7f3a...d20 | 40 | l3.entropy.hex@calibrated-0.81 |
| L3 | SECRET_ENTROPIC_TOKEN | /steps/23/llm/messages_in/16/content | 7f3a...d20 | 40 | l3.entropy.hex@calibrated-0.81 |
| L3 | SECRET_ENTROPIC_TOKEN | /steps/17/tool/result | 7f3a...d20 | 40 | l3.entropy.hex@calibrated-0.81 |
| L3 | SECRET_ENTROPIC_TOKEN | /steps/19/llm/messages_in/14/content | 7f3a...d20 | 40 | l3.entropy.hex@calibrated-0.81 |
| L3 | SECRET_ENTROPIC_TOKEN | /outcome/final_answer | 7f3a...d20 | 40 | l3.entropy.hex@calibrated-0.81 |
| L2 | SECRET_DSN_PASSWORD | /outcome/final_answer | *** | 3 | l2.dsn_password@l2-2026-09-28 |
| L3 | SECRET_ENTROPIC_TOKEN | /steps/3/llm/completion | 7f3a...d20 | 40 | l3.entropy.hex@calibrated-0.81 |
| L2 | SECRET_DSN_PASSWORD | /steps/3/llm/completion | *** | 3 | l2.dsn_password@l2-2026-09-28 |
| L3 | SECRET_ENTROPIC_TOKEN | /steps/5/state_delta/final_answer | 7f3a...d20 | 40 | l3.entropy.hex@calibrated-0.81 |
| L2 | SECRET_DSN_PASSWORD | /steps/5/state_delta/final_answer | *** | 3 | l2.dsn_password@l2-2026-09-28 |
| L3 | SECRET_ENTROPIC_TOKEN | /outcome/final_answer | 7f3a...d20 | 40 | l3.entropy.hex@calibrated-0.81 |
| L2 | SECRET_DSN_PASSWORD | /outcome/final_answer | *** | 3 | l2.dsn_password@l2-2026-09-28 |

</details>

### Base-rate adjusted precision

A detector with excellent corpus precision still has poor precision when what it looks for is rare. Applying the measured true- and false-positive rates at realistic prevalence:

| prevalence | precision |
| --- | --- |
| 1e-01 | 0.41841 |
| 1e-02 | 0.061387 |
| 1e-03 | 0.00644 |
| 1e-04 | 0.000647 |
| 1e-05 | 6.5e-05 |
| 1e-06 | 6e-06 |

At the measured operating point (TPR 0.4425, FPR 0.068342). This curve is the answer to "your precision on an injected corpus is fiction" -- and it agrees.

### Sanitization and utility loss

Replacing a secret is easy; replacing it without destroying what the artifact is for is the actual problem. Four properties are measured per profile.

| profile | tokens | destruction footprint | path preservation | audit agreement | idempotent | re-chained |
| --- | --- | --- | --- | --- | --- | --- |
| redacted | 106 | 0.0594 | 1.0000 | 1.0000 | True | False |
| scrubbed | 106 | 0.0594 | 1.0000 | 1.0000 | True | False |

*Destruction footprint* is bytes rewritten over bytes total: a sanitizer that rewrites a third of the artifact to hide half a percent of bytes is over-sanitizing, which is a security failure with no attacker behind it. *Audit agreement* re-asks a fixed set of auditor questions of the original and the sanitized artifact and counts how many answers survive -- one number that catches both under- and over-redaction.


Length preservation by token form: value-derived tokens 50 (all length-preserving: True), typed placeholders 56. Placeholders exist because a token derived from a low-entropy password would be enumerable by anyone holding the key; correlation is given up on purpose for those values.


**Structural replay only.** The sanitized artifact loads, validates and re-chains, but its tool observations have been replaced, so re-executing the tools from it is not possible and is not claimed.

### Access control

A record's classification is the lattice join of the classifications of the finding types present in it. Detector output is the ACL input, which is what makes a detection miss an authorization failure rather than a missed alert.


The adversarial suite runs **22 cases** with **22 passing**. Every denial is paired with a positive control that must be allowed, because a suite in which everything denies would pass even if the store denied everything. Unpaired denials: `none`.


| case | expectation | outcome | passed | detail |
| --- | --- | --- | --- | --- |
| N1_idor_cross_tenant | deny | deny | True | tenant_mismatch (policy.read_payload_sanitized.tenant) |
| N1p_owner_reads_own | allow | allow | True |  |
| N2_analyst_raw | deny | deny | True | role_not_permitted (policy.raw.owner_only) |
| N2p_owner_raw_with_purpose | allow | allow | True |  |
| N3_clearance_escalation | deny | deny | True | insufficient_clearance (policy.raw.clearance) |
| N3p_owner_with_clearance | allow | allow | True |  |
| N4_purpose_violation | deny | deny | True | purpose_not_permitted (policy.raw.purpose) |
| N4p_purpose_debug | allow | allow | True |  |
| N5a_auditor_replay | deny | deny | True | role_not_permitted (policy.replay.owner_only) |
| N5b_auditor_delete | deny | deny | True | append_only_store (policy.delete.deny_all) |
| N5p_auditor_reads_audit | allow | allow | True |  |
| N5bp_auditor_reads_sanitized | allow | allow | True |  |
| N6_forged_session | deny | deny | True | invalid_session (policy.session.signature) |
| N6p_valid_session | allow | allow | True |  |
| N7_unknown_action | deny | deny | True | unknown_action (policy.fail_closed) |
| N7p_defined_action | allow | allow | True |  |
| N8_service_cross_tenant_write | deny | deny | True | role_not_permitted (policy.write.service_only) |
| N8p_service_own_tenant_write | allow | allow | True |  |
| N9_export_raw_anyone | deny | deny | True | export_raw_never_permitted (policy.export_raw.deny_all) |
| N9p_third_party_published | allow | allow | True |  |
| N10_admin_reads_content | deny | deny | True | role_not_permitted (policy.read_payload_sanitized.role) |
| N10p_admin_manages_policy | allow | allow | True |  |


#### Detection failure becomes authorization failure

| configuration | records | P(under-classified) | P(exported in clear) |
| --- | --- | --- | --- |
| full stack | 180 | 0.6 | 0.6 |
| without L3 | 180 | 1.0 | 1.0 |

*Exported in clear* means the secret's plaintext still appears in the artifact a third party would receive -- not that someone could read something sensitive, but that the vendor is handed the credential. The mechanism is demonstrated by ablating a layer and watching the cascade appear where it was absent, rather than by asserting it.


Note the different denominator. This table aggregates every corpus seed and both splits (180 records), so it is deliberately more pessimistic than the headline, which reports one seed's test split. The gap between them is the same T4 instability described above: the single-seed figure is the favourable case, and this one is what an operator would actually face.


#### Audit log tamper detection

| operation | detected | expected |
| --- | --- | --- |
| MODIFY_FIELD | True | True |
| DELETE_ENTRY | True | True |
| REORDER | True | True |
| TRUNCATE_TAIL | True | True |
| REPLAY_DUPLICATE | True | True |
| FORGE_APPEND | False | False |

Detection rate **5/6 = 0.8333**. Detects tampering by a principal that can write records but not the anchor. Does not detect a forged append, and does not detect an attacker who rewrites the chain from genesis and the anchor together.


#### Token scope

A globally stable token is itself a leak: anyone who can read two tenants' trajectories could confirm that the same credential appears in both. Scoping the token domain is how that is controlled rather than assumed away.


```json
{
  "run_scope": {
    "scope": "run",
    "within_run_stable": true,
    "across_run_distinct": true,
    "run_a": "k3vu3c-Ts06y2ic",
    "run_b": "m2es3k-Rz54m2vy"
  },
  "global_scope": {
    "scope": "global",
    "across_run_identical": true,
    "note": "global scope is what creates the cross-tenant confirmation oracle; it is not the default"
  },
  "confirmed": true
}
```

---

## Limitations

- **Injected secrets are a model of reality, not reality.** They are tiered and in-context, but a reviewer is right to discount them.
- **The trajectory count bounds every interval.** The bootstrap is clustered over runs, so the width of every confidence interval is set by how many independent runs exist, not by how many secrets. More runs is the cheapest available improvement to this work.
- **Natural-text adjudication is conservative in one direction.** A finding counts as real only if its value is a known planted credential, so genuine disclosure that is not credential-shaped is counted as a false positive.
- **Zero-width splicing and cross-field splitting are not detected.** Declared in the pre-registration, not discovered afterwards.
- **The allowlist can be evaded by an adversary who controls field names.** Keys named `hash` or `build_id` are suppressed by design; a secret hidden under one would be missed. The allowlist is a false-positive control, not an adversarial filter.
- **On one machine the sanitizer's two profiles are a policy separation, not a cryptographic guarantee.** Anyone holding the key can reverse the reversible profile, and the one-way profile is one-way only because no vault is written. DPAPI protects against another user of the machine, not against the same user or an administrator.
- **The tamper claim is bounded.** A forged append is not detectable by the chain alone, and an attacker who can rewrite both the chain and its anchor defeats it entirely. The measured detection rate includes that gap rather than hiding it.
- **Utility metrics measure structural utility, not human utility.** Audit agreement is a proxy for whether an analyst could still do their job.
- **Single annotator.** The natural-text adjudication was not independently reproduced, so no inter-rater agreement figure is available.

## Reproduction

```bash

python run_experiment.py --e1        # live agent runs -> trajectories
python run_experiment.py --e2        # detection, sanitization, access control
python run_experiment.py --all       # both, then this report
python -m pytest trajsec/tests/ -q   # the security assertions
```

| parameter | value |
| --- | --- |
| thresholds version | calibrated-0.81 |
| policy version | trajsec-policy-1.0 |
| rule files fingerprint | 97440d9328086b2fb68b22b0686de91ba8eeaa81811487d5643e372c5b724e48 |
| corpus seeds | [1337, 2024, 7, 99, 4242] |
| key custody | dpapi-file |
| key material absent from artifacts | True |

This project is not a git repository, so content hashes are the reproducibility anchor: `reports/artifacts/MANIFEST.json` records the digest of every rule file, every input trajectory and every output artifact, so a reader can verify that the figures above came from exactly those inputs.


### Sandbox declaration


```json
{
  "all_secrets_synthetic": true,
  "real_credentials_recorded": false,
  "note": "every credential in the corpus and the workspace is synthetic; the live model API key is never recorded, only its fingerprint"
}
```
