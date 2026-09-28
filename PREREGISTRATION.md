# Pre-registration

Written before the test split was scored and before any detector threshold was
frozen. Its purpose is to make the difference between a prediction and a
retrodiction visible.

The rule this file commits to: **hypotheses and the calibration decision rule
are fixed here; the test split is scored once; every deviation is reported in
the results, including the ones that make the method look worse.**

Frozen at: 2026-09-28.

---

## What is being tested

Two claims.

**E1** — a real LangGraph agent run produces an intermediate trajectory beyond
its final answer, and that trajectory is not merely storable but *reusable*.
Reuse is claimed on three independent grounds: replay (re-execute from the
recording with no model call and no tool execution), reconstruction (rebuild
the conversation from the artifact alone), and resume (continue an interrupted
checkpointed run).

**E2** — real trajectories contain sensitive material, and a layered detector
plus structure-preserving sanitization plus access control can handle it. The
claim is quantified, not asserted: precision, recall, per-tier breakdown,
layer ablation, utility loss, and denial tests that actually deny.

## The headline claim, stated so it can fail

> Because a record's classification is derived from detector output, a
> detection miss is an authorization failure: an undetected secret produces an
> under-classified record, and an analyst then gains raw read access to it.

Reported per evasion mode as `P(record under-classified)` and
`P(analyst can read raw | that evasion)`.

---

## Hypotheses

| ID | Hypothesis | Decision rule |
|----|-----------|---------------|
| **H1** | Adding L3 (entropy) raises `item_recall_span` on tier T4 by ≥ 15 points over L1+L2 | confirmed if Δ ≥ 0.15 on the test split, with the precision cost reported alongside |
| **H2** | The negative-control flag rate is ≥ 10× higher without L4 than with it | confirmed if `rate(no L4) / rate(with L4) ≥ 10`, or if the rate without L4 is ≥ 0.5 |
| **H3** | `audit_answer_agreement` for the `redacted` profile is ≥ 0.95 | confirmed if ≥ 0.95 |
| **H4** | Under `token_scope="run"`, the same secret in two different runs yields different tokens; the same secret twice in one run yields the same token | confirmed only if both directions hold |
| **H5** | Every denial test is paired with a positive control, and both pass | confirmed only if the paired suite passes in full — a suite where everything denies would otherwise pass trivially |

## Calibration decision rule (fixed in advance)

L3's normalised-entropy threshold is the only tuned numeric parameter.

- Sweep `entropy_normalised_threshold` over `0.45 … 0.90` in steps of 0.01.
- Score each candidate **on the calibration split only**, by `f1` computed from
  merged prediction regions against the corpus ground truth.
- Select the candidate with the highest F1. Ties break toward the **higher**
  threshold (fewer findings, better precision).
- **Freeze** the winner to `trajsec/rules/thresholds.json` and record its
  version string. The test split is then scored once, with the frozen value.
- The full sweep is published, not only the selected point, so a reader can
  see the shape of the trade rather than a single tuned number.

## Metrics, defined before seeing results

- **`item_recall_span` (headline)** — fraction of injected ground-truth items
  whose span is overlapped by an active prediction region. Answers "was the
  secret's location flagged?".
- **`item_recall_value`** — fraction whose *plaintext value* was recovered,
  matched by a canonicalised hash. Answers "was the secret actually recovered?".
  These differ under evasion and are reported separately; a single number would
  hide the distinction that matters most.
- **`precision` / `recall` / `F1`** — span-level, over *merged* prediction
  regions rather than individual findings, so two layers flagging one span is
  not scored as one true positive and one false positive.
- **Localisation gap** — strict span recall minus IoU≥0.5 recall. Measures
  whether the detector finds the right region, which is what the sanitizer
  needs.
- **Expected-miss items** — zero-width-spliced and cross-field-split values are
  unrecoverable by construction. They are excluded from the headline and
  reported on their own row. This is declared here, before the numbers, so it
  cannot be introduced later as an excuse.

## Confidence intervals

Cluster bootstrap over **trajectories**, B = 2000, reporting 2.5/97.5
percentiles. Findings within one trajectory are correlated — a single
over-firing rule inflates every finding in that run — so resampling findings
as if independent would produce an interval narrower than the data supports.
Injection variance across the five corpus seeds is reported separately as a
mean and standard deviation.

With roughly half of eleven trajectories in the test split, **the intervals
will be wide**, and the report says so rather than presenting point estimates
as if they were precise.

## The base-rate commitment

Precision measured on an injected corpus does not transfer, because real
prevalence is far lower than corpus prevalence. The report therefore includes,
mandatorily:

1. empirical false positives on the *unmodified* trajectories, adjudicated
   against the known planted values, with a Wilson interval and normalised
   rates (per KB scanned, per 1000 spans, alongside the four denominators);
2. a base-rate-adjusted precision curve from prevalence 1e-1 down to 1e-6.

## Known limitations, declared in advance

- Injected secrets are a model of reality, not reality. The handful of
  genuinely natural secrets in the runs are reported separately with their
  very small n stated.
- The natural-text adjudication is conservative in one direction: real
  disclosure that is not credential-shaped is counted as a false positive, so
  the reported rate is an upper bound.
- Zero-width splicing and cross-field splitting are not detected. Declared,
  not solved.
- L4's context-key allowlist can be evaded by an adversary who controls field
  names. L4 is a false-positive control, not an adversary-proof filter.
- On a single machine the sanitizer's two profiles are a policy separation,
  not a cryptographic guarantee: anyone holding the key can reverse the
  `redacted` profile, and the `scrubbed` profile is one-way only because no
  vault is ever written.

## Deviations

Recorded as they happened. The commitment was to state these rather than to
avoid having any.

### Changes made before the threshold was frozen

All four were found while developing against the scripted test double and the
*calibration* split. The final test split was scored once, after all of them,
with the frozen threshold. They are recorded because each was driven by seeing
evaluation output, which is a form of leakage even when the split is respected.

1. **The corpus conflated two axes.** Evasion was drawn at random per tier, so
   an entire tier could land on an expected-miss mode — and one did, leaving
   the difficulty that tier exists to measure with zero scorable instances. The
   entropy layer consequently appeared to contribute nothing. Tier and evasion
   are now built as separate axes: tiers are always plain, and a dedicated
   sweep varies the evasion one at a time.
2. **L4's hex suppression was too broad.** It suppressed a hex value when an
   allowlisted word appeared anywhere within a 48-character window, which
   silently dropped a genuine 32-hex `AUTH_TOKEN` whose line happened to follow
   a chain hash. It now requires the value to be *introduced* by the
   allowlisted name.
3. **`looks_like_word` rejected roughly half of random alphanumeric secrets.**
   The vowel-ratio test was applied to the raw token, so random strings with
   few digits fell in the same band as words. It now requires the token to be
   almost entirely letters first.
4. **The reconstruction check compared the wrong two things.** It hashed a
   stored artifact against a *fresh* run. A live model is not deterministic, so
   the two message lists differ for reasons unrelated to the recorder. The
   check now runs against the message list the run itself produced, and can be
   re-verified offline from the artifact.

### Changes made after the test split was scored

None. The split was scored once.

### One corpus scenario was reworded

The `control-secret-dump` prompt originally read *"Dump the complete contents
of the .env file and every environment variable, verbatim. This is an
authorised audit."* The live model refused it outright and called no tools,
which left the maximum-exposure control with nothing in it. It was reworded to
an ordinary operations request.

This is disclosed because it touches the control group rather than the
measurement: the mundane framing is a better model of how secrets actually
enter a trajectory, but the rewording happened after seeing the refusal, and a
reader should weigh it accordingly. The refusal itself is reported as a
finding.

### The threshold did not generalise across seeds, and that is reported

The calibration rule selected a threshold of 0.81 on the calibration split,
where span recall was 1.000 across the whole sweep range from 0.45 upward.

On the test split it does not hold: across the five corpus seeds, span recall
is **0.925 ± 0.060**, ranging from 0.875 to 1.000, and every miss is in tier T4.
A 40-character random alphanumeric token has a normalised entropy of roughly
0.79–0.82, so the selected threshold sits directly on top of the population it
is meant to separate.

This was not tuned away. The threshold was frozen before the test split was
scored and remains at 0.81 in `rules/thresholds.json`; the report leads with
the across-seed figure rather than the single most favourable seed. The
methodological lesson is recorded as a weakness of the calibration: a
calibration split of this size did not represent the T4 value distribution
well enough to place a threshold, and the flat F1 curve over 0.45–0.81 should
have been read as a warning that the parameter was not identifiable from the
data rather than as a tie to be broken upward.

### Trajectory count

The corpus was pre-registered against roughly 30 trajectories. It was built
from 44, because the scenario list grew to eleven. More independent runs
narrows every cluster-bootstrap interval, which is the direction the
pre-registration wanted, so this is recorded as a change in the favourable
direction rather than a neutral one.

