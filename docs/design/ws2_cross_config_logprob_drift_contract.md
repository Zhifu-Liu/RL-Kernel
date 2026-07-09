# WS2 Cross-Config Logprob Drift Contract

Status: RFC

Tracking issues:

- [#111: WS2 cross-config alignment](https://github.com/RL-Align/RL-Kernel/issues/111)
- [#108: WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/issues/108)

## Motivation

WS2 covers rollout and training paths that use different parallelism strategies, such as
rollout tensor parallelism and training FSDP. The alignment problem is not a single-op
accuracy check. It is end-to-end floating-point drift across tokenizer, masks, serving,
rollout, and training recomputation before any optimizer update.

For PPO, GRPO, and related RL post-training algorithms, the most direct pre-update signal
is selected-token log probability drift. If rollout-side `old_logprobs` and train-side
recomputed log probabilities disagree for the same checkpoint, same token ids, same masks,
and same model version, classify the failure as infrastructure, precision, mask,
tokenizer, or serving-path drift. Do not classify that failure as an algorithm or reward
problem until pre-update logprob alignment is clean.

Aggregate KL-style diagnostics are useful but not sufficient as the primary WS2 contract.
In training-inference mismatch cases, KL estimates can stay flat or fail to expose the
early failure phase, because the first-order issue is token-level rollout-vs-training
probability disagreement before the optimizer update, not necessarily a large aggregate
policy-space shift.

## Scope

This RFC defines what WS2 cross-config alignment measures and how failures are classified.
It does not add a test harness, distributed tests, runtime gates, layer-wise probes, or
distributed fixes.

Out of scope for this document:

- Implementing multi-GPU test infrastructure.
- Adding runtime pass/fail gates.
- Adding automatic layer-wise drift probes.
- Fixing TP, FSDP, SP, cache, mask, tokenizer, or serving-path bugs.
- Defining a second numerical tolerance table.

## Measurement Contract

The primary metric is selected-token logprob drift:

```text
dlogp = train_recomputed_logp - rollout_old_logp
```

Compute `dlogp` only on active response/action tokens. Prompt tokens, padding tokens, and
masked-out response positions are excluded from every aggregate metric.

The comparison must use teacher-forcing scoring on the training side. The scored sequence
is the already-sampled rollout sequence; the training path must not resample or regenerate
tokens for this contract.

The rollout and training values are comparable only when they share the same logical
inputs:

- Same checkpoint and same model version.
- Same input token ids.
- Same selected response/action token ids.
- Same attention mask and action mask.
- Same tokenizer version and tokenization policy.
- Same padding layout semantics, including left-padding or right-padding behavior.
- Same pre-update state, before any optimizer step, weight sync, or policy mutation that
  belongs to the next training step.

If the implementation has explicit position ids, cache-position metadata, sequence ids, or
packed-sequence metadata, those inputs are part of the comparison contract as well.

## Primary Failure Signal

The pass/fail decision starts from `dlogp` over active tokens. Reward, gradnorm,
weightnorm, and update norm are downstream symptoms. They are useful for debugging and
triage, but they are not the primary contract for cross-config alignment.

The zero-update expectation is:

```text
train_recomputed_logp ~= rollout_old_logp
ratio0 ~= 1
approx_kl0 ~= 0
```

The acceptable meaning of `~=` is defined by the WS1 per-dtype numerical threshold table
from [#108](https://github.com/RL-Align/RL-Kernel/issues/108). This RFC defines the
measurement surface and classification rules only.

## Diagnostics

All diagnostics are computed on active response/action tokens only.

| Metric | Definition | Purpose |
| --- | --- | --- |
| `ratio0` | `exp(dlogp)` | Zero-update policy ratio implied by train-vs-rollout logprob drift. |
| `clipfrac0` | Mean indicator that `ratio0` falls outside the configured PPO/GRPO clip range. | Detects whether drift alone would trigger clipping before any update. |
| `approx_kl0` | Masked mean of `exp(dlogp) - 1 - dlogp`. | Zero-update approximate KL implied by logprob drift. |
| `mean_abs_dlogp` | Mean of `abs(dlogp)`. | Average selected-token drift. |
| `p95_abs_dlogp` | 95th percentile of `abs(dlogp)`. | Tail drift below outliers. |
| `p99_abs_dlogp` | 99th percentile of `abs(dlogp)`. | High-tail drift. |
| `max_abs_dlogp` | Maximum of `abs(dlogp)`. | Worst selected-token mismatch. |

When the run is distributed, report optional per-rank versions of the same metrics. The
per-rank view should preserve enough metadata to identify the rollout rank, training rank,
parallelism mode, dtype, padding side, cache mode, and local active-token count for that
rank.

## Tolerance Source

This RFC does not define a separate numerical tolerance table. The single source of truth
for acceptable numerical drift is the per-dtype threshold table owned by
[#108](https://github.com/RL-Align/RL-Kernel/issues/108).

For WS2, acceptable numerical drift means that `max_abs_dlogp` over active
response/action tokens satisfies the WS1 per-dtype threshold from #108. If the #108 table
changes, WS2 inherits that policy without editing this document or maintaining a second
table.

## Tolerance Interpretation and Effect-Based Validation

Numerical tolerances in this RFC are infrastructure contract thresholds, not a universal
statement of algorithmic harmlessness. There is no model-independent scale that proves a
given train-vs-rollout logprob difference is harmless for every algorithm, reward model,
prompt distribution, sequence length, or optimization schedule. Any hand-written threshold
encodes a prior about acceptable numerical error. WS2 therefore does not introduce an
additional algorithmic noise budget, nor does it define a new estimator for tolerable
logprob noise.

The #108 threshold defines whether rollout and training paths are numerically aligned
enough to continue debugging the failure as an algorithmic or reward problem. It does not
prove that all smaller drift is behaviorally irrelevant, and it does not imply that all
larger drift is the only cause of downstream failure.

When downstream model-effect validation is available, such as reward trajectory, train KL,
eval win rate, collapse rate, policy regression tests, or task-specific success metrics,
use it as a severity and root-cause prioritization signal. It must not replace the
pre-update selected-token logprob contract. A run can be numerically out of contract even
if a short downstream run appears healthy, and a run can be numerically in contract while
still failing because of algorithmic tuning, reward hacking, insufficient KL control, or
data issues.

The intended interpretation is:

```text
#108 per-dtype threshold:
    numerical infrastructure contract

selected-token dlogp:
    primary WS2 train-vs-rollout drift surface

downstream model effect:
    practical severity and algorithmic relevance signal

KL / ratio / percentile diagnostics:
    debugging and triage signals, not replacement pass/fail criteria
```

## Drift Source Taxonomy

Before treating train-vs-rollout drift as generic algorithmic noise, WS2 should classify
likely sources of mismatch. At minimum, the following source classes should be considered
separately.

### Arithmetic Schedule Drift

Arithmetic schedule drift comes from different floating-point operation order between
rollout and training. This includes different kernels, fused vs unfused implementations,
compiler-generated graph rewrites, attention implementation differences, matmul epilogue
differences, accumulation dtype differences, and changes introduced by advanced compilers
or graph optimizers.

This class answers the question:

```text
Do rollout and training compute mathematically equivalent expressions using different
floating-point schedules?
```

Examples include:

- Fused attention vs unfused attention.
- Different FlashAttention or SDPA backends.
- Fused RMSNorm or LayerNorm vs decomposed normalization.
- Compiler-reordered graph segments.
- Different matmul epilogues or activation fusion.
- Different accumulation precision in otherwise equivalent kernels.

### Reduction and Collective Drift

Reduction drift comes from operations whose floating-point result depends on reduction
order, parallel topology, or concurrent execution. This includes local reductions,
cross-rank reductions, all-reduce, reduce-scatter, gather/scatter patterns, sharded logits
or loss computation, tensor-parallel collectives, FSDP reductions, and nondeterministic
reduction scheduling.

This class answers the question:

```text
Does the mismatch appear because rollout and training aggregate partial results in
different orders or across different rank topologies?
```

Examples include:

- TP logits produced through a different collective path from the training path.
- FSDP reduce-scatter or all-gather changing accumulation order.
- Per-rank partial reductions with different shard boundaries.
- Loss or logprob reductions performed before vs after cross-rank communication.
- Nondeterministic collective algorithms or concurrent reductions.

### Quantization and Dequantization Drift

Quantization drift comes from representing weights, activations, KV cache, logits, or
intermediate tensors with different quantization policies between rollout and training.
Quantization is not merely a floating-point ordering issue; it introduces representation
noise through scales, zero points, clipping, grouping, calibration, and dequantization
paths.

This class answers the question:

```text
Does the mismatch appear because rollout and training use different numerical
representations or quantization policies?
```

Examples include:

- Rollout uses weight-only quantization while training recomputation uses bf16/fp16
  weights.
- Different quantization group sizes.
- Different activation quantization or KV-cache quantization policy.
- Different scale computation or calibration data.
- Different dequantization placement relative to fused kernels.
- Serving-path quantization that is absent from the training path.

### Logical Input and Metadata Drift

Logical input mismatch must be ruled out before interpreting any result as numerical
drift. This class includes tokenizer version, tokenization policy, attention mask, action
mask, padding side, explicit position ids, cache positions, sequence ids, packed-sequence
metadata, and serving-path request formatting.

This class answers the question:

```text
Are rollout and training actually scoring the same logical sequence under the same masking
and positional semantics?
```

If this class is not clean, the comparison is invalid rather than merely noisy.

## Decision Rule

Use this order when classifying a cross-config failure:

1. If pre-update selected-token logprobs do not match under the same checkpoint, same token
   ids, same masks, and same model version, treat the failure as infrastructure,
   precision, mask, tokenizer, or serving-path drift.
2. If `max_abs_dlogp` violates the #108 threshold but downstream metrics look healthy in a
   short run, keep the issue classified as infrastructure drift. Short-horizon model
   health does not prove the drift is safe.
3. If KL or ratio diagnostics move before gradnorm or update norm moves, treat the failure
   as likely infrastructure or logprob plumbing.
4. If gradnorm or update norm moves first and KL moves later, treat the failure as more
   likely algorithmic tuning, such as learning rate, KL beta, reward scale, or advantage
   outliers.
5. If only some ranks drift, treat the failure as distributed infrastructure until rank
   placement, shard boundaries, collective algorithms, local active-token counts, masks,
   and cache-position issues are ruled out.
6. If reward rises and then collapses while pre-update logprob alignment is clean, treat
   the failure as more likely algorithmic, reward hacking, data-related, or insufficient
   KL constraint.

This classification does not prove root cause by itself. It defines the first branch in
the debugging tree so WS2 bugs do not get misfiled as reward or algorithm regressions
before the zero-update logprob contract is satisfied.

## Layered Ablation Strategy

WS2 should not treat train-vs-rollout mismatch as a single undifferentiated error source.
Later tests should use a layered ablation strategy that changes one source class at a time
whenever the implementation allows it.

The minimum useful ablation structure is:

```text
A0. Fully aligned reference
    Same checkpoint, same dtype policy, same kernels where possible, same reduction
    topology where possible, same quantization policy, same tokenizer, same masks, same
    padding, same cache/position metadata.

A1. Arithmetic-schedule-only mismatch
    Keep logical inputs, reduction topology, and quantization policy aligned. Allow only
    kernel, fusion, compiler, or graph execution differences.

A2. Reduction-topology-only mismatch
    Keep logical inputs, kernel policy, and quantization policy aligned. Allow only
    reduction order, collective topology, sharding, or rank placement differences.

A3. Quantization-only mismatch
    Keep logical inputs, kernel policy, and reduction topology aligned. Allow only
    quantization, dequantization, scale, group size, or representation differences.

A4. Pairwise mismatches
    Enable two mismatch classes at a time:
        arithmetic + reduction
        arithmetic + quantization
        reduction + quantization

A5. Full production mismatch
    Use the real rollout and training configurations, including all production
    differences.
```

Each ablation should collect the same primary and diagnostic metrics:

```text
primary:
    dlogp over active response/action tokens
    max_abs_dlogp

diagnostics:
    mean_abs_dlogp
    p95_abs_dlogp
    p99_abs_dlogp
    ratio0
    clipfrac0
    approx_kl0
    per-rank versions when distributed

metadata:
    dtype
    kernel/backend choices
    fusion/compiler mode
    reduction/collective topology
    quantization policy
    padding side
    cache mode
    position/cache-position metadata
    active-token count
```

When downstream model-effect validation is available, the same ablations should also
record practical training outcomes, for example reward trajectory, training KL, entropy,
clip fraction, update norm, collapse rate, and task-specific evaluation metrics. These
downstream metrics are not the WS2 pass/fail contract, but they help rank which numerical
mismatch class matters most for the workload.

## Ablation Interpretation Rules

Use these rules when reading the ablation matrix:

1. If the fully aligned reference fails, the issue is not a cross-config mismatch yet.
   First debug the base scoring path, masks, tokenizer, position metadata, checkpoint
   identity, or implementation correctness.
2. If a single-source ablation fails the `max_abs_dlogp` contract, that source class is
   sufficient to create unacceptable train-vs-rollout drift under the tested workload. For
   example, if only quantization is misaligned and the run fails, quantization is a
   dominant source candidate for that task and configuration.
3. If all single-source ablations pass, but pairwise or full-production mismatches fail,
   the failure is likely an interaction effect. Identify the minimal failing pair before
   attributing the issue to any single subsystem.
4. If one single-source ablation passes the numerical contract but shows materially worse
   downstream model effect, record it as behaviorally sensitive even if it remains
   numerically in contract. This is a signal that the #108 infrastructure tolerance may be
   sufficient for numerical alignment but not necessarily predictive of algorithmic
   robustness for that workload.
5. If pre-update logprob alignment is clean but downstream training still collapses,
   classify the failure as more likely algorithmic, reward-related, data-related, or
   KL-control-related rather than cross-config numerical drift.

## Follow-Up Test Matrix

Later PRs should implement a systematic matrix around this contract. The minimum planned
coverage is:

- Fully aligned reference configuration.
- Kernel/fusion/compiler-only mismatch.
- Reduction/collective-only mismatch.
- Quantization-only mismatch.
- Arithmetic plus reduction mismatch.
- Arithmetic plus quantization mismatch.
- Reduction plus quantization mismatch.
- Full production rollout-vs-training mismatch.
- Single-process reference vs TP.
- Single-process reference vs FSDP.
- TP vs FSDP.
- SP on vs SP off.
- Batch size 1 vs batch size N.
- Left padding vs right padding.
- Cache on vs cache off.
- fp32, bf16, and fp16.
- Per-rank reporting for each distributed ablation.
- Downstream model-effect recording when the test setup supports short training runs.

The purpose of these tests is not to invent a second tolerance table. The purpose is to
identify which source class, or which interaction between source classes, explains the
observed `dlogp` drift and downstream sensitivity. The test implementation belongs in
later PRs, not in this RFC.
