# Surtur — Product Roadmap

> **EAL Surtur** — Epoch's spatial training and capability-preservation runner.
> Post-training that confines the "fire" of parameter optimization to the behavioral periphery (layers 24–27), keeping the model's core logic engine frozen and insulated.
>
> **Document owner:** Charter (Product)
> **Last updated:** 2026-07-08
> **Status:** Draft v1 — pending research-lead sign-off on the two blocking definitions (eval suite + thresholds)

---

## 0. How to read this document

This roadmap is written **problem-first**. Every phase states *why it exists* before *what it contains*. Items are ranked with a cut line — the "Won't (this release)" column is the most useful part. Success is defined by **outcomes that move a number**, not by features shipped.

Charter decides *what* and *why*. Atlas decides *how* (system design). Forge decides *implementation* (build). This document is handed to Atlas after research leads resolve the blocker in §2.

**Conventions**
- **MVP** = the cheapest experiment that tests the riskiest assumption. It is *never* "the full thing, later."
- **Definition of Done (DoD)** = testable Given/When/Then or a checklist, written before build starts.
- **Kill condition** = the explicit signal to stop, so sunk cost can't carry a dead feature to launch.

---

## 1. Vision & north-star

**Vision:** Make capability-preserving post-training a default, cheap, reproducible habit at Epoch — align models on a single GPU without silently burning the core they already earned.

**North-star metric (placeholder — needs sign-off):**
> *Capability retention* — a Surtur run holds core-eval scores within **≤2% of the frozen baseline** while delivering a **measurable alignment gain** versus the unaligned model, executed on **one GPU** at **≤30% the compute** of equivalent full fine-tuning.

| Component | Definition | Current state |
|---|---|---|
| Capability retention | Core-eval score under Surtur ÷ core-eval score of frozen baseline | Undefined suite (blocker) |
| Alignment gain | Alignment-eval score under Surtur − alignment-eval score of unaligned baseline | Undefined suite (blocker) |
| Compute budget | GPU-hours of Surtur run ÷ GPU-hours of full fine-tune for same objective | To be measured in P0 |
| Hardware target | Single consumer/prosumer-class GPU | Stated in README |

**Why a single number matters:** a roadmap without a north-star drifts into "ship more features." This metric is the spine — every phase below either moves it or clears the path to move it.

---

## 2. The problem (and the blocker)

| Field | Content |
|---|---|
| **Problem** | Post-training (SFT/GRPO) backpropagates across the *entire* model, degrading the central logic engine. Epoch researchers want alignment *without* re-earning capabilities they already paid for — and without a GPU cluster. |
| **Evidence it's real** | ⚠️ **Assumption.** The README asserts degradation occurs but cites no metric. **Phase 0 exists specifically to confirm this is true before anything else is built.** |
| **Primary user** | Epoch ML researchers / post-training engineers running alignment jobs. |
| **Secondary stakeholders** | Research leads (own the capability scores), Compute/Infra (GPU budget owner), The maintainer (owns the tool after launch). |
| **Form (confirmed)** | Internal research tool. UI mockups in `/UI` are an *internal dashboard*, not a customer surface. |

### The one blocker that gates the entire roadmap

> **Action (research leads, this week):** specify (1) the **core-capability eval suite** — the "do not regress this" set — and (2) the **retention / alignment thresholds** that define pass vs. fail.

Without these two definitions, **Phase 0 cannot pass or fail**, and every downstream phase is built on an unverifiable premise. This is the single highest-priority item in the document.

---

## 3. Strategic principles (the lines we don't cross)

1. **Preservation is the product.** If a Surtur run improves alignment but quietly degrades the core, it has failed. The core-eval suite is sacred.
2. **Single GPU is the constraint, not a nice-to-have.** Anything that requires a cluster is out of scope for v1.
3. **Reproducibility is non-negotiable.** A number we can't rerun isn't evidence.
4. **Constraint strategy, not new architecture.** Surtur is a *where-gradients-flow* decision. We do not invent optimizers or attention variants.
5. **Say no to good ideas.** The cut lines below are deliberate. A roadmap is what we chose *not* to do.

---

## 4. Phase overview (the arc)

| Phase | Name | Question it answers | Gate to next |
|---|---|---|---|
| **P0** | Validate the core claim | *Does layer-constrained training actually preserve the core AND align?* | Core claim holds within thresholds |
| **P1** | Core runner (engine) | *Can we make this a reusable single-GPU runner?* | Runner completes, retention holds |
| **P2** | Reproducibility (Anchor) | *Can anyone rerun any result, anywhere?* | Every P0/P1 number reruns from one command |
| **P3** | Internal dashboard (UI) | *Can researchers use it day-to-day without CLI heroics?* | Launch → watch → see retention, all in UI |
| **P4** | Adoption & hardening | *Is Surtur the default, not a curiosity?* | Adoption + median retention targets met |

**Dependency chain:** P0 → P1 → P2 → P3 → P4. P2 can run in parallel with late P1. P3 can begin design (Atlas) once P1's runner interface stabilizes, but must not ship before P2's reproducibility foundation exists.

---

## 5. Phase 0 — Validate the core claim

**This is the MVP.** It is an experiment, not a product. If the premise fails, we stop before spending a sprint on an engine nobody needs.

### Problem this phase tests
"We believe confining gradients to layers 24–27 preserves core capabilities while still allowing alignment." That belief is currently unverified.

### MVP definition
One head-to-head experiment:
- **Arm A — Surtur:** layers 24–27 trainable, core frozen (no grad).
- **Arm B — Full fine-tune:** standard SFT/GRPO across all layers (the incumbent method).
- **Arm C — Frozen baseline:** no training (the capability reference).
- **Arm D — Untrained-but-aligned reference** (if available) for alignment gain.

All arms use the **same data, same seed, same objective**.

### Scope
| Priority | Item | Why |
|---|---|---|
| **Must** | Fixed target-layer config (24–27) + frozen-core gradient masking | The actual hypothesis under test |
| **Must** | Core-capability eval suite (defined by research leads) | Without it, "preservation" is unmeasurable |
| **Must** | Alignment eval suite (defined by research leads) | Without it, "alignment" is unmeasurable |
| **Must** | Single shared training pipeline for all arms | Fair comparison; removes confounders |
| **Should** | Per-layer gradient-norm logging | Diagnoses *why* it works or fails |
| **Should** | Ablation: test layers {22–25}, {24–27}, {26–29} | Is 24–27 universal or model-dependent? |
| **Won't** | UI, multi-GPU, novel methods, hyperparameter sweeps | Pure research question, nothing else |

### Acceptance criteria (DoD)
- Given the same data + seed, each arm's run is reproducible from a single command.
- Core-eval: Surtur (Arm A) ≥ Frozen baseline (Arm C) − 2%.
- Alignment-eval: Surtur (Arm A) > Untrained reference (Arm D).
- A results table is committed with config + metric provenance for every arm.
- Ablation (Should) documents whether 24–27 generalizes across layer bands.

### Success metric
Core claim **validated** if retention holds *and* alignment improves. Otherwise see kill condition.

### Kill / revisit if
Core capabilities degrade beyond threshold **and** only full fine-tuning (Arm B) reaches the alignment target → the spatial-constraint premise fails. **Decision point:** pivot (different layer band / different freezing schedule) or stop. Do not proceed to P1 on a failed premise.

### Estimated effort
1 researcher + 1 engineer, ~1–2 weeks including eval-suite definition.

---

## 6. Phase 1 — Core runner (the engine)

**Only starts if P0 passes.** Turn the validated experiment into a reusable, config-driven runner.

### Problem this phase solves
Today (post-P0) the method lives in an experiment script. Researchers can't easily re-run it on new tasks without copy-pasting. We need a real runner.

### MVP definition
A config-driven training runner that accepts a target-layer spec, freezes the core, and runs SFT *or* GRPO on a single GPU, producing checkpoints and logs.

### Scope
| Priority | Item | Why |
|---|---|---|
| **Must** | Config-driven target-layer selection | Reusable across tasks/models |
| **Must** | Frozen-core masking (no-grad on core) | The core mechanism |
| **Must** | Single-GPU optimization path (grad accumulation / CPU offload) | The stated efficiency goal |
| **Must** | SFT **and** GRPO training loops | Both methods named in README |
| **Must** | Checkpoint + clean resume | Researchers can't babysit runs |
| **Should** | Live gradient-norm + capability-trend logging | Proves the shield works mid-run |
| **Should** | Guardrail: hard error if a frozen layer receives grad | Prevents silent premise break |
| **Should** | Config schema validation | Fail fast on bad input |
| **Won't** | Multi-GPU / distributed training | Out of scope for single-GPU goal |
| **Won't** | Novel optimizers / custom attention | Not the bet being tested |
| **Won't** | Auto layer-band selection | Research question, defer to P4 |

### Acceptance criteria (DoD)
- Given a config naming layers 24–27, a probe confirms *only* those layers have non-zero gradients.
- Given a valid config, a full SFT and a full GRPO run each complete on one GPU within the compute budget.
- Given a run interrupted at 60%, resume restores state and continues to completion.
- Core retention under the runner matches the P0 validated result (regression guard).

### Success metric
Runner reproduces P0's retention/alignment numbers; median single-GPU run cost ≤ 30% of full fine-tune equivalent.

### Kill / revisit if
The runner cannot hit the compute budget on real tasks, or retention regresses vs. P0 — indicates the method doesn't generalize beyond the P0 toy setup.

### Estimated effort
1–2 engineers, ~3–4 weeks.

---

## 7. Phase 2 — Reproducibility (Anchor)

**Runs in parallel with late P1.** Makes every number rerunnable — what separates research from a one-off script.

### Problem this phase solves
Results that can't be rerun aren't evidence. As the runner matures, we accumulate claims that must survive scrutiny and time.

### MVP definition
A pinned, versioned, one-command reproduction path for every published result.

### Scope
| Priority | Item | Why |
|---|---|---|
| **Must** | Pinned environment (lockfile or container image) | "Works on my GPU" is not science |
| **Must** | Seed control + dataset versioning (hash-pinned) | Controls the two biggest confounders |
| **Must** | Run command + config hash captured per result | Ties every number to its inputs |
| **Must** | Results registry (run → config → metrics table) | Lets researchers compare & audit |
| **Should** | Deterministic dataloader / sampler ordering | Closes the last reproducibility gap |
| **Should** | One-command "reproduce result #ID" entrypoint | Turns reproducibility into a button |
| **Won't** | CI that auto-runs all configs | Over-engineering for internal scale |
| **Won't** | Public result portal | Internal only |

### Acceptance criteria (DoD)
- Given a result ID from the registry, a fresh environment reproduces the metrics within tolerance from one command.
- Every metric in the registry links to its exact config + data version + seed.
- A new researcher can reproduce a P0 result on a different machine without author help.

### Success metric
100% of "published" Surtur results are reproducible from the registry; time-to-reproduce < 1 hour for a new researcher.

### Kill / revisit if
Reproducibility consistently requires author intervention — signals the env or data isn't actually pinned.

### Estimated effort
1 engineer, ~2 weeks (overlaps P1).

---

## 8. Phase 3 — Internal dashboard (the UI)

**Design (Atlas) can start when P1's runner interface stabilizes; ship only after P2 foundation exists.** The `/UI` mockups are the target surface.

### Problem this phase solves
Researchers shouldn't need CLI heroics to launch a run, watch it, and confirm the core survived. The dashboard is the daily surface that makes Surtur *usable*, not just *possible*.

### MVP definition
A web dashboard where a researcher submits a run config, watches live training, and sees the capability-preservation delta versus the frozen baseline — without leaving the page.

### Scope
| Priority | Item | Why |
|---|---|---|
| **Must** | Submit a run config (target layers, method, dataset) | The launch action |
| **Must** | Live training-metrics view (loss, grad norm, step) | Researchers need to watch |
| **Must** | Capability-preservation comparison (frozen baseline vs. Surtur) | The whole point, made visible |
| **Must** | Run history + status (queued / running / done / failed) | Researchers iterate across runs |
| **Must** | Authentication via existing Epoch SSO | Internal tool, reuse, don't build auth |
| **Should** | Per-layer gradient visualization | Diagnoses misconfiguration fast |
| **Should** | Side-by-side comparison of two runs | A/B alignment strategies |
| **Should** | Artifact links to the P2 results registry | Closes the loop: UI → reproducible number |
| **Won't** | Multi-tenant auth / external sharing | Internal only, per product decision |
| **Won't** | Run scheduling / queueing system | Not needed at Epoch's scale yet |
| **Won't** | Billing / quotas / cost center chargeback | Out of scope for internal tool |

### Acceptance criteria (DoD)
- Given a researcher authenticated via SSO, they launch a run from the UI.
- Given a running run, live metrics render within a bounded latency (e.g., ≤10s staleness).
- Given a completed run, the dashboard shows core-retention delta vs. baseline without leaving the page.
- Given a run ID, the dashboard links to its P2 registry entry (config + metrics + reproduction command).

### Success metric
≥ 80% of Epoch alignment runs launched via the dashboard (not raw CLI) within one quarter of launch; zero "I didn't know the core regressed" incidents (the comparison is always shown).

### Kill / revisit if
Researchers keep using the CLI despite the dashboard — signals the UI solves a problem they don't have; rethink the surface.

### Estimated effort
1–2 engineers (frontend + backend bridging runner), ~4–6 weeks including Atlas design.

---

## 9. Phase 4 — Adoption & hardening

**Starts after P3 ships.** Move Surtur from "available" to "default."

### Problem this phase solves
A tool nobody uses is waste. Adoption is the real test of whether the roadmap delivered value.

### Scope
| Priority | Item | Why |
|---|---|---|
| **Must** | Template configs for common Epoch alignment jobs | Lowers activation energy |
| **Must** | One end-to-end worked example (doc) | Onboarding in practice |
| **Must** | Track adoption + median retention | The outcome metrics |
| **Should** | Auto layer-band suggestion from a quick probe run | Reduces "which layers?" friction |
| **Should** | Integration with Epoch's existing eval/reporting | Meets researchers where they are |
| **Won't** | External community features | Internal tool |
| **Won't** | Custom model architectures | Constraint strategy, not architecture |

### Acceptance criteria (DoD)
- A new researcher completes a Surtur alignment run from template to reported result in < 1 day (onboarding validated).
- Adoption metric meets target (see below) for two consecutive months.
- Median capability retention across all production Surtur runs ≥ the P0 threshold.

### Success metrics (outcome, not output)
- **Adoption:** % of Epoch alignment runs using Surtur (target: ≥ 60% within 2 quarters of P3).
- **Quality:** median capability retention across runs (target: ≥ P0 threshold).
- **Efficiency:** median GPU-cost ratio vs. full fine-tune (target: ≤ 30%).

### Kill / revisit if
After active onboarding effort, adoption stays < 20% — the tool may solve a problem too narrow, or the workflow doesn't fit. Re-interview users before investing further.

---

## 10. Cross-cutting risks & open questions

| Risk / Question | Impact | Mitigation |
|---|---|---|
| Core-capability eval suite undefined | **Blocks P0 entirely** | Research leads define it this week (§2 blocker) |
| Retention/alignment thresholds undefined | **Blocks P0 pass/fail** | Same blocker; sign off before build |
| Is 24–27 universal or model-dependent? | Affects generalizability | P0 ablation across layer bands |
| Degradation claim may be false | Invalidates premise | P0 kill condition; stop if failed |
| Single-GPU ceiling on large jobs | Some tasks won't fit | Document where Surtur refuses to apply |
| "Core" vs "periphery" boundary fuzzy | Masks may be wrong | P0 gradient-norm logging informs boundary |
| Researcher adoption uncertain | Tool unused = waste | P4 metrics + interviews; kill if < 20% |
| Maintainer burden after launch | Tool rots | P2 registry + P3 SSO reuse reduces load |

---

## 11. What we are explicitly NOT building

- ❌ A **hosted / productized platform** with external customers (you chose internal tool).
- ❌ An **open-source / community release** (no external community features).
- ❌ **Multi-GPU / distributed training** (contradicts the single-GPU goal).
- ❌ **Multi-tenant auth, external sharing, billing/quotas** (internal only).
- ❌ **Run scheduling / queueing** (not needed at Epoch's scale yet).
- ❌ **Novel ML methods / optimizers / architectures** (Surtur is a constraint strategy).

---

## 12. Milestone summary & sequencing

| Milestone | Phase | Exit criteria | Dependency |
|---|---|---|---|
| M0: Eval suite + thresholds defined | Pre-P0 | Research leads sign off | None (start here) |
| M1: Core claim validated | P0 | Retention + alignment pass within thresholds | M0 |
| M2: Reusable runner ships | P1 | P0 numbers reproduced via config | M1 |
| M3: Results reproducible 1-command | P2 | Registry + pinned env, rerun verified | M2 |
| M4: Dashboard launch | P3 | Launch → watch → retention, all in UI | M3 |
| M5: Surtur is default | P4 | Adoption + retention targets met | M4 |

**Critical path:** M0 → M1 → M2 → M3 → M4 → M5. The longest pole is M0 — until the eval suite exists, no real work can validate anything.

---

## 13. Open decisions for the team

1. **Who owns the eval suite?** (research lead name + date)
2. **What are the retention/alignment thresholds?** (numbers + date)
3. **Which base model(s) does P0 target?** (affects ablation scope)
4. **Which alignment objective** is the P0 proving ground? (harmlessness? instruction-following? a specific Epoch task?)
5. **Compute budget reality** — what single-GPU class is the target, and what's the full-fine-tune baseline cost to compare against?

---

*Prepared by Charter. Hand off to Atlas (system design) once M0 is resolved. Hand off to Forge (build) after Atlas delivers the P1/P3 designs.*
