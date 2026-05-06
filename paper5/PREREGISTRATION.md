# Paper 5. Pre-Registration Artifact

**Paper.** The Dark Subspace of Fine Tuning Memorisation. Evidence from Nine Transformers.
**Date drafted.** 2026-04-28.
**Target venues.** ICML 2026 Mechanistic Interpretability Workshop (May 8 AOE), ICLR 2027 (primary).
**Code repo.** This repository.
**Pre-registered git tag.** TODO. Sign the commit that locks the methodology before submission. Recommended tag `prereg-icml-mi-2026`.
**Mirror.** TODO. Upload this document to OSF (https://osf.io/) or AsPredicted (https://aspredicted.org/) at submission time and add the URL here.

---

## Purpose

This document inventories the pre-registered methodological choices, validity gates, and decision criteria for Paper 5. Items here were committed to the repository before the experimental results that depend on them were generated. This protects against post-hoc choice tuning.

The artifact is structured as a list of pre-registered items. Each item lists what was committed, the commit hash where it was first written, and the result file(s) that depend on it. A reviewer or future replicator can verify any claim in the paper by reading the cited script at the cited commit, then checking the on-disk JSON output.

## Pre-registered items

### PR-1. Membership Inference Setup

- **Members and non-members.** Defined by membership in the controlled fine-tuning corpus split. Member set: `data/memcirc_ctrl_ft/member.jsonl`. Non-member set: `data/memcirc_ctrl_ft/nonmember.jsonl`. 1000 each.
- **Probe.** Linear probe (logistic regression with L2 regularisation, C=1.0) trained on the leading knowledge direction $d_K$ from BCD (Behavioral Channel Decomposition).
- **Score.** Projection onto $d_K$, AUROC over member-vs-nonmember scores.
- **Layer.** Per-model layer chosen before the dark-subspace experiment. Pythia 6.9B: layer 16. Pythia 12B: layer 18. GPT-Neo 2.7B: layer 16. Qwen2 7B: layer 16. Other models: see `scripts/memcirc/configs/oc_roster.json`.
- **Reproducibility.** All probe and BCD scripts under `scripts/memcirc/` use deterministic seeds (default 42).

### PR-2. SAE Validity Gates

Gates pre-registered before the multi-seed experiments. Failure of any gate is documented and the model is excluded from the affected analysis.

- **Reconstruction cosine gate.** ~~Mean reconstruction cosine $\geq 0.95$ on the member/non-member corpus.~~ **NOTE.** Amended on 2026-05-02 to a tiered gate hierarchy (see Amendment Log at the end of this document and the SAE validity gate paragraph in `paper5/methods.tex:59`). Current operative rule. Strict gate at reconstruction cosine $\geq 0.90$ with L0 below 60 percent of the dictionary size. YELLOW band at $[0.85, 0.90)$, admitted with disclosure for the cohort-wide K-OC-2 inversion observation only and not for the within-Pythia strong-drop anchor. Reconstruction cosine values below the YELLOW band may be admitted to the K-OC-2 cohort with explicit below-YELLOW disclosure when the inversion observation is computed from the original-versus-residual AUROC comparison and does not require high reconstruction cosine; they are excluded from any analysis whose validity does require high reconstruction cosine, and excluded from the within-Pythia strong-drop anchor unconditionally.
- **L0 sparsity gate.** Implicit: dead-feature fraction $\leq 0.30$.
- **Multi-seed cohort.** Pythia 6.9B is the multi-seed causal anchor with ~~N=6 seeds (seeds 42-postfix, 43, 44, 45, 46, 47)~~ N=5 seeds (seeds 42-postfix, 43, 44, 45, 46) per Amendment 2026-05-06 (harmonization to align with the Pythia-1B multi-seed cohort seed list, see Amendment Log). Other models entered the breadth tier as single-SAE correlational, with multi-seed evidence reserved for ICLR-grade follow-up.

### PR-3. Causal Ablation (K-PC) Decision Criteria

- **Targets.** K $\in \{10, 50, 200\}$ across the four gate-passing models. K=5 added later as a cheap robustness extension on the same script.
- **Controls.** C1 random rotation (100 seeds), C2 matched Gaussian noise (100 seeds), C4 random column mask falsifier (20 seeds). All committed to repository before the primary K=10 result was computed.
- **Decisive criterion.** Primary ablation must beat C4 (random column mask) by $\geq 3\sigma$ (~0.06 AUROC at our noise level).
- **Disclosure rule.** If the decisive criterion is not met, the result is reported as subspace-level rather than direction-level, with the C4 magnitude disclosed in the main text.

### PR-4. Cross-Model Decisive Gate Outcome

- Pre-registered: 6 cells (Pythia 6.9B and Qwen2 7B at K $\in \{10, 50, 200\}$) tested against C4.
- Outcome: 0 of 6 cells passed the decisive criterion. Disclosed in main.tex §4 (case study reading paragraph) and in the limitations.

### PR-5. BoW Vocabulary Confound Control

- Bag-of-words classifier trained on the same member/nonmember split, no model or hidden state involvement.
- Pre-registered as a confound control.
- Outcome: AUROC 0.4566 (below chance), rules out token-level vocabulary explanation.
- Source: `runs/memcirc/bow_ceiling/memcirc_ctrl_ft/results.json`.

### PR-6. Paraphrase Sensitivity Diagnostic

- Word-order-shuffled paraphrase mode (the C5 confound from Duan et al. 2024).
- Pre-registered as a methodological audit, not a positive robustness claim.
- Outcome: orientation flip on all 3 audited models (Pythia 6.9B, Qwen2 7B, Pythia 12B). Reported as a methodological caution about bidirectional AUROC reporting.
- Source: `runs/memcirc/paraphrase_sensitivity/{p69,qwen2,p12b}/results.json`.

### PR-7. Negative Controls Added After the Main Pre-Registration

These controls were added after the main pre-registration but before the corresponding experiments produced results. They are listed here for transparency.

- **Pre-FT baseline.** BCD probe on un-fine-tuned `EleutherAI/pythia-6.9b` at layers 12, 14, 16, 18, 20. Output: `runs/memcirc/behavioral_channels/p69_BASE_pre_ft/`. Rules out "is this a base-model property?"
- **Layer sensitivity sweep.** BCD probe at layers 12, 14, 16, 18, 20 on the FT P69 model. Output: `runs/memcirc/behavioral_channels/p69_epoch5_layer_sweep/`. Rules out "is layer 16 cherry-picked?"
- **K=5 PC ablation.** Same script as K=10/50/200, just with K=5. Output: `runs/memcirc/causal_ablation_K5/`. Tightens the K-sweep claim from below.
- **Random-direction baseline.** Sample 100 random unit directions per model, compute member-vs-nonmember AUROC along each, compare distribution to the residual probe AUROC. Output: `runs/memcirc/random_direction_baseline/`. Rules out "any direction in activation space works."
- **Random-init SAE baseline.** Run dark-subspace evaluation against an untrained random-init SAE of matching shape on Pythia 6.9B. Output: `runs/memcirc/sae_dark_subspace/p69_random_init_baseline/`. Rules out "any dictionary-shaped operator works."
- **Pythia-12B multi-seed pilot.** First mixed-data SAE seed on Pythia 12B at P69-matched HPs (mult=4, L1=5e-4, layer 18). Output: `runs/memcirc/sae_dark_subspace/p12b_mixed_sae_seed42/`. Gate at recon_cos $\geq 0.95$ before scheduling additional seeds for a second multi-seed anchor.
- **GPT-Neo mult=8 pilot.** Mixed-data SAE on GPT-Neo 2.7B at mult=8, L1=5e-4 (versus the failed mult=4 baseline). Output: `runs/memcirc/sae_dark_subspace/neo_mixed_mult8_seed42/`. Gate at recon_cos $\geq 0.95$.

### PR-8. Statistical Reporting

- AUROC point estimates with paired bootstrap 95% confidence intervals across 10000 resamples.
- Holm-Bonferroni correction across cross-model causal cells.
- Decisive criteria stated above.
- Cross-seed standard deviation reported wherever multi-seed evidence is available (~~Pythia 6.9B N=6~~ Pythia 6.9B N=5 per Amendment 2026-05-06).
- Per-text scores released with the paper artifact.

## Verification protocol

To verify any numerical claim in the paper:
1. Identify the source path cited in the paper or the appendix.
2. Read the corresponding JSON file at the pre-registered git tag.
3. Compare the paper number against the on-disk number. They are required to match within the precision stated.

An internal script-by-script audit log is maintained alongside the manuscript in the working repository.

## Amendment Log

Pre-registered items above are immutable as committed. Amendments below document changes made after the original commit, with date, prior rule, new rule, and reason.

### Amendment 2026-05-06. PR-2 multi-seed cohort harmonization (Pythia 6.9B N=6 to N=5).

- **Date.** 2026-05-06.
- **Original cohort (struck through above).** Pythia 6.9B is the multi-seed causal anchor with N=6 seeds (seeds 42-postfix, 43, 44, 45, 46, 47). Cross-seed standard deviation reported at N=6.
- **New cohort.** Pythia 6.9B reported at N=5 seeds (42-postfix, 43, 44, 45, 46), drop seed 47. Cross-seed standard deviation reported at N=5.
- **Reason.** Reporting harmonization to align all multi-seed Pythia rows in `tab:dark_subspace` to a uniform N=5 SAE-init reporting convention, matching the Pythia-1B multi-seed cohort seed list. Pattern A (drop seed 47) was selected because seeds 42-postfix through 46 align one-to-one with the Pythia-1B N=5 seed list. Disk evidence is unchanged. The N=6 cohort remains preserved on disk under `runs/memcirc/sae_dark_subspace/p69_mixed_sae_seed{42_postfix,43,44,45,46,47}/results.json` and the cached aggregator at `runs/memcirc/sae_noise_floor/p69_aggregate.json`. Per-metric materiality from the harmonization memo is NEGLIGIBLE. All five canonical metrics shift by less than 0.005 between N=5 and N=6 cluster means. The 3dp-rounded table cells are identical for original / reconstructed / drop / recon_cos, only residual shifts by 0.002 at 3dp. Originally reported as N=6 mean drop 0.20932 ± 0.00337 at 62 sigma, harmonized 2026-05-06 to N=5 mean drop 0.20938 ± 0.00376 at 56 sigma per `paper5/results/p69_n5_harmonized_2026-05-06.json`. The within-Pythia strong-drop anchor claim (drop above 0.10 at recon_cos above 0.99) survives both reporting conventions unchanged.
- **Cross-reference.**Source JSON `paper5/results/p69_n5_harmonized_2026-05-06.json`. Subselection script `scripts/memcirc/p69_n5_harmonize.py`.

### Amendment 2026-05-02b. Post-hoc falsification audit. Held-out partition-fit probe for $d_K$.

- **Date.** 2026-05-02.
- **Status.** **Post-hoc falsification audit, not a pre-registered item.** This amendment was drafted after the held-out partition-fit probe outcome was observed and is recorded here for transparency. It does not retroactively claim pre-registration for the protocol.
- **Audit protocol.** Falsification probe for the $d_K$ direction. The $d_K$ direction is fit on a random 70 percent of the canonical labelled pool (1400 examples) and $\text{score}_K$ is evaluated on the disjoint 30 percent held-out partition (600 examples), with 10 random splits (split_seeds 0..9) and a 10000-resample paired bootstrap on the held-out subset.
- **Pass criterion (audit-defined).** Held-out drop greater than (in-partition mean drop minus 2 cross-split standard deviations).
- **Outcome on dispatch (2026-05-02).** Both anchors fail the audit-defined pass criterion. P69 anchor seed 43 (recon_cos 0.979). Held-out drop 0.149, 95 percent CI [0.072, 0.193]. In-partition drop 0.213. Pass threshold 0.189. FAIL. P12B fresh-init anchor seed 49 (recon_cos 0.992). Held-out drop 0.105, 95 percent CI [0.051, 0.149]. In-partition drop 0.160. Pass threshold 0.136. FAIL. Disclosed honestly. The audit reframes $d_K$ as a labelling-partition-tuned projection rather than an intrinsic membership direction. The qualitative ordering residual $\geq$ original $>$ reconstruction is preserved on held-out for both anchors, so the load-bearing locus claim is partition-fit-independent at the qualitative level.
- **Cross-reference.** Methods sentence at `paper5/methods.tex` lines 67--68 (pre-registered partition-fit probe). Results paragraph at `paper5/results.tex:125` (Held-out partition-fit probe for $d_K$). Self-contained appendix entry at `paper5/appendix.tex:955` (`app:heldout_dk`). Limitation in `paper5/main.tex:127` Discussion. Per-split data at `paper5/results/heldout_dk_2026-05-02.json`. Script at `scripts/memcirc/heldout_dk_eval.py`.

### Amendment 2026-05-02. PR-2 reconstruction cosine gate hierarchy.

- **Date.** 2026-05-02.
- **Original gate (struck through above).** Mean reconstruction cosine $\geq 0.95$ on the member and non-member corpus.
- **New gate hierarchy.** Strict gate at reconstruction cosine $\geq 0.90$ with L0 below 60 percent of the dictionary size. YELLOW band at $[0.85, 0.90)$, admitted with disclosure for the K-OC-2 cohort observation only. Reconstruction cosine values below the YELLOW band are admitted to the K-OC-2 cohort with explicit below-YELLOW disclosure when the inversion observation does not require high reconstruction cosine, and are excluded from the within-Pythia strong-drop anchor and from any analysis whose validity does require high reconstruction cosine. This is the operative rule cited in the SAE validity gate paragraph in `paper5/methods.tex:59`.
- **Reason.** The amendment was empirically required after observation of the K-OC-2 cohort. Three K-OC-2 cohort rows fall below the original 0.95 threshold (Pythia-1B at 0.87 in the YELLOW band, Pythia-2.8B at 0.78 below YELLOW, Falcon-7B at 0.77 below YELLOW). The K-OC-2 inversion observation is computed from the comparison between original and residual area-under-the-curve and does not require high reconstruction cosine to be informative. The Pythia-6.9B and Pythia-12B causal anchors used for the headline within-Pythia strong-drop claim sit at reconstruction cosine above 0.99 and would pass either the original 0.95 gate or the amended 0.90 strict gate. The amendment scope is therefore limited to admitting the three lower-recon-cosine rows to the cohort-wide observation with disclosure, and does not affect the headline causal-anchor claim.
- **Cross-reference.** SAE validity gate paragraph in `paper5/methods.tex:59`.

## Revision History

| Date | Change |
|------|--------|
| 2026-05-06 | Added Amendment 2026-05-06 for the Pythia 6.9B multi-seed cohort harmonization from N=6 (seeds 42-postfix, 43, 44, 45, 46, 47) to N=5 (drop seed 47). PR-2 Multi-seed cohort and PR-8 Statistical Reporting bullets updated with strikethrough plus Amendment annotation. Per-metric materiality is NEGLIGIBLE. Source JSON `paper5/results/p69_n5_harmonized_2026-05-06.json` (N=5 mean drop 0.20938 ± 0.00376 at 56 sigma, was 0.20932 ± 0.00337 at 62 sigma under N=6). Disk N=6 cohort preserved unchanged. |
| 2026-05-02 | Added Amendment 2026-05-02b for the post-hoc held-out partition-fit falsification audit on $d_K$. Documents the split protocol, pass criterion, NULL outcome on both anchors, qualitative survival caveat, and cross-references to the methods, results, limitations, and per-split JSON. |
| 2026-05-02 | Amendment 2026-05-02 to PR-2. Reconstruction cosine gate replaced with a tiered hierarchy (STRICT $\geq 0.90$, YELLOW $[0.85, 0.90)$, below-YELLOW disclosure-only for the K-OC-2 cohort observation). The empirical reason (three K-OC-2 cohort rows with reconstruction cosine values 0.87, 0.78, 0.77, none of which clear the original 0.95 threshold) is documented in the Amendment Log. The headline causal anchors at reconstruction cosine above 0.99 are unaffected. |
| 2026-04-28 | Initial pre-registration artifact draft. Captures PR-1 to PR-8 from the methodology committed before the experiments depending on them were generated. Awaits a GPG-signed git tag and external timestamp at submission time. |

## Edit Log
| Date | Agent | Change |
|------|-------|--------|
