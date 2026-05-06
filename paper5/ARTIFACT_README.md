# The Dark Subspace of Fine-Tuning Memorisation

**Cross-Architecture Residual Concentration with Pythia-6.9B and 12B Causal Reference Points**

| Metadata | Value |
|----------|-------|
| Authors | Anonymous (double-blind submission) |
| Correspondence | Withheld for double-blind review |
| Target venue | ICML 2026 Mechanistic Interpretability Workshop (May 8 AOE submission) |
| Style | `icml2026` (blind submission) |
| Paper sources | `paper5/main.tex`, `paper5/intro.tex`, `paper5/methods.tex`, `paper5/results.tex`, `paper5/related.tex`, `paper5/appendix.tex`, `paper5/references.bib` |
| License | MIT (see repo `LICENSE`) |
| Pre-registration | `paper5/PREREGISTRATION.md` (PR-1 through PR-8). The pre-registration tag will be GPG-signed at the lock commit before submission. |
| Citation file | `CITATION.cff` (machine readable, repo root) |

This artifact README is the reviewer entry point.

---

## 1. Paper summary

We probe where the membership signal in fine-tuned transformers lives in the sparse autoencoder (SAE) decomposition of the residual stream. On Pythia-6.9B with five independently trained mixed-data SAEs at layer 16 (harmonized N=5 cohort, seeds 42-postfix, 43, 44, 45, 46), encode-decode reconstruction drops membership AUROC by 0.209 (cross-seed std 0.004, approximately 56 standard deviations above the cross-seed noise floor), and the reconstruction residual recovers AUROC 0.78. Pythia-12B at layer 18 replicates the drop in the range 0.139 to 0.169 across a clean three-init cohort (seeds 47, 48, 49, drops 0.169, 0.139, 0.160, mean 0.156, std 0.015) with reconstruction cosine above 0.99 on every seed. On the broader cohort of nine fine-tuned transformers spanning three vendor families, the SAE residual carries a stronger membership signal than the original activation on 5 of 7 cohort rows, with the two strong-drop reference points (Pythia-6.9B and Pythia-12B) sitting on the opposite side. A bag-of-words blind baseline reaches AUROC 0.4566 (below chance), ruling out a token-level vocabulary confound.

We also report a methodological caution. Under a word-order-shuffled paraphrase (the confound from Duan et al. 2024), the residual probe orientation reverses on all three audited models (Pythia-6.9B, Qwen2-7B, Pythia-12B), which we use to argue against bidirectional probe reporting under paraphrase. We scope out cross-corpus generalisation (single-corpus common cause is a stated limitation), output-layer behaviour for standard published probes (loss attack, MIN-K%, zlib all saturate at AUROC 1.000 on epoch-5 fine-tuning so the SAE layer-16 patch is not visible at the LM head), and any claim that the dark-subspace effect is universal across architectures (it is a within-Pythia reference point with cohort-wide cross-cohort inversion).

---

## 2. Repository quickstart

### 2.1 Environment

The canonical Python interpreter is `env/bin/python3`. Never use the system Python. Dependencies are declared in `pyproject.toml` (lower bounded, see `[project.dependencies]`).

```bash
# From repository root
python3 -m venv env
env/bin/python3 -m pip install --upgrade pip
env/bin/python3 -m pip install -e .
```

### 2.2 Where the code lives

| Area | Path |
|------|------|
| Paper 5 paper-specific scripts | `scripts/memcirc/` |
| SAE training (shared) | `scripts/shared/train_sae.py` |
| SLURM wrappers | `scripts/memcirc/shell/sbatch_*.sh` |
| Verifier scripts | `scripts/memcirc/verify_paper5.py` (CPU only, the reviewer-facing path) |
| Plotting | `scripts/memcirc/plot_paper5_figures.py`, `scripts/memcirc/plot_paper5_advanced.py` |

### 2.3 Where the data lives

| Data | Path | Notes |
|------|------|-------|
| Member, non-member, and mixed OWT corpus | `data/memcirc_ctrl_ft/{member,nonmember,mixed}.jsonl` | 1000 documents per split. mixed.jsonl is the union of member and non-member (used as label-blind SAE training corpus). |
| Disjoint OWT corpus | `data/memcirc_ctrl_disjoint/mixed_disjoint.jsonl` plus `disjointness_proof.json` | 2000 documents, SHA-256 disjointness proof intersection_count=0 vs the union of member and non-member pools. |
| Pile subset | `data/memcirc_pile_subset/` | Used by Pile-eval BCD experiments. |

### 2.4 Where the checkpoints live

| Checkpoint | Path |
|------------|------|
| Controlled fine-tuned models | `runs/controlled_ft/run_<timestamp>/ft_epoch{1,3,5}/model/` |
| Per-model SAE training | `runs/sae/train_sae__<model>__layer<L>__mult<m>__l1<coeff>__<ts>/sae_final.pt` |
| P12B per-task isolated SAE training dirs (collision-safe) | `runs/sae_array/p12b_multiseed/task{0..4}_seed{47..51}/` |
| Disjoint-corpus SAE control (corpus contamination check) | `runs/sae_disjoint/.../sae_final.pt` |
| Random-init SAE control (operator-shape check) | `runs/sae/random_init_p69_layer16/sae_final.pt` |

### 2.5 Where the results live

| Result class | Path |
|--------------|------|
| BCD per-model orthogonality and probe AUROC | `runs/memcirc/behavioral_channels/<condition>/orthogonality.json` |
| SAE dark subspace evaluation (per row) | `runs/memcirc/sae_dark_subspace/<condition>/results.json` |
| Causal ablation (K-PC, K=5, K=10, K=50, K=200) | `runs/memcirc/causal_ablation/`, `runs/memcirc/causal_ablation_K5/` |
| Behavioral channel layer sweep (P69 pre-FT and FT) | `runs/memcirc/behavioral_channels/{p69_BASE_pre_ft,p69_epoch5_layer_sweep}/` |
| Bag-of-words ceiling | `runs/memcirc/bow_ceiling/memcirc_ctrl_ft/results.json` |
| Paraphrase orientation flip | `runs/memcirc/paraphrase_sensitivity/{p69,qwen2,p12b}/results.json` |
| Standard published MIA probes (loss attack, MIN-K%, zlib) | `runs/memcirc/standard_mia_probes/p69_dark_subspace_replication/results.json` |
| P69 cross-seed noise floor aggregate (originally N=6 on disk, harmonized N=5 reporting 2026-05-06) | `runs/memcirc/sae_noise_floor/p69_aggregate.json` (N=6 disk artifact). Harmonized N=5 cluster at `paper5/results/p69_n5_harmonized_2026-05-06.json` |
| Cohort per-row paired bootstrap | `paper5/results/cohort_bootstrap.json` |
| Held-out d_K falsification probe | `paper5/results/heldout_dk.json` |

---

## 3. Hardware and runtime

### 3.1 Preferred SLURM nodes

H100 and L40S nodes are preferred. A40 nodes are acceptable for CPU-bound or small-model work. Do not run GPU scripts outside SLURM on shared nodes.

### 3.2 Per-experiment runtime envelopes

Calibrated from `sacct` over the experimental campaign that produced the artifact.

| Experiment class | Hardware | Wall time | Notes |
|------------------|----------|-----------|-------|
| One Pythia-12B mixed-SAE training plus dark-subspace eval | H100 | about 7.3 GPU-h | All five seeds in the P12B array landed in 7h09m to 7h22m. |
| One Pythia-12B mixed-SAE training plus dark-subspace eval | L40S | about 7 to 10 GPU-h | Disjoint-corpus retrain landed in 7.7h. |
| One smaller model full pipeline (P1B, P2.8B, Neo mult=8) | H100, L40S, or A40 | about 3 to 4 GPU-h | |
| One control closure (random-init SAE, standard MIA probes, BoW ceiling) | A40 or CPU | under 0.5 GPU-h | |
| Full ground-up family pipeline (50K docs, 5 epochs, BCD, mixed SAE, eval) | L40S | about 24 to 30 GPU-h | |
| Total compute the paper rests on | various | in the low hundreds of GPU-h | Do not attempt full reproduction in a workshop review window. |

### 3.3 CPU-only verification path for reviewers

The reviewer is not expected to launch any GPU job. The CPU-only verification path at section 6 reads the on-disk JSONs and checks that paper numbers match. It runs in seconds.

---

## 4. Paper-to-code map (claim by claim)

This table maps each numbered paper claim (C1 through C20) to its source script, on-disk result JSON, and a CPU-only reproduction command.

| Claim | One-line statement | Paper location | Source script | Source result JSON | Reproduction (CPU) |
|-------|--------------------|----------------|---------------|--------------------|---------------------|
| C1 | Nine fine-tuned models, seven families, 1B to 12B | `methods.tex:19-23` | `scripts/memcirc/behavioral_channels.py` | `runs/memcirc/behavioral_channels/<condition>/orthogonality.json` (per model) | `env/bin/python3 scripts/memcirc/verify_paper5.py` (Section 7 of the script lists per-model BCD rows) |
| C2 | 1000 member and 1000 non-member OWT documents per model | `methods.tex:25-27` | data preparation under `scripts/memcirc/` and `data/memcirc_ctrl_ft/` configs | `data/memcirc_ctrl_ft/{member,nonmember,mixed}.jsonl` | `env/bin/python3 -c "import json; print(sum(1 for _ in open('data/memcirc_ctrl_ft/member.jsonl')))"` (expect 1000) |
| C3 | Recall labels via loss median split, ROUGE-L Spearman -0.32 to -0.67 | `methods.tex:38-42`, `appendix.tex:49-53` | `scripts/memcirc/recall_label_validation.py` | `runs/memcirc/bcd_extractability/<model>_epoch5/extractability_predictor.json` | open the per-model JSON and read `spearman_corr` |
| C4 | BCD via cPCA (covariance-difference, paired-difference noted as superseded) | `methods.tex:43-52` | `scripts/memcirc/behavioral_channels.py:136-202` | n/a (methodological) | inspect the script |
| C5 | dK and dR cosine below 0.40 in nine models, seven below 0.23 | `results.tex:24-53` | `scripts/memcirc/behavioral_channels.py` | `runs/memcirc/behavioral_channels/<model>/orthogonality.json` (`per_layer.<L>.cosine_d_K_d_R`) | verify script Section 7 |
| C6 | Multi-seed BCD std below 0.005 across five seeds | `results.tex:52` | `scripts/memcirc/sae_noise_floor_aggregate.py` (aggregator over per-seed `behavioral_channels.py` runs) | `runs/memcirc/sae_noise_floor/p69_aggregate.json` (`bcd_seed_std`) plus `runs/memcirc/behavioral_channels/p69_epoch5_seed{42_postfix,43,44,45,46,47}/orthogonality.json` | verify script Section 7 |
| C7 | Pythia-6.9B N=5 mixed SAE drop 0.209 plus or minus 0.004, approximately 56 sigma (harmonized to N=5 cluster; the larger N=6 evidence pool is preserved on disk) | `intro.tex:11-16`, `main.tex:49-50` | `scripts/memcirc/sae_noise_floor_aggregate.py`, harmonization script `scripts/memcirc/p69_n5_harmonize.py` | harmonized cluster `paper5/results/p69_n5_harmonized_2026-05-06.json`, original N=6 aggregator `runs/memcirc/sae_noise_floor/p69_aggregate.json`, per-seed at `runs/memcirc/sae_dark_subspace/p69_mixed_sae_seed{42_postfix,43,44,45,46}/results.json` (seed 47 retained on disk but excluded from the canonical N=5 cluster).| verify script Section 7 |
| C8 | Pythia-12B mixed SAE drop range across the clean three-init cohort | `main.tex:63-64`, `results.tex:306-307` | `scripts/memcirc/shell/sbatch_p12b_multiseed_array.sh` | `runs/memcirc/sae_dark_subspace/p12b_mixed_sae_seed{47,48,49}/results.json` (final cohort, drops 0.169, 0.139, 0.160). | open the per-seed JSON and read `original.score_K_auroc` minus `sae_reconstructed.score_K_auroc` |
| C9 | Residual AUROC exceeds original on 5 of 7 cohort rows | `main.tex:79-80`, `intro.tex:42` | `scripts/memcirc/per_row_bootstrap_kocl2.py` | `paper5/results/cohort_bootstrap.json` (per-row 95% CIs and binomial sign test) | `env/bin/python3 -c "import json; print(json.load(open('paper5/results/cohort_bootstrap.json'))['per_row'])"` |
| C10 | Feature sufficiency, classifier features under 16% of S_K | `results.tex:77-88`, `appendix.tex:565-581` | `scripts/memcirc/fsc_random_null.py` | per-model FSC outputs (Table A6) | open the per-model FSC JSON |
| C11 | Ablating P69 classifier features changes detection by under 0.002 but collapses extraction | `results.tex:84-87` | `scripts/memcirc/feature_ablation_dark_subspace.py` | `runs/memcirc/sae_dark_subspace/p69_feature_ablation/results.json` | open the JSON, read the k=ALL_ACTIVE row |
| C12 | Privacy-aware SAE captures dK but not full membership signal | `results.tex:172-211`, `methods.tex:114-127`, `appendix.tex:652-703` | `scripts/memcirc/finetune_sae_dk.py` plus `scripts/memcirc/fresh_probe_test.py` | `runs/memcirc/sae_dark_subspace/p69_ft_dk{0.1,1.0}/results.json` plus `runs/memcirc/fresh_probe/p69_ft_dk{0.1,1.0}/results.json` | verify script Section 7 (top block) |
| C13 | P12B K=10 residual-PC ablation drops AUROC by 0.176, CI [0.165, 0.187] | `intro.tex:26-34`, `main.tex:69-77` | `scripts/memcirc/subspace_ablation_eval.py` | `runs/memcirc/causal_ablation/p12b_errPC_K10/results.json` (`auroc_drop_mean`, `auroc_drop_ci`) | `env/bin/python3 -c "import json; r=json.load(open('runs/memcirc/causal_ablation/p12b_errPC_K10/results.json')); print(r.get('auroc_drop_mean'), r.get('auroc_drop_ci'))"` |
| C14 | K=5 P12B drop 0.103, CI [0.094, 0.112] | `results.tex:291-295` | `scripts/memcirc/subspace_ablation_eval.py` | `runs/memcirc/causal_ablation_K5/p12b_errPC_K5/results.json` | as above with the K5 path |
| C15 | Architecture-family geometry split (descriptive) | `results.tex:215-261`, `main.tex:82-87` | `scripts/memcirc/norm_baseline.py` plus BCD outputs | `runs/memcirc/norm_baseline/<model>_epoch5/results.json` plus per-model BCD | verify script Section "Norm baseline" |
| C16 | BoW baseline AUROC 0.4566 rules out vocabulary confound | `results.tex:275-279`, `main.tex:119` | `scripts/memcirc/bow_ceiling.py` | `runs/memcirc/bow_ceiling/memcirc_ctrl_ft/results.json` | `env/bin/python3 -c "import json; print(json.load(open('runs/memcirc/bow_ceiling/memcirc_ctrl_ft/results.json')))"` |
| C17 | Pre-FT P69 baseline at chance, FT reaches 0.828 at L16 | `results.tex:280-283` | `scripts/memcirc/behavioral_channels.py` | `runs/memcirc/behavioral_channels/{p69_BASE_pre_ft,p69_epoch5_layer_sweep}/orthogonality.json` | verify script Section 8 |
| C18 | Word-order shuffle flips probe orientation across three models | `results.tex:315-323`, `main.tex:120` | `scripts/memcirc/paraphrase_sensitivity.py` | `runs/memcirc/paraphrase_sensitivity/{p69,qwen2,p12b}/results.json` (look at `orientation_sign` flip) | open each per-model JSON, compare original and paraphrased orientation signs |
| C19 | Standard published probes do not replicate output-layer dark-subspace effect | `results.tex:328-329`, `methods.tex:192-193` | `scripts/memcirc/standard_mia_probe_decomposition.py` | `runs/memcirc/standard_mia_probes/p69_dark_subspace_replication/results.json` | open the JSON and read the per-probe `dark_subspace_replicates` boolean |
| C20 | Bootstrap resample count discipline | `methods.tex:195-196`, `results.tex:109-112` | `scripts/memcirc/subspace_ablation_eval.py:75-80` plus per-script CI computations | n/a (statistical disclosure) | inspect script. The disclosure subsection in `methods.tex` reconciles the count internally |

The supporting per-row cohort bootstrap (n_boot=10000) and the held-out d_K falsification probe write their per-row JSONs to `paper5/results/` and are referenced from the main results section of the paper.

---

## 5. CPU-only verification path

Reviewers can confirm that the paper numbers match the on-disk JSONs without launching any GPU job. The verifier script is read-only and runs in seconds.

```bash
cd /path/to/repository
env/bin/python3 scripts/memcirc/verify_paper5.py
```

Expected output (abbreviated). The script prints sectioned banners (Section 7 BCD per-model, Section 8 pre-FT plus layer sweep, Table 2 top block, Table 2 bottom block, Norm baseline, Scaling, Pythia-1B epoch dynamics, standard MIA probes, Bibliography sanity check). Each section prints either the disk-read value or `MISSING` for any unavailable file. A few example lines that should appear.

```
======================================================================
7. Behavioral channels (BCD) per-model AUROC + cosine (Table 1)
======================================================================
  Pythia-6.9B L16: cos=...  angle=...  mem_auroc=0.803... rec_auroc=...
  Pythia-12B (L18) L18: cos=...  angle=...  mem_auroc=0.764... rec_auroc=...
  ...

======================================================================
Top block of Table 2 -- Pythia-6.9B and GPT-Neo-2.7B rows
======================================================================
  P69 mixed (paper 0.803/0.594/0.781/0.976) -- N=5 mean (harmonized 2026-05-06):
    orig=0.8030  recon=0.5937  drop_pp=20.9  resid=0.7807  rc=0.9762  L0=11
  ...

======================================================================
Bibliography sanity check
======================================================================
  PASS: meeus2025sok
  PASS: duan2024membership
  PASS: muhamed2025dsg
  ...
```

If any line shows `MISSING` or a number that disagrees with the paper, that is a verification failure to flag.

The script does not load any PyTorch model, does not access the network, does not write any file outside its own stdout, and works on a stock Python install with the `pyproject.toml` dependencies (`numpy` and `pandas` are not strictly required for this script, only the standard library plus `json`).

---

## 6. Caveats

### 6.1 Permanently excluded paths

Three sticky exclusions are load bearing on the cohort.

- **Mistral-7B SAE.** All L1-only and elastic-net configurations failed the pre-registered recon_cos and dead-feature-fraction gates. Mistral does not appear in the dark-subspace cohort.
- **Llama-3-8B SAE.** Permanently broken (recon_cos -0.033, 3814 dead features). Llama-3 does not appear in the dark-subspace cohort.
- **GPT-Neo mult=4 multi-seed.** All five L1 configurations fail the recon_cos gate. The mult=8 single-seed pilot gate-passes but with a weak drop of 0.036.

### 6.2 Single-corpus common cause

All nine fine-tuned models share the same OpenWebText corpus partition. The dark-subspace effect could be a property of OWT plus the controlled fine-tuning recipe rather than a property of memorisation in general. We disclose this as a stated scope limitation. Multi-corpus replication is identified as future work.

### 6.3 SAE validity gate band

Per `paper5/methods.tex`. The headline-band gate is reconstruction cosine above 0.90 (STRICT). The disclosure band is 0.85 to 0.90 (YELLOW), admitted with explicit row-level disclosure. Reconstruction cosine values below the YELLOW band are admitted to the cross-cohort observation with explicit below-YELLOW disclosure when the inversion observation does not require high reconstruction cosine. The pre-registration originally used 0.95, and the methods text plus the Amendment Log in PREREGISTRATION.md document the relaxation history.

- Pythia-6.9B N=5 mean recon cosine 0.976 (harmonized to N=5, see harmonization script). STRICT.
- Pythia-12B three-init cohort recon cosine above 0.99 on every seed. STRICT.
- Pythia-2.8B layer 20 mixed SAE recon cosine 0.780. BELOW YELLOW, admitted to the cross-cohort observation only with disclosure.
- Pythia-1B layer 14 mixed SAE recon cosine 0.867. YELLOW, admitted with disclosure.
- Falcon-7B mixed SAE. YELLOW or below, see per-row disclosure in the paper.

### 6.4 Held-out d_K is partition-fit dependent

The held-out d_K falsification probe returned NULL on both reference points. Held-out drop sits below in-partition by more than 2 sigma on both Pythia-6.9B (held-out 0.149, in-partition 0.213, drop difference -0.064) and Pythia-12B (held-out 0.105, in-partition 0.160, drop difference -0.055). The qualitative dark-subspace ordering (residual at or above original above reconstruction) is preserved on held-out, but the magnitude is partly a property of the labelling partition. `methods.tex` reframes d_K as a labelling-partition-tuned projection rather than an intrinsic direction.

---

## 7. Reproduction notes

The full pipeline (controlled fine-tuning, BCD, SAE training, dark-subspace evaluation, controls) is large and was developed across several months on a cluster. Reviewers should not expect to reproduce it in a workshop review window. The CPU-only verification path in Section 5 is the intended reviewer path.

For full reproduction, the SLURM wrappers under `scripts/memcirc/shell/sbatch_*.sh` are the canonical entry points. The collision-safe pattern for multi-seed SAE arrays is at `scripts/memcirc/shell/sbatch_p12b_multiseed_array.sh`, which isolates each task to a per-task working directory to avoid cross-task overwrite. Seed policy is at least five seeds per (model, benchmark, method) cell (seeds 0 through 4) per repository convention.

For figures and tables, plotting scripts are under `scripts/memcirc/plot_paper5_*.py`. These scripts read all numerical values from on-disk JSONs via `scripts/memcirc/figure_data_loader.py`, so any reviewer-driven regeneration sources values directly from the JSONs cited in Section 4.

---

## 8. Citation

```bibtex
@inproceedings{anonymous2026darksubspace,
  title  = {The Dark Subspace of Fine-Tuning Memorisation},
  author = {Anonymous},
  booktitle = {ICML 2026 Workshop on Mechanistic Interpretability},
  year   = {2026},
  note   = {Anonymous double-blind submission.}
}
```

---

