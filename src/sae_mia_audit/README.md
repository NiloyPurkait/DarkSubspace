# sae_mia_audit/

Core Python package implementing the SAE-MIA evaluation framework for auditing pretraining data membership in large language models.

## Package Architecture

```mermaid
flowchart TB
    subgraph "Data Layer"
        D1[data/pdd.py<br/>Benchmark loaders]
        D2[data/splits.py<br/>Train/val/test splitting]
        D3[data/tokenizer.py<br/>Tokenization utils]
        D4[data/sae_corpus.py<br/>SAE training corpus]
    end
    
    subgraph "Model Layer"
        M1[models/wrapper.py<br/>CausalLMWrapper]
        M2[models/logprobs.py<br/>Log-probability extraction]
    end
    
    subgraph "SAE Layer"
        S1[sae/sae.py<br/>SparseAutoencoder]
        S2[sae/trainer.py<br/>SAE/MultiSAE Trainer]
        S3[sae/io.py<br/>Checkpoint I/O]
        S4[sae/interventions.py<br/>Feature ablations]
        S5[sae/interpret.py<br/>Feature interpretation]
    end
    
    subgraph "Methods Layer"
        ME1[methods/baselines.py<br/>Loss, Zlib, Length]
        ME2[methods/min_k.py<br/>Min-K%, Min-K%++]
        ME3[methods/na_pdd.py<br/>NA-PDD]
        ME4[methods/probe.py<br/>Linear probes]
        ME5[methods/sae_audit.py<br/>SAE-Feature]
        ME6[methods/sae_na_pdd.py<br/>SAE-NA-PDD]
    end
    
    subgraph "Evaluation Layer"
        E1[eval/metrics.py<br/>AUROC, TPR@FPR]
        E2[eval/bootstrap.py<br/>Confidence intervals]
        E3[eval/calibration.py<br/>Score normalization]
        E4[eval/groupwise.py<br/>Fairness slicing]
    end
    
    D1 --> ME1
    D1 --> ME2
    M1 --> ME3
    M1 --> ME4
    S1 --> ME5
    S1 --> ME6
    ME1 --> E1
    ME6 --> E1
```

## Design Goals

| Goal | Implementation |
|------|----------------|
| **Reproducibility** | Every run snapshots config + git state into `runs/...` |
| **Scientific Rigor** | Calibrated splits, bootstrap CIs, low-FPR metrics |
| **Interpretability** | Feature-level attributions and causal ablation checks |
| **Modularity** | Each subpackage has a clear, single responsibility |

## Subpackages

### `data/` - Dataset Handling

Benchmark loaders for WikiMIA, MIMIR, ArxivMIA, CCNewsPDD with proper train/val/test splitting.

### `baselines/` - Group C Baselines

Neuron-level probe and random rotation control baselines for mechanistic validation experiments.

### `models/` - LLM Wrappers

HuggingFace model wrappers with activation hooking for SAE interventions.

### `sae/` - Sparse Autoencoders

SAE architecture, training, checkpointing, and mechanistic interventions.

### `methods/` - MIA Methods

All 14+ membership inference methods from baselines to SAE-NA-PDD.

### `eval/` - Evaluation Metrics

AUROC, TPR@FPR, bootstrap CIs, calibration, and groupwise fairness metrics.

### `utils/` - Infrastructure

Logging, run directories, seeding, and HuggingFace utilities.

## Entry Point

```python
from sae_mia_audit.data.pdd import load_pdd_dataset, PDDDatasetSpec
from sae_mia_audit.models.wrapper import load_model_and_tokenizer
from sae_mia_audit.methods.sae_na_pdd import SAENAPDDConfig, fit_sae_na_pdd
from sae_mia_audit.eval.metrics import compute_metrics

# Load benchmark
examples = load_pdd_dataset(PDDDatasetSpec(name="wikia", model="pythia-1b"))

# Load model
model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1b")

# Fit SAE-NA-PDD
config = SAENAPDDConfig(sae_paths=[...], layer_indices=[4, 8])
scorer = fit_sae_na_pdd(config, model, tokenizer, train_examples)

# Evaluate
scores = scorer.score(test_examples)
metrics = compute_metrics(test_labels, scores)
print(f"AUROC: {metrics.auroc:.3f}, TPR@5%FPR: {metrics.tpr_at_fpr_5pct:.3f}")
```

## Installation

```bash
# Development install
pip install -e .

# Or add src/ to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/sae-mia-audit/src"
```
