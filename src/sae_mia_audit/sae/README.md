# `sae_mia_audit.sae`

Sparse-autoencoder implementation and checkpoint I/O used by the paper scripts.

| Module | Purpose | Used by |
| --- | --- | --- |
| `sae.py` | `SparseAutoencoder` and `SAEConfig` (encoder, decoder, sparsity loss). | `trainer.py`, `scripts/dark_subspace/sae_dark_subspace.py`, `subspace_ablation_eval.py` |
| `trainer.py` | `SAETrainConfig`, `SAETrainer`, `MultiSAETrainer` (training-loop orchestration). | `scripts/shared/train_sae_saif.py`, `scripts/dark_subspace/finetune_sae_dk.py` |
| `io.py` | `load_sae_checkpoint`, `load_sae_cfg`, `load_sae_checkpoint_any` (checkpoint readers; supports both this package's format and the SAIF format). | All paper scripts that load an SAE |
| `adapters.py` | `SAEProtocol` typing protocol and `SAIFSparseAutoencoderAdapter` for the SAIF checkpoint format. | `scripts/dark_subspace/behavioral_channels.py`, `sae_dark_subspace.py` |

Earlier iterations of this package included SAE feature-intervention helpers, consistency-matching, top-k feature collection, and post-hoc interpretation utilities. None of those modules were imported by any paper script and they have been removed from the public artefact.
