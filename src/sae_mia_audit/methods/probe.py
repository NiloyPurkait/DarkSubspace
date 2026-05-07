# ---------------------------------------------------------------------------
# Source: Liu et al., "Probing Language Models for Pre-training Data
#         Detection" (ACL 2024).
# Original repo: https://github.com/zhliu0106/probing-lm-data
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from sae_mia_audit.data.pdd import PDDExample
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.models.wrapper import CausalLMWrapper


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for the linear-probe membership-detection baseline.

    Specifies which residual-stream layer to extract activations from, the
    prompt template, tokenisation parameters, and logistic-regression
    hyper-parameters (including an optional ``c_grid`` for sensitivity
    analysis).
    """

    layer_idx: int = 8  # 0-indexed transformer layer
    prompt_template: str = "Here is a statement: {sample}\nIs the above statement correct? Answer:"
    seq_len: int = 256
    batch_size: int = 8
    max_train: Optional[int] = None  # optional cap
    C: float = 1.0  # inverse regularisation for logistic regression
    # C grid for regularisation sensitivity analysis
    # When provided, fit_with_c_grid() will try all values and report metrics
    c_grid: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    # Centering/scaling: match official implementation (zhliu0106/probing-lm-data)
    # The official code centers and scales activations per feature before probing.
    # Disabling these degrades performance, especially on ArxivMIA.
    center: bool = True
    scale: bool = True


class ProbePDD:
    """Probe-based PDD baseline (internal activation -> logistic regression)."""

    def __init__(self, model: CausalLMWrapper, cfg: ProbeConfig):
        self.model = model
        self.cfg = cfg
        self.clf: Optional[LogisticRegression] = None
        # Centering/scaling parameters (fitted on training data only)
        self._feat_mean: Optional[np.ndarray] = None  # [D]
        self._feat_std: Optional[np.ndarray] = None    # [D]

    def _format(self, text: str) -> str:
        return self.cfg.prompt_template.format(sample=text)

    @torch.no_grad()
    def _extract_features(self, texts: Sequence[str]) -> np.ndarray:
        # Explicit random_crop=False for deterministic evaluation
        tok_cfg = TokenizeConfig(seq_len=self.cfg.seq_len, random_crop=False)
        feats = []
        for i in tqdm(range(0, len(texts), self.cfg.batch_size), desc="probe_feats", dynamic_ncols=True):
            chunk = [self._format(t) for t in texts[i : i + self.cfg.batch_size]]
            batch = tokenize_batch(self.model.tokenizer, chunk, tok_cfg)
            input_ids = batch["input_ids"].to(self.model.model.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(self.model.model.device)

            out = self.model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            hs = out.hidden_states  # tuple length n_layers+1
            # hidden state after layer l is hs[l+1]
            l = self.cfg.layer_idx
            if l < 0 or (l + 1) >= len(hs):
                raise ValueError(f"layer_idx out of range: {l} (hidden_states has {len(hs)} elements)")
            h = hs[l + 1]  # [B, T, D]
            # final token position per sample
            if attn is None:
                pos = torch.full((h.shape[0],), h.shape[1] - 1, device=h.device, dtype=torch.long)
            else:
                pos = attn.sum(dim=1) - 1
            x = h[torch.arange(h.shape[0], device=h.device), pos]  # [B, D]
            feats.append(x.detach().float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    def fit(self, examples: Sequence[PDDExample]) -> None:
        """Train the probe on labelled PDD examples.

        Extracts last-token hidden states at ``cfg.layer_idx``, optionally
        centres/scales features, and fits a logistic-regression classifier.
        """
        ex = list(examples)
        if self.cfg.max_train is not None:
            ex = ex[: self.cfg.max_train]

        X = self._extract_features([e.text for e in ex])
        y = np.asarray([e.label for e in ex], dtype=int)

        # Fit centering/scaling on training data (matches official implementation)
        # See zhliu0106/probing-lm-data ActDataset.collect_acts():
        #   if center: acts = acts - torch.mean(acts, dim=0)
        #   if scale:  acts = acts / torch.std(acts, dim=0)
        if self.cfg.center:
            self._feat_mean = X.mean(axis=0)
            X = X - self._feat_mean
        if self.cfg.scale:
            self._feat_std = X.std(axis=0) + 1e-8  # avoid division by zero
            X = X / self._feat_std

        clf = LogisticRegression(
            C=self.cfg.C,
            max_iter=1000,
            solver="lbfgs",
            n_jobs=1,
        )
        clf.fit(X, y)
        self.clf = clf

    def score(self, examples: Sequence[PDDExample]) -> np.ndarray:
        """Return the predicted probability of class 1 (member) per example.

        Applies the same centring/scaling that was fitted on training data.
        Must be called after :meth:`fit` (or :meth:`fit_with_c_grid`).
        """
        if self.clf is None:
            raise RuntimeError("Call fit() before score().")
        X = self._extract_features([e.text for e in examples])
        # Apply same centering/scaling as training
        if self._feat_mean is not None:
            X = X - self._feat_mean
        if self._feat_std is not None:
            X = X / self._feat_std
        # probability of class 1
        proba = self.clf.predict_proba(X)[:, 1]
        return proba

    def fit_with_c_grid(
        self,
        train_examples: Sequence[PDDExample],
        val_examples: Sequence[PDDExample],
    ) -> Dict[str, any]:
        """Fit the probe with regularisation sensitivity analysis (C grid).
        
        This method trains multiple probes with different C values and reports
        metrics for each, enabling analysis of regularisation sensitivity.
        
        Args:
            train_examples: Training examples (for fitting the probe).
            val_examples: Validation examples (for evaluating performance).
            
        Returns:
            Dictionary with:
              - 'best_c': C value with best validation AUROC
              - 'c_results': Dict mapping C -> metrics dict
              - 'sensitivity_range': Max AUROC - Min AUROC across C values
              
        NOTE: This method sets self.clf to the best-performing model.
        """
        train_ex = list(train_examples)
        if self.cfg.max_train is not None:
            train_ex = train_ex[: self.cfg.max_train]
        
        # Extract features once (expensive)
        X_train = self._extract_features([e.text for e in train_ex])
        y_train = np.asarray([e.label for e in train_ex], dtype=int)
        
        X_val = self._extract_features([e.text for e in val_examples])
        y_val = np.asarray([e.label for e in val_examples], dtype=int)
        
        # Fit centering/scaling on training data
        if self.cfg.center:
            self._feat_mean = X_train.mean(axis=0)
            X_train = X_train - self._feat_mean
            X_val = X_val - self._feat_mean
        if self.cfg.scale:
            self._feat_std = X_train.std(axis=0) + 1e-8
            X_train = X_train / self._feat_std
            X_val = X_val / self._feat_std
        
        c_results: Dict[float, Dict[str, float]] = {}
        best_c = self.cfg.C
        best_auc = -1.0
        best_clf = None
        
        for c_val in self.cfg.c_grid:
            clf = LogisticRegression(
                C=c_val,
                max_iter=1000,
                solver="lbfgs",
                n_jobs=1,
            )
            clf.fit(X_train, y_train)
            
            # Evaluate on validation set
            proba = clf.predict_proba(X_val)[:, 1]
            auc = float(roc_auc_score(y_val, proba)) if len(np.unique(y_val)) > 1 else 0.5
            
            # Compute TPR at low FPR
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_val, proba)
            
            def tpr_at_fpr(target_fpr):
                mask = fpr <= target_fpr
                return float(np.max(tpr[mask])) if np.any(mask) else 0.0
            
            c_results[c_val] = {
                "auroc": auc,
                "tpr_at_1pct_fpr": tpr_at_fpr(0.01),
                "tpr_at_0.1pct_fpr": tpr_at_fpr(0.001),
                "train_acc": float((clf.predict(X_train) == y_train).mean()),
                "val_acc": float((clf.predict(X_val) == y_val).mean()),
            }
            
            if auc > best_auc:
                best_auc = auc
                best_c = c_val
                best_clf = clf
        
        # Set the best classifier
        self.clf = best_clf
        
        # Compute sensitivity metrics
        aucs = [r["auroc"] for r in c_results.values()]
        sensitivity_range = max(aucs) - min(aucs)
        
        return {
            "best_c": best_c,
            "c_results": c_results,
            "sensitivity_range": sensitivity_range,
            "best_auroc": best_auc,
        }
