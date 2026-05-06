##PACHED

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class TopKItem:
    activation: float
    text_window: str
    meta: dict


class FeatureTopKCollector:
    """Maintain top-k activating contexts for a set of SAE features."""

    def __init__(
        self,
        feature_ids: Sequence[int],
        k: int,
        tokenizer: PreTrainedTokenizerBase,
        window: int = 20,
    ):
        self.feature_ids = list(map(int, feature_ids))
        self.k = int(k)
        self.tokenizer = tokenizer
        self.window = int(window)

        # per feature heap of (activation, tiebreaker, TopKItem)
        self.heaps: Dict[int, List[Tuple[float, int, TopKItem]]] = {
            fid: [] for fid in self.feature_ids
        }

        # global monotonic counter for tie-breaking
        self._counter: int = 0

    def _decode_window(self, input_ids: torch.Tensor, pos: int) -> str:
        # input_ids: [T]
        lo = max(0, pos - self.window)
        hi = min(int(input_ids.shape[0]), pos + self.window + 1)
        ids = input_ids[lo:hi].tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def update(self, z: torch.Tensor, input_ids: torch.Tensor, meta: dict) -> None:
        """Update heaps given activations.

        Args:
            z: [B, T, F] or [T, F]
            input_ids: [B, T] or [T]
        """
        if z.ndim == 2:
            z = z.unsqueeze(0)
            input_ids = input_ids.unsqueeze(0)

        B, T, F = z.shape

        for b in range(B):
            ids = input_ids[b]

            for fid in self.feature_ids:
                # max activation across tokens for this feature in this example
                act_vals = z[b, :, fid]
                max_val, pos = torch.max(act_vals, dim=0)
                a = float(max_val.item())

                if a <= 0:
                    continue

                item = TopKItem(
                    activation=a,
                    text_window=self._decode_window(ids, int(pos.item())),
                    meta={**meta, "pos": int(pos.item())},
                )

                heap = self.heaps[fid]

                entry = (a, self._counter, item)
                self._counter += 1

                if len(heap) < self.k:
                    heapq.heappush(heap, entry)
                else:
                    # only replace if strictly better than smallest activation
                    if a > heap[0][0]:
                        heapq.heapreplace(heap, entry)

    def to_jsonable(self) -> dict:
        out = {}
        for fid, heap in self.heaps.items():
            # heap is min-heap; return sorted descending by activation
            items = sorted(heap, key=lambda x: -x[0])
            out[str(fid)] = [
                {
                    "activation": float(a),
                    "text_window": it.text_window,
                    "meta": it.meta,
                }
                for a, _, it in items
            ]
        return out
