"""Protein featurisation with ESM-2.

The DHFR target sequence (wild-type or a resistant variant) is embedded with a
pretrained ESM-2 protein language model and mean-pooled over residues to a
single fixed-length vector. This is the same 'apply a frozen protein LM and
build on its embeddings' approach used in the AMR work -- ESM-2 weights are
NOT updated here (that would be fine-tuning; this is feature extraction).

Because there are only a handful of unique DHFR sequences, embeddings are
cached by sequence so you compute each one once.

Model sizes (trade speed vs quality):
  esm2_t12_35M_UR50D   - fast, 480-d, good for a laptop / small Colab GPU
  esm2_t33_650M_UR50D  - stronger, 1280-d, wants a real GPU
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


class ESMEmbedder:
    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        cache_dir: str = "data/esm_cache",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._device = device
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self.embedding_dim: int | None = None

    # -- lazy load so importing the module never triggers a big download ----
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        import esm

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        loader = getattr(esm.pretrained, self.model_name)
        self._model, self._alphabet = loader()
        self._model = self._model.to(self._device).eval()
        self._batch_converter = self._alphabet.get_batch_converter()
        self._repr_layer = self._model.num_layers
        # infer embedding dim from the model
        self.embedding_dim = self._model.embed_dim

    def _cache_path(self, sequence: str) -> str:
        key = hashlib.md5((self.model_name + "|" + sequence).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.npy")

    def embed(self, sequence: str) -> np.ndarray:
        """Return a 1-D mean-pooled embedding for a single protein sequence."""
        path = self._cache_path(sequence)
        if os.path.exists(path):
            return np.load(path)

        self._ensure_loaded()
        import torch

        data = [("protein", sequence)]
        _, _, tokens = self._batch_converter(data)
        tokens = tokens.to(self._device)
        with torch.no_grad():
            out = self._model(tokens, repr_layers=[self._repr_layer])
        reps = out["representations"][self._repr_layer][0]  # [L+2, dim]
        # drop BOS/EOS, mean-pool real residues
        emb = reps[1 : len(sequence) + 1].mean(0).cpu().numpy().astype(np.float32)
        np.save(path, emb)
        return emb

    def embed_many(self, sequences: list[str]) -> dict[str, np.ndarray]:
        """Embed a list of unique sequences, returning {sequence: embedding}."""
        return {seq: self.embed(seq) for seq in dict.fromkeys(sequences)}
