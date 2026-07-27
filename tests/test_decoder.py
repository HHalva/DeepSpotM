"""Contract tests for the cross-attention gene decoder.

These build a tiny randomly-initialised decoder instead of downloading the gated
checkpoint, so CI can assert the output contract that downstream integrations rely on:
`forward` returns (expression, gene_features, attn_weights), expression is
(batch, n_requested_genes), and `gene_indices` both subsets and orders the columns.
TIAToolbox unpacks exactly this three-tuple.
"""

import pytest
import torch

from deepspotm.modules import (
    MODALITY_BY_SOURCE,
    MODALITY_VOCABULARY,
    CrossAttentionGeneDecoder,
)

BATCH = 2
N_PATCHES = 5
PATCH_DIM = 16
N_GENES = 12
GENE_EMBED_DIM = 8


@pytest.fixture
def decoder():
    """A small decoder with random gene embeddings."""
    torch.manual_seed(0)
    return CrossAttentionGeneDecoder(
        n_genes=N_GENES,
        patch_dim=PATCH_DIM,
        gene_embed_dim=GENE_EMBED_DIM,
        num_heads=2,
        num_layers=1,
    ).eval()


@pytest.fixture
def patch_tokens():
    """A batch of patch tokens as produced by the image encoder."""
    torch.manual_seed(1)
    return torch.randn(BATCH, N_PATCHES, PATCH_DIM)


def test_forward_returns_the_documented_triple(decoder, patch_tokens):
    """forward yields (expression, gene_features, attn_weights) with attn None by default."""
    with torch.no_grad():
        expression, gene_features, attn = decoder(patch_tokens)

    assert expression.shape == (BATCH, N_GENES)
    assert gene_features.shape == (BATCH, GENE_EMBED_DIM)
    assert attn is None
    assert torch.isfinite(expression).all()


def test_gene_indices_subset_the_output(decoder, patch_tokens):
    """Requesting a subset computes only those genes, in the requested order."""
    wanted = torch.tensor([7, 1, 4])
    with torch.no_grad():
        subset, _, _ = decoder(patch_tokens, gene_indices=wanted)
        full, _, _ = decoder(patch_tokens)

    assert subset.shape == (BATCH, len(wanted))
    torch.testing.assert_close(subset, full[:, wanted], rtol=1e-4, atol=1e-5)


def test_gene_indices_order_is_respected(decoder, patch_tokens):
    """Reversing the requested indices reverses the output columns."""
    wanted = torch.tensor([2, 9])
    with torch.no_grad():
        forward_order, _, _ = decoder(patch_tokens, gene_indices=wanted)
        reverse_order, _, _ = decoder(patch_tokens, gene_indices=wanted.flip(0))

    torch.testing.assert_close(
        forward_order, reverse_order.flip(1), rtol=1e-4, atol=1e-5
    )


def test_need_weights_returns_attention(decoder, patch_tokens):
    """Interpretability path returns per-layer attention instead of None."""
    with torch.no_grad():
        _, _, attn = decoder(patch_tokens, need_weights=True)

    assert attn is not None
    assert len(attn) >= 1


def test_modality_registry_is_consistent():
    """Every registered source maps to a modality in the canonical vocabulary."""
    assert set(MODALITY_BY_SOURCE.values()) <= set(MODALITY_VOCABULARY)
    # The ordering is load-bearing: the index is what the one-hot encoding uses.
    assert MODALITY_VOCABULARY[0] == "dna"
