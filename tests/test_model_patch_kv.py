"""Contract tests for the DeepSpot-M projected patch-token API."""

from pathlib import Path

import pytest
import torch
from torch import nn

import deepspotm.model as model_module
from deepspotm.model import DeepSpotM

BATCH_SIZE = 2
GENE_EMBED_DIM = 8
N_GENES = 6
PATCH_DIM = 10


class TinyBackbone(nn.Module):
    """Small deterministic image encoder used instead of gated weights."""

    patch_dim = PATCH_DIM
    hidden_dim = PATCH_DIM

    def __init__(self) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(3, PATCH_DIM)

    def forward_patch_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = pixel_values.flatten(2).transpose(1, 2)
        return self.patch_projection(patches)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward_patch_tokens(pixel_values).mean(dim=1)


@pytest.fixture
def token_path(tmp_path: Path) -> str:
    """Create a minimal gene vocabulary."""
    path = tmp_path / "tokens.csv"
    rows = ["token,token_type"]
    rows.extend(f"GENE_{index},gene" for index in range(N_GENES))
    path.write_text("\n".join(rows), encoding="utf-8")
    return str(path)


@pytest.fixture
def model_factory(monkeypatch, token_path):
    """Build tiny DeepSpot-M variants without network or model assets."""

    def build_encoder(*_args, **_kwargs):
        return TinyBackbone()

    monkeypatch.setattr(model_module, "build_encoder", build_encoder)

    def build_model(**overrides):
        options = {
            "use_cross_attention": True,
            "gene_embed_dim": GENE_EMBED_DIM,
            "cross_attn_heads": 2,
            "cross_attn_layers": 1,
            "cross_attn_dropout": 0.0,
            "init_gene_embeddings": "xavier",
            "use_router": False,
            "use_lora": False,
            "token_path": token_path,
            "backbone_pretrained": False,
        }
        options.update(overrides)
        torch.manual_seed(0)
        return DeepSpotM(**options).eval()

    return build_model


@pytest.fixture
def pixel_values() -> torch.Tensor:
    """Return a deterministic batch of small RGB images."""
    torch.manual_seed(1)
    return torch.randn(BATCH_SIZE, 3, 4, 4)


def test_direct_forward_matches_split_path(model_factory, pixel_values):
    """Direct inference and explicit encode/decode produce identical output."""
    model = model_factory()

    with torch.inference_mode():
        direct_expression, direct_pooled, direct_attention = model(pixel_values)
        patch_kv = model.encode_patch_kv(pixel_values)
        split_expression, split_pooled, split_attention = (
            model.decode_patch_kv(patch_kv)
        )

    torch.testing.assert_close(direct_expression, split_expression)
    torch.testing.assert_close(direct_pooled, split_pooled)
    assert direct_attention is None
    assert split_attention is None


def test_forward_projects_patch_tokens_once(model_factory, pixel_values):
    """The normal forward path must not repeat the patch projection."""
    model = model_factory()
    projection_calls = []

    handle = model.gene_decoder.patch_proj.register_forward_hook(
        lambda *_args: projection_calls.append(None)
    )
    try:
        with torch.inference_mode():
            model(pixel_values)
    finally:
        handle.remove()

    assert len(projection_calls) == 1


def test_split_decode_preserves_gene_subset_order(model_factory, pixel_values):
    """Subset decoding follows the caller's requested gene order."""
    model = model_factory()
    wanted = torch.tensor([4, 1, 3])

    with torch.inference_mode():
        patch_kv = model.encode_patch_kv(pixel_values)
        full_expression, _, _ = model.decode_patch_kv(patch_kv)
        subset_expression, _, _ = model.decode_patch_kv(
            patch_kv,
            gene_indices=wanted,
        )

    torch.testing.assert_close(
        subset_expression,
        full_expression[:, wanted],
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("source", ["evo2", "orthrus"])
def test_split_path_preserves_multi_source_dispatch(
    model_factory,
    pixel_values,
    source,
):
    """Each selected source has direct-versus-split inference parity."""
    model = model_factory(
        init_gene_embeddings=["evo2", "orthrus"],
        bio_dims={"evo2": 4, "orthrus": 6},
    )
    model.set_source(source)

    with torch.inference_mode():
        direct_expression, direct_pooled, _ = model(pixel_values)
        patch_kv = model.encode_patch_kv(pixel_values)
        split_expression, split_pooled, _ = model.decode_patch_kv(patch_kv)

    assert model.current_source == source
    assert split_expression.shape == (BATCH_SIZE, N_GENES)
    torch.testing.assert_close(direct_expression, split_expression)
    torch.testing.assert_close(direct_pooled, split_pooled)


def test_cache_methods_reject_non_cross_attention_models(
    model_factory,
    pixel_values,
):
    """The cache boundary is unavailable for the pooled MLP gene head."""
    model = model_factory(use_cross_attention=False)
    expected_error = "use_cross_attention=True"

    with pytest.raises(RuntimeError, match=expected_error):
        model.encode_patch_kv(pixel_values)

    patch_kv = torch.randn(BATCH_SIZE, 4, GENE_EMBED_DIM)
    with pytest.raises(RuntimeError, match=expected_error):
        model.decode_patch_kv(patch_kv)
