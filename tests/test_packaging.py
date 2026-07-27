"""Packaging and asset tests.

The model weights are gated, so CI can never load a real checkpoint. What CI can
guarantee is that the package installs, that its public API is importable, and that
the data files the model depends on are actually shipped in the wheel. Those are the
failures that silently reach users.
"""

import csv
import importlib.metadata
import importlib.resources

import pytest

import deepspotm

# The released panel size. The model reports len(gene_names) == 19338, which is
# derived from this file, so a change here is a change to the model's output width.
EXPECTED_GENES = 19338


def test_version_is_installed():
    """The distribution is installed and exposes a version."""
    assert importlib.metadata.version("deepspotm")


@pytest.mark.parametrize("name", deepspotm.__all__)
def test_public_api_is_importable(name):
    """Every name promised by __all__ actually exists on the package."""
    assert hasattr(deepspotm, name), (
        f"deepspotm.__all__ advertises missing name {name!r}"
    )


@pytest.mark.parametrize(
    "asset",
    ["tokens.csv", "ensp_to_gene.csv", "midnight_config.json"],
)
def test_asset_is_packaged(asset):
    """Data files declared in package-data are present in the installed package."""
    resource = importlib.resources.files("deepspotm.assets") / asset
    assert resource.is_file(), f"{asset} is missing from the installed package"
    assert resource.read_bytes(), f"{asset} is empty"


def test_token_vocabulary_matches_model_panel():
    """tokens.csv carries the gene panel the model reports."""
    text = (importlib.resources.files("deepspotm.assets") / "tokens.csv").read_text()
    rows = list(csv.DictReader(text.splitlines()))

    genes = [r for r in rows if r["token_type"] == "gene"]
    assert len(genes) == EXPECTED_GENES

    # Token ids must be a dense 0..n-1 range; the decoder indexes into them directly.
    ids = sorted(int(r["token_id"]) for r in rows)
    assert ids == list(range(len(rows)))


def test_config_alphabet_path_resolves():
    """Config resolves the packaged vocabulary rather than a source-tree path."""
    from deepspotm.config import config

    assert config.ALPHABET_PATH.is_file()
