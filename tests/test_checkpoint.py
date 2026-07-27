"""Tests for the checkpoint-loading guards.

`check_state_dict_load` exists to stop a checkpoint/architecture mismatch from silently
producing predictions out of uninitialised weights. That guard is worth a test, because
a regression in it fails quietly and looks like a bad model rather than a bad load.
"""

import pytest

from deepspotm.checkpoint import (
    VOCAB_SPECIFIC_PREFIXES,
    check_state_dict_load,
    filter_state_dict,
)


def test_vocab_specific_prefixes_shape():
    """The prefix list is a non-empty tuple of strings."""
    assert isinstance(VOCAB_SPECIFIC_PREFIXES, tuple)
    assert VOCAB_SPECIFIC_PREFIXES
    assert all(isinstance(p, str) for p in VOCAB_SPECIFIC_PREFIXES)


def test_filter_state_dict_drops_matching_prefixes():
    """Keys under a dropped prefix are removed and everything else is kept."""
    state = {
        "gene_decoder.gene_embeddings.weight": 1,
        "gene_decoder._router_bio_emb": 2,
        "image_encoder.blocks.0.weight": 3,
    }
    filtered = filter_state_dict(state, VOCAB_SPECIFIC_PREFIXES)
    assert filtered == {"image_encoder.blocks.0.weight": 3}


def test_filter_state_dict_without_prefixes_is_a_copy():
    """Filtering on nothing returns an equal but distinct dict."""
    state = {"a": 1}
    filtered = filter_state_dict(state, ())
    assert filtered == state
    assert filtered is not state


def test_check_state_dict_load_accepts_clean_load():
    """No drift is not an error."""
    check_state_dict_load([], [])


@pytest.mark.parametrize(
    ("missing", "unexpected"),
    [
        (["image_encoder.blocks.0.weight"], []),
        ([], ["image_encoder.blocks.0.weight"]),
    ],
)
def test_check_state_dict_load_rejects_unexplained_drift(missing, unexpected):
    """Drift outside the allow lists raises rather than loading silently."""
    with pytest.raises(RuntimeError, match="unaccounted-for key drift"):
        check_state_dict_load(missing, unexpected)


def test_check_state_dict_load_honours_allow_lists():
    """Drift that the caller declared expected is permitted."""
    check_state_dict_load(
        ["gene_decoder.gene_embeddings.weight"],
        ["gene_decoder._router_bio_emb"],
        allow_missing_prefixes=VOCAB_SPECIFIC_PREFIXES,
        allow_unexpected_prefixes=VOCAB_SPECIFIC_PREFIXES,
    )
