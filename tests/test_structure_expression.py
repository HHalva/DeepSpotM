"""Tests for the gene-name to fixed-length-tensor mapping.

`StructureExpression` reads the packaged vocabulary, so it doubles as a check that the
shipped tokens.csv is usable from an installed wheel rather than only from a source tree.
"""

import torch

from deepspotm.modules import StructureExpression


def test_vocabulary_loads_from_packaged_assets():
    """The default constructor resolves the packaged gene vocabulary."""
    structurer = StructureExpression()
    assert len(structurer.gene_names_ordered) == 19338
    # token_type filtering must exclude the disease/tissue/special tokens.
    assert "<pad>" not in set(structurer.gene_names_ordered)


def test_structure_expression_places_values_by_gene_name():
    """Measured genes land at their vocabulary position; the rest are zero."""
    structurer = StructureExpression()
    genes = list(structurer.gene_names_ordered)
    target, other = genes[3], genes[100]

    values, mask = structurer.structure_expression({target: 2.5, other: 1.0})

    assert values.shape == (len(genes),)
    assert mask.shape == (len(genes),)
    assert values[3].item() == 2.5
    assert values[100].item() == 1.0
    assert mask[3] and mask[100]
    assert mask.sum().item() == 2
    assert values.sum().item() == 3.5


def test_negative_values_are_clamped():
    """Expression is non-negative; negatives are clamped rather than passed through."""
    structurer = StructureExpression()
    target = structurer.gene_names_ordered[0]

    values, _ = structurer.structure_expression({target: -5.0})

    assert values.min().item() == 0.0


def test_unknown_gene_names_are_ignored():
    """A symbol outside the vocabulary does not shift or corrupt the output."""
    structurer = StructureExpression()

    values, mask = structurer.structure_expression({"NOT_A_REAL_GENE": 9.0})

    assert not mask.any()
    assert torch.count_nonzero(values).item() == 0
