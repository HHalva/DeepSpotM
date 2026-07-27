"""Tests for the image preprocessing helpers.

`get_eval_transforms` is the boundary every downstream integration crosses: TIAToolbox
hands it channel-last uint8 tiles, the WSI example hands it PIL images, and the model's
own transform hands it tensors. Each of those has to come out the same shape.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from deepspotm.utils import get_eval_transforms, get_normalize_params

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@pytest.fixture
def transform():
    """The evaluation transform used at 224 px with a center crop."""
    return get_eval_transforms(MEAN, STD, target_img_size=224, center_crop=True)


@pytest.mark.parametrize(
    "make_input",
    [
        pytest.param(
            lambda: np.random.randint(0, 256, (256, 300, 3), dtype=np.uint8),
            id="numpy-uint8",
        ),
        pytest.param(
            lambda: np.random.rand(256, 300, 3).astype(np.float32), id="numpy-float"
        ),
        pytest.param(
            lambda: Image.fromarray(np.zeros((256, 300, 3), np.uint8)), id="pil"
        ),
        pytest.param(lambda: torch.rand(3, 256, 300), id="tensor-chw"),
    ],
)
def test_accepts_every_input_type(transform, make_input):
    """Numpy, PIL and tensor inputs all normalize to the same (3, 224, 224) tensor."""
    out = transform(make_input())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_rejects_unsupported_input(transform):
    """An unsupported type fails loudly instead of silently producing garbage."""
    with pytest.raises(TypeError):
        transform("not an image")


def test_normalization_is_applied():
    """With mean/std given, the output is standardized; without, it stays in [0, 1]."""
    tile = np.full((224, 224, 3), 128, dtype=np.uint8)

    normalized = get_eval_transforms(MEAN, STD, target_img_size=224)(tile)
    raw = get_eval_transforms(None, None, target_img_size=224)(tile)

    assert 0.0 <= raw.min() and raw.max() <= 1.0
    torch.testing.assert_close(raw.mean().item(), 128 / 255, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(normalized, raw)


def test_center_crop_requires_a_size():
    """center_crop without target_img_size is a configuration error."""
    with pytest.raises(AssertionError):
        get_eval_transforms(MEAN, STD, target_img_size=-1, center_crop=True)


def test_get_normalize_params_from_compose():
    """Normalize parameters are recovered from a torchvision pipeline."""
    compose = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    mean, std = get_normalize_params(compose)
    assert tuple(mean) == MEAN
    assert tuple(std) == STD


def test_get_normalize_params_without_normalize():
    """A pipeline with no Normalize step reports no parameters."""
    compose = transforms.Compose([transforms.ToTensor()])
    assert get_normalize_params(compose) == (None, None)


def test_get_normalize_params_from_processor_dict():
    """HuggingFace-style processor dicts are supported."""
    processor = {"image_mean": list(MEAN), "image_std": list(STD)}
    mean, std = get_normalize_params(processor)
    assert tuple(mean) == MEAN
    assert tuple(std) == STD
