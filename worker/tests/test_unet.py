"""Detector architecture. prd.md §5.2.2, D-03."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from demosaic_worker.detect.unet import (  # noqa: E402
    MosaicUNet,
    dice_bce_loss,
    false_positive_area,
    mask_iou,
)


def test_the_output_is_full_resolution() -> None:
    """prd.md §5.11 blends on the mask, so boundary precision is the product requirement."""
    model = MosaicUNet(width=8)
    x = torch.zeros(2, 1, 64, 64)

    assert model(x).shape == (2, 1, 64, 64)


def test_it_stays_small() -> None:
    """Starting large would hide whether the signal is learnable, which is the only question yet."""
    assert MosaicUNet(width=32).parameter_count < 10_000_000


@pytest.mark.parametrize("size", [64, 128, 256])
def test_it_accepts_the_sizes_the_pipeline_uses(size: int) -> None:
    model = MosaicUNet(width=8)
    assert model(torch.zeros(1, 1, size, size)).shape == (1, 1, size, size)


def test_iou_is_one_for_a_perfect_prediction() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:6, 2:6] = 1.0

    logits = torch.where(target > 0, 10.0, -10.0)

    assert mask_iou(logits, target).item() == pytest.approx(1.0)


def test_iou_is_one_when_there_is_nothing_to_find_and_nothing_predicted() -> None:
    target = torch.zeros(1, 1, 8, 8)
    logits = torch.full((1, 1, 8, 8), -10.0)

    assert mask_iou(logits, target).item() == pytest.approx(1.0)


def test_iou_falls_for_a_displaced_prediction() -> None:
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 4:12, 4:12] = 1.0

    displaced = torch.full((1, 1, 16, 16), -10.0)
    displaced[0, 0, 8:16, 8:16] = 10.0

    assert 0.0 < mask_iou(displaced, target).item() < 1.0


def test_false_positive_area_counts_only_wrongly_marked_pixels() -> None:
    """prd.md §5.2.5a — firing on clean footage is the failure that damages a user's video."""
    target = torch.zeros(1, 1, 10, 10)
    logits = torch.full((1, 1, 10, 10), -10.0)
    logits[0, 0, 0:5, 0:2] = 10.0        # 10 of 100 pixels

    assert false_positive_area(logits, target).item() == pytest.approx(0.10)


def test_false_positive_area_ignores_correctly_marked_pixels() -> None:
    target = torch.ones(1, 1, 10, 10)
    logits = torch.full((1, 1, 10, 10), 10.0)

    assert false_positive_area(logits, target).item() == pytest.approx(0.0)


def test_the_loss_rewards_a_correct_mask() -> None:
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0

    good = dice_bce_loss(torch.where(target > 0, 6.0, -6.0), target)
    bad = dice_bce_loss(torch.where(target > 0, -6.0, 6.0), target)

    assert good < bad


def test_the_loss_punishes_predicting_nothing_on_a_small_region() -> None:
    """BCE alone would be happy here; Dice is what supplies the gradient that cares."""
    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 30:34, 30:34] = 1.0     # 16 of 4096 pixels

    empty = dice_bce_loss(torch.full((1, 1, 64, 64), -6.0), target)

    assert empty > 0.9, "a model predicting nothing must not score well"


def test_a_gradient_step_reduces_the_loss() -> None:
    torch.manual_seed(0)
    model = MosaicUNet(width=8)

    x = torch.rand(2, 1, 64, 64)
    y = torch.zeros(2, 1, 64, 64)
    y[:, :, 16:48, 16:48] = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    before = dice_bce_loss(model(x), y)
    for _ in range(5):
        loss = dice_bce_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert dice_bce_loss(model(x), y) < before
