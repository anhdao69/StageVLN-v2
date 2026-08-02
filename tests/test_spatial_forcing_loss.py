import pytest
import torch

from qwen_vl.model.spatial_forcing import (
    SpatialForcingProjector,
    add_vggt_position_embedding,
    create_aspect_ratio_uv_grid,
    position_grid_to_sincos_embedding,
    resize_teacher_spatial_grid,
    spatial_forcing_cosine_loss,
)


def test_resize_teacher_uses_spatial_grid():
    teacher = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    resized = resize_teacher_spatial_grid(
        teacher,
        teacher_grid_hw=(2, 2),
        student_grid_hw=(3, 4),
    )
    assert resized.shape == (12, 3)
    assert torch.isfinite(resized).all()


def test_vggt_position_embedding_matches_reference_uv_encoding():
    positioned = add_vggt_position_embedding(
        torch.zeros(2, 8),
        teacher_grid_hw=(1, 2),
        image_hw=(1, 2),
    )
    expected = torch.tensor(
        [
            [
                -0.0432454832,
                -0.0044706455,
                0.0901655629,
                0.0999000221,
                0.0,
                0.0,
                0.1,
                0.1,
            ],
            [
                0.0432454832,
                0.0044706455,
                0.0901655629,
                0.0999000221,
                0.0,
                0.0,
                0.1,
                0.1,
            ],
        ]
    )
    assert torch.allclose(positioned, expected, atol=1e-7)
    assert torch.allclose(positioned.norm(dim=-1), torch.full((2,), 0.2))


def test_vggt_position_embedding_preserves_grid_for_pooling():
    teacher = torch.randn(6, 16, dtype=torch.bfloat16)
    positioned = add_vggt_position_embedding(
        teacher,
        teacher_grid_hw=(2, 3),
        image_hw=(200, 600),
    )
    resized = resize_teacher_spatial_grid(
        positioned,
        teacher_grid_hw=(2, 3),
        student_grid_hw=(4, 5),
    )
    assert positioned.shape == teacher.shape
    assert positioned.dtype == torch.float32
    assert resized.shape == (20, 16)
    assert torch.isfinite(resized).all()


def test_vggt_position_embedding_validates_dimensions():
    grid = create_aspect_ratio_uv_grid(3, 2, aspect_ratio=1.5)
    assert grid.shape == (2, 3, 2)
    with pytest.raises(ValueError, match="divisible by four"):
        position_grid_to_sincos_embedding(grid, embed_dim=10)
    with pytest.raises(ValueError, match="token count"):
        add_vggt_position_embedding(
            torch.zeros(5, 8),
            teacher_grid_hw=(2, 3),
            image_hw=(200, 300),
        )


def test_cosine_loss_is_zero_for_identical_features():
    features = torch.randn(7, 16)
    loss, cosine = spatial_forcing_cosine_loss(features, features)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(cosine, torch.tensor(1.0), atol=1e-6)


def test_projector_and_student_receive_gradients_but_teacher_does_not():
    projector = SpatialForcingProjector(
        student_dim=8,
        hidden_dim=12,
        teacher_dim=6,
    )
    student = torch.randn(5, 8, requires_grad=True)
    teacher = torch.randn(5, 6, requires_grad=True)
    projected = projector(student)
    loss, _ = spatial_forcing_cosine_loss(projected, teacher)
    loss.backward()

    assert student.grad is not None and torch.count_nonzero(student.grad)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in projector.parameters()
    )
    assert teacher.grad is None
