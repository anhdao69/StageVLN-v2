import torch

from qwen_vl.model.spatial_forcing import (
    SpatialForcingProjector,
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
