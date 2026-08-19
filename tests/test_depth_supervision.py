from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from qwen_vl.data.utils import prepare_image_inputs
from qwen_vl.model.depth_supervision import (
    GeoVRDepthHead,
    compute_geovr_depth_loss,
    create_normalized_uv_grid,
    filter_by_quantile,
)
from qwen_vl.model.geometry_encoders.vggt_encoder import VGGTEncoder
from qwen_vl.model.spatial_forcing import (
    Qwen3_5ForConditionalGenerationWithSpatialForcing,
)
from qwen_vl.train.argument import TrainingArguments
from qwen_vl.train.trainer import SpatialForcingTrainer


def make_small_depth_head():
    return GeoVRDepthHead(
        dim_in=16,
        patch_size=28,
        target_patch_size=28,
        features=8,
        out_channels=(8, 16, 32, 32),
    )


def test_dense_head_consumes_four_selected_features_in_relative_order():
    torch.manual_seed(0)
    head = make_small_depth_head()
    selected = [torch.randn(1, 6, 16, requires_grad=True) for _ in range(4)]
    observed_inputs = []
    handles = [
        projection.register_forward_pre_hook(
            lambda _module, inputs, index=index: observed_inputs.append(
                (index, inputs[0].detach().clone())
            )
        )
        for index, projection in enumerate(head.projects)
    ]
    prediction = head(
        selected,
        student_grid_hw=(2, 3),
        image_hw=(56, 84),
        target_hw=(56, 84),
    )
    for handle in handles:
        handle.remove()

    assert prediction.shape == (1, 56, 84)
    assert torch.isfinite(prediction).all()
    assert (prediction > 0).all()
    assert [index for index, _ in observed_inputs] == [0, 1, 2, 3]
    for index, projection_input in observed_inputs:
        expected = head.norm(selected[index]).transpose(1, 2).reshape(
            1, 16, 2, 3
        )
        assert torch.allclose(projection_input, expected)

    prediction.mean().backward()
    assert all(feature.grad is not None for feature in selected)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in head.parameters()
    )


def test_dense_head_rejects_absolute_index_style_input():
    head = make_small_depth_head()
    with pytest.raises(ValueError, match="exactly four"):
        head(
            [torch.randn(1, 6, 16) for _ in range(33)],
            student_grid_hw=(2, 3),
            image_hw=(56, 84),
            target_hw=(56, 84),
        )


def test_dense_head_matches_teacher_size_not_divisible_by_target_patch():
    head = make_small_depth_head()
    prediction = head(
        [torch.randn(1, 6, 16) for _ in range(4)],
        student_grid_hw=(2, 3),
        image_hw=(56, 84),
        target_hw=(42, 63),
    )
    assert prediction.shape == (1, 42, 63)


def test_depth_head_checkpoint_round_trip_is_numerically_identical():
    torch.manual_seed(1)
    head = make_small_depth_head().eval()
    features = [torch.randn(1, 6, 16) for _ in range(4)]
    expected = head(
        features,
        student_grid_hw=(2, 3),
        image_hw=(56, 84),
        target_hw=(56, 84),
    )
    restored = make_small_depth_head().eval()
    restored.load_state_dict(head.state_dict())
    actual = restored(
        features,
        student_grid_hw=(2, 3),
        image_hw=(56, 84),
        target_hw=(56, 84),
    )
    assert torch.equal(expected, actual)


def test_spatial_model_strict_restore_allows_only_omitted_teacher_keys():
    model_class = Qwen3_5ForConditionalGenerationWithSpatialForcing
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.student_depth_head = nn.Linear(4, 2)
    model.spatial_teacher = nn.Linear(4, 4)
    checkpoint = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    assert checkpoint
    assert not any(key.startswith("spatial_teacher.") for key in checkpoint)

    expected_weight = checkpoint["student_depth_head.weight"].clone()
    model.student_depth_head.weight.data.zero_()
    incompatible = model.load_state_dict(checkpoint, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(model.student_depth_head.weight, expected_weight)

    incomplete = dict(checkpoint)
    incomplete.pop("student_depth_head.weight")
    with pytest.raises(RuntimeError, match="student_depth_head.weight"):
        model.load_state_dict(incomplete, strict=True)


def test_geovr_depth_loss_matches_l1_plus_strided_gradient_formula():
    teacher = torch.ones(1, 4, 4)
    difference = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 10
    predicted = (teacher + difference).requires_grad_(True)
    output = compute_geovr_depth_loss(
        predicted,
        teacher,
        gradient_scales=(1,),
        outlier_keep_ratio=0.98,
    )
    expected_regression = difference.abs().mean()
    expected_gradient = (
        (difference[..., :, 1:] - difference[..., :, :-1]).abs().mean()
        + (difference[..., 1:, :] - difference[..., :-1, :]).abs().mean()
    )
    assert torch.allclose(output.regression, expected_regression)
    assert torch.allclose(output.gradient, expected_gradient)
    assert torch.allclose(output.loss, expected_regression + expected_gradient)
    output.loss.backward()
    assert predicted.grad is not None and torch.count_nonzero(predicted.grad)


def test_quantile_filter_preserves_geovr_edge_cases_and_gradients():
    small = torch.arange(1000.0, requires_grad=True)
    assert filter_by_quantile(small, 0.98) is small

    large = torch.linspace(0.0, 10.0, 2001, requires_grad=True)
    filtered = filter_by_quantile(large, 0.98)
    assert filtered.numel() == 1961
    assert filtered.max().item() == pytest.approx(9.8)
    filtered.mean().backward()
    assert large.grad is not None
    assert torch.count_nonzero(large.grad) == filtered.numel()


def test_invalid_teacher_depth_uses_mask_and_all_invalid_is_graph_connected():
    predicted = torch.ones(1, 4, 4, requires_grad=True)
    teacher = torch.ones_like(predicted)
    teacher[..., 0, 0] = float("nan")
    teacher[..., 0, 1] = 0
    output = compute_geovr_depth_loss(predicted, teacher, gradient_scales=(1,))
    assert output.valid_fraction == pytest.approx(14 / 16)
    assert torch.isfinite(output.loss)

    invalid = torch.full_like(predicted, float("nan"))
    zero_output = compute_geovr_depth_loss(predicted, invalid)
    assert zero_output.loss.requires_grad
    zero_output.loss.backward()
    assert predicted.grad is not None


def test_uv_grid_is_deterministic_and_uses_requested_dtype_device():
    first = create_normalized_uv_grid(
        3, 2, aspect_ratio=1.5, dtype=torch.float64, device=torch.device("cpu")
    )
    second = create_normalized_uv_grid(
        3, 2, aspect_ratio=1.5, dtype=torch.float64, device=torch.device("cpu")
    )
    assert first.shape == (2, 3, 2)
    assert first.dtype == torch.float64
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert not first.requires_grad


class CountingAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_hw = None

    def forward(self, images):
        self.calls += 1
        batch, frames, _, height, width = images.shape
        self.last_hw = (height, width)
        patches = (height // 14) * (width // 14)
        tokens = torch.randn(batch, frames, 1 + patches, 8)
        return [tokens.clone() for _ in range(24)], 1


class CountingDepthHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, aggregated_tokens_list, *, images, patch_start_idx):
        self.calls += 1
        batch, frames, _, height, width = images.shape
        depth = torch.ones(batch, frames, height, width, 1)
        confidence = torch.full((batch, frames, height, width), 2.0)
        return depth, confidence


class FakeVGGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = CountingAggregator()
        self.depth_head = CountingDepthHead()


def test_teacher_features_and_depth_share_one_aggregator_execution():
    encoder = VGGTEncoder.__new__(VGGTEncoder)
    nn.Module.__init__(encoder)
    encoder.reference_frame = "first"
    encoder.freeze_encoder = True
    encoder.enable_depth = True
    encoder.patch_size = 14
    encoder.vggt = FakeVGGT()
    result = encoder.encode_features_and_depth(
        torch.rand(1, 3, 28, 42),
        layer_indices=[23],
        spatial_merge_size=1,
        include_camera_token=False,
    )
    assert encoder.vggt.aggregator.calls == 1
    assert encoder.vggt.depth_head.calls == 1
    assert result["sf_features"][0].shape == (1, 6, 8)
    assert result["depth"].shape == (1, 28, 42)
    assert result["depth_conf"].shape == (1, 28, 42)
    assert result["teacher_image_hw"] == (28, 42)
    assert result["teacher_padding_hw"] == (0, 0)


def test_teacher_pads_only_for_vggt_and_crops_dense_outputs_to_qwen_fov():
    encoder = VGGTEncoder.__new__(VGGTEncoder)
    nn.Module.__init__(encoder)
    encoder.reference_frame = "first"
    encoder.freeze_encoder = True
    encoder.enable_depth = True
    encoder.patch_size = 14
    encoder.vggt = FakeVGGT()
    result = encoder.encode_features_and_depth(
        torch.rand(1, 3, 32, 48),
        layer_indices=[23],
        spatial_merge_size=1,
        include_camera_token=False,
    )
    assert encoder.vggt.aggregator.calls == 1
    assert encoder.vggt.aggregator.last_hw == (42, 56)
    assert result["teacher_image_hw"] == (42, 56)
    assert result["teacher_grid_hw"] == (3, 4)
    assert result["teacher_padding_hw"] == (10, 8)
    assert result["sf_features"][0].shape == (1, 12, 8)
    assert result["depth"].shape == (1, 32, 48)
    assert result["depth_conf"].shape == (1, 32, 48)


def test_hidden_state_capture_mapping_uses_final_norm_for_index_32():
    layers = [nn.Identity() for _ in range(32)]
    final_norm = nn.LayerNorm(4)
    fake_model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=layers, norm=final_norm)
        )
    )
    capture = Qwen3_5ForConditionalGenerationWithSpatialForcing._student_capture_module
    assert capture(fake_model, 7) is layers[6]
    assert capture(fake_model, 16) is layers[15]
    assert capture(fake_model, 24) is layers[23]
    assert capture(fake_model, 32) is final_norm


class DummyImageProcessor:
    merge_size = 2
    patch_size = 14

    def __call__(self, images, **_kwargs):
        _, _, height, width = images.shape
        return {
            "pixel_values": images.clone(),
            "image_grid_thw": torch.tensor([[1, height // 14, width // 14]]),
        }


def test_depth_preprocessing_reuses_exact_qwen_crop():
    pixels = torch.zeros(480, 640, 3, dtype=torch.uint8).numpy()
    pixels[:, :, 0] = torch.arange(640, dtype=torch.uint8).numpy()[None, :]
    image = Image.fromarray(pixels, mode="RGB")
    result = prepare_image_inputs(
        image,
        DummyImageProcessor(),
        model_type="qwen3.5",
        prepare_geometry=True,
        depth_supervision_enabled=True,
    )
    assert torch.equal(
        result["geometry_encoder_inputs"], result["pixel_values"][0]
    )


class TinyOptimizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.spatial_projector = nn.Linear(4, 4)
        self.student_depth_head = nn.Linear(4, 4)

    def forward(self, input_ids=None, labels=None):
        del input_ids, labels
        return {"loss": self.base.weight.sum() * 0}


def test_optimizer_assigns_every_depth_parameter_once_at_depth_lr(tmp_path):
    model = TinyOptimizerModel()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        learning_rate=1e-6,
        mm_projector_lr=1e-5,
        depth_head_lr=2e-5,
        report_to="none",
    )
    trainer = SpatialForcingTrainer(model=model, args=args)
    optimizer = trainer.create_optimizer()
    depth_ids = {id(parameter) for parameter in model.student_depth_head.parameters()}
    grouped_depth_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if group.get("group_name") == "depth_head"
        for parameter in group["params"]
    }
    all_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert grouped_depth_ids == depth_ids
    assert len(all_ids) == len(set(all_ids))
    assert all(
        group["lr"] == pytest.approx(2e-5)
        for group in optimizer.param_groups
        if group.get("group_name") == "depth_head"
    )
