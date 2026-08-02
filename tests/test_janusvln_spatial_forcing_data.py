import json

import pytest
import torch

from qwen_vl.data.data_qwen import (
    build_current_image_token_mask,
    normalize_janusvln_image_path,
    read_json_array,
)


@pytest.mark.parametrize("frame_count", [1, 2, 4, 9])
def test_current_image_mask_selects_only_final_frame(frame_count):
    vision_start = 10
    image_pad = 11
    vision_end = 12
    tokens_per_frame = [index + 2 for index in range(frame_count)]
    token_ids = [1]
    spans = []
    for token_count in tokens_per_frame:
        span_start = len(token_ids) + 1
        token_ids.extend([vision_start] + [image_pad] * token_count + [vision_end, 2])
        spans.append(range(span_start, span_start + token_count))

    mask = build_current_image_token_mask(
        torch.tensor(token_ids),
        image_token_id=image_pad,
        vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
        expected_token_count=tokens_per_frame[-1],
    )

    assert mask.sum().item() == tokens_per_frame[-1]
    assert set(torch.nonzero(mask).flatten().tolist()) == set(spans[-1])
    for historical_span in spans[:-1]:
        assert not mask[list(historical_span)].any()


def test_current_image_mask_rejects_wrong_grid_count():
    with pytest.raises(ValueError, match="does not match"):
        build_current_image_token_mask(
            torch.tensor([10, 11, 11, 12]),
            image_token_id=11,
            vision_start_token_id=10,
            vision_end_token_id=12,
            expected_token_count=3,
        )


def test_read_json_array_stops_at_requested_prefix(tmp_path):
    path = tmp_path / "large.json"
    path.write_text(json.dumps([{"index": index} for index in range(100)]))
    assert read_json_array(path, max_samples=3) == [
        {"index": 0},
        {"index": 1},
        {"index": 2},
    ]


def test_normalize_janus_path_uses_configured_root(tmp_path):
    old_absolute_path = "/some/old/location/R2R-CE-640x480/train/7/step_0003_STOP.png"
    relative_path = normalize_janusvln_image_path(old_absolute_path, tmp_path)
    assert relative_path == "R2R-CE-640x480/train/7/step_0003_STOP.png"
