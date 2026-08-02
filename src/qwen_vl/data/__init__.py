"""JanusVLN dataset configuration."""

import re
from pathlib import Path


def _sampling_rate(dataset_name: str) -> float:
    match = re.search(r"%(\d+)$", dataset_name)
    return int(match.group(1)) / 100.0 if match else 1.0


def data_list(dataset_names, janusvln_data_root=None):
    """Resolve the single dataset supported by SpatialForcing-VLN."""
    if not janusvln_data_root:
        raise ValueError("janusvln_data_root must be provided")

    dataset_root = Path(janusvln_data_root).expanduser()
    configs = []
    for requested_name in dataset_names:
        dataset_name = re.sub(r"%(\d+)$", "", requested_name)
        if dataset_name != "janusvln_r2r":
            raise ValueError(
                f"Unsupported dataset {dataset_name!r}; expected 'janusvln_r2r'"
            )
        configs.append(
            {
                "annotation_path": str(dataset_root / "train_r2r.json"),
                "data_path": str(dataset_root),
                "tag": "3d",
                "sampling_rate": _sampling_rate(requested_name),
                "dataset_name": dataset_name,
            }
        )
    return configs
