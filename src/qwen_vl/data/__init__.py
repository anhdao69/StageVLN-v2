"""SpatialStack-style JanusVLN dataset configuration."""

import json
import os
import re
from pathlib import Path


def _sampling_rate(dataset_name: str) -> float:
    match = re.search(r"%(\d+)$", dataset_name)
    return int(match.group(1)) / 100.0 if match else 1.0


def _expand_path(path):
    return str(Path(os.path.expandvars(path)).expanduser())


def _read_dataset_config(dataset_config):
    config_path = Path(dataset_config).expanduser()
    with config_path.open() as config_file:
        raw_config = json.load(config_file)

    if isinstance(raw_config, dict) and "annotation_path" in raw_config:
        entries = [raw_config]
    elif isinstance(raw_config, list):
        entries = raw_config
    else:
        raise ValueError(
            "dataset_config must contain one dataset object or a list of objects"
        )

    normalized = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"dataset_config entry {index} must be an object")
        missing = {"annotation_path", "data_path"} - entry.keys()
        if missing:
            raise ValueError(
                f"dataset_config entry {index} is missing: {', '.join(sorted(missing))}"
            )
        config = entry.copy()
        config["annotation_path"] = _expand_path(config["annotation_path"])
        config["data_path"] = _expand_path(config["data_path"])
        config.setdefault("tag", "3d")
        config.setdefault("dataset_format", "janusvln")
        normalized.append(config)
    return normalized


def data_list(dataset_names, dataset_config=None, janusvln_data_root=None):
    """Resolve JanusVLN data entries using SpatialStack's path-pair format.

    ``dataset_config`` is a JSON object (or list of objects) containing at least
    ``annotation_path`` and ``data_path``.  ``janusvln_data_root`` remains as a
    compatibility fallback for older R2R launch commands.
    """
    requested_names = [name.strip() for name in dataset_names if name.strip()]

    if dataset_config:
        configs = _read_dataset_config(dataset_config)
        if len(configs) == 1 and "dataset_name" not in configs[0]:
            if len(requested_names) > 1:
                raise ValueError(
                    "A single unnamed dataset_config cannot satisfy multiple datasets"
                )
            configs[0]["dataset_name"] = (
                re.sub(r"%(\d+)$", "", requested_names[0])
                if requested_names
                else Path(configs[0]["annotation_path"]).stem
            )

        config_by_name = {}
        for config in configs:
            dataset_name = config.get("dataset_name")
            if not dataset_name:
                raise ValueError(
                    "Every entry in a multi-dataset config needs dataset_name"
                )
            if dataset_name in config_by_name:
                raise ValueError(f"Duplicate dataset_name {dataset_name!r}")
            config_by_name[dataset_name] = config

        selected_names = requested_names or list(config_by_name)
        selected = []
        for requested_name in selected_names:
            dataset_name = re.sub(r"%(\d+)$", "", requested_name)
            if dataset_name not in config_by_name:
                raise ValueError(
                    f"Dataset {dataset_name!r} is not present in {dataset_config}"
                )
            config = config_by_name[dataset_name].copy()
            config["sampling_rate"] = _sampling_rate(requested_name)
            selected.append(config)
        return selected

    if not janusvln_data_root:
        raise ValueError("Provide dataset_config or janusvln_data_root")
    if not requested_names:
        requested_names = ["janusvln_r2r"]

    dataset_root = Path(janusvln_data_root).expanduser()
    configs = []
    for requested_name in requested_names:
        dataset_name = re.sub(r"%(\d+)$", "", requested_name)
        if dataset_name != "janusvln_r2r":
            raise ValueError(
                "The root-only compatibility mode supports janusvln_r2r; "
                "use dataset_config for other annotations"
            )
        configs.append(
            {
                "annotation_path": str(dataset_root / "train_r2r.json"),
                "data_path": str(dataset_root),
                "tag": "3d",
                "dataset_format": "janusvln",
                "sampling_rate": _sampling_rate(requested_name),
                "dataset_name": dataset_name,
            }
        )
    return configs
