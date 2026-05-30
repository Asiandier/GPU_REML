from __future__ import annotations

import dataclasses
import json
import os
import re
from typing import Any, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class ComponentSpec:
    name: str
    variant_indices: np.ndarray
    annotation: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None


_NPZ_RESERVED_KEYS = {
    "component_names",
    "component_annotations_json",
    "component_provenance_json",
}


def _ordered_npz_component_keys(keys: Sequence[str]) -> list[str]:
    keys = list(keys)
    arr_pat = re.compile(r"arr_(\d+)\Z")
    arr_matches = [arr_pat.fullmatch(key) for key in keys]
    if all(match is not None for match in arr_matches):
        return [
            key
            for _, key in sorted(
                (int(match.group(1)), key)
                for key, match in zip(keys, arr_matches)
            )
        ]
    return keys


def _load_optional_json_array(data, key: str, n_components: int) -> list[dict[str, Any] | None] | None:
    if key not in data:
        return None
    raw = np.asarray(data[key]).reshape(-1)
    if raw.size != n_components:
        raise ValueError(
            f"{key} length mismatch: expected {n_components}, got {int(raw.size)}."
        )
    out: list[dict[str, Any] | None] = []
    for item in raw.tolist():
        text = str(item).strip()
        if not text:
            out.append(None)
            continue
        value = json.loads(text)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{key} entries must decode to JSON objects or null.")
        out.append(value)
    return out


def _load_npz_component_specs(path: str) -> list[ComponentSpec]:
    with np.load(path, allow_pickle=False) as data:
        all_keys = list(data.files)
        array_keys = [key for key in all_keys if key not in _NPZ_RESERVED_KEYS]
        if not array_keys:
            raise ValueError(f"{path}: no component arrays found.")
        ordered_keys = _ordered_npz_component_keys(array_keys)
        n_components = len(ordered_keys)

        if "component_names" in data:
            raw_names = np.asarray(data["component_names"]).reshape(-1)
            if raw_names.size != n_components:
                raise ValueError(
                    f"component_names length mismatch: expected {n_components}, got {int(raw_names.size)}."
                )
            names = [str(name) for name in raw_names.tolist()]
        elif ordered_keys != array_keys:
            names = [f"component_{idx:03d}" for idx in range(n_components)]
        else:
            names = ordered_keys

        annotations = _load_optional_json_array(data, "component_annotations_json", n_components)
        provenance = _load_optional_json_array(data, "component_provenance_json", n_components)

        specs: list[ComponentSpec] = []
        for idx, key in enumerate(ordered_keys):
            specs.append(
                ComponentSpec(
                    name=str(names[idx]),
                    variant_indices=np.asarray(data[key], dtype=np.int64).reshape(-1),
                    annotation=None if annotations is None else annotations[idx],
                    provenance=None if provenance is None else provenance[idx],
                )
            )
        return specs


def _load_json_component_specs(path: str) -> list[ComponentSpec]:
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        components = payload.get("components")
    else:
        components = payload
    if not isinstance(components, list) or not components:
        raise ValueError(f"{path}: component spec must contain a non-empty components list.")

    specs: list[ComponentSpec] = []
    for idx, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError(f"{path}: component {idx} must be an object.")
        if "variant_indices" not in component:
            raise ValueError(f"{path}: component {idx} is missing variant_indices.")
        annotation = component.get("annotation")
        provenance = component.get("provenance")
        if annotation is not None and not isinstance(annotation, dict):
            raise ValueError(f"{path}: component {idx} annotation must be an object when provided.")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError(f"{path}: component {idx} provenance must be an object when provided.")
        specs.append(
            ComponentSpec(
                name=str(component.get("name") or f"component_{idx:03d}"),
                variant_indices=np.asarray(component["variant_indices"], dtype=np.int64).reshape(-1),
                annotation=annotation,
                provenance=provenance,
            )
        )
    return specs


def load_component_specs(path: str) -> list[ComponentSpec]:
    if not path:
        return []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return _load_json_component_specs(path)
    if ext == ".npz":
        return _load_npz_component_specs(path)
    raise ValueError(
        f"Unsupported component spec format for {path!r}. Use .json or .npz."
    )


__all__ = ["ComponentSpec", "load_component_specs"]
