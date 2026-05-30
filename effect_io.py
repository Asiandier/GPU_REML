from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .io_utils import write_joined_rows
from .reml_model import EffectEstimates


def write_effect_outputs(
    *,
    out_prefix: str,
    effects: EffectEstimates,
    sample_ids: Sequence[str],
    component_global_offsets: Sequence[int] | None = None,
    component_source_variant_indices: Sequence[Sequence[int]] | None = None,
    component_names: Sequence[str] | None = None,
    component_annotations: Sequence[dict[str, object] | None] | None = None,
    component_provenance: Sequence[dict[str, object] | None] | None = None,
) -> dict[str, str]:
    """Write post-REML effect estimates to TSV/JSON outputs."""
    if not out_prefix:
        raise ValueError("write_effect_outputs requires a non-empty out_prefix.")
    if component_global_offsets is not None and component_source_variant_indices is not None:
        raise ValueError(
            "Use either component_global_offsets or component_source_variant_indices, not both."
        )

    sample_ids = list(sample_ids)
    g_total = np.asarray(effects.random_effect, dtype=np.float64).reshape(-1)
    if len(sample_ids) != g_total.size:
        raise ValueError(
            f"sample_ids length mismatch: expected {g_total.size}, got {len(sample_ids)}."
        )

    fixed_path = out_prefix + ".fixed_effects.tsv"
    random_path = out_prefix + ".random_effect.tsv"
    random_comp_path = out_prefix + ".random_effect_components.tsv"
    meta_path = out_prefix + ".effect_metadata.json"

    beta = np.asarray(effects.fixed_effects, dtype=np.float64).reshape(-1)
    write_joined_rows(
        fixed_path,
        "covariate_index\tlabel\tbeta\n",
        (
            f"{idx}\t{'intercept' if idx == 0 else f'covariate_{idx}'}\t{val:.8e}\n"
            for idx, val in enumerate(beta)
        ),
    )

    write_joined_rows(
        random_path,
        "sample_index\tiid\trandom_effect\n",
        (f"{idx}\t{iid}\t{val:.8e}\n" for idx, (iid, val) in enumerate(zip(sample_ids, g_total))),
    )

    g_components = [np.asarray(g, dtype=np.float64).reshape(-1) for g in effects.random_effect_components]
    n_components = len(g_components)
    resolved_component_names = (
        [str(name) for name in component_names]
        if component_names is not None
        else [f"component_{g:03d}" for g in range(n_components)]
    )
    emit_component_name_col = component_names is not None
    if len(resolved_component_names) != n_components:
        raise ValueError(
            f"component_names length mismatch: expected {n_components}, got {len(resolved_component_names)}."
        )
    resolved_component_annotations = (
        list(component_annotations)
        if component_annotations is not None
        else [None] * n_components
    )
    if len(resolved_component_annotations) != n_components:
        raise ValueError(
            "component_annotations length mismatch: "
            f"expected {n_components}, got {len(resolved_component_annotations)}."
        )
    resolved_component_provenance = (
        list(component_provenance)
        if component_provenance is not None
        else [None] * n_components
    )
    if len(resolved_component_provenance) != n_components:
        raise ValueError(
            "component_provenance length mismatch: "
            f"expected {n_components}, got {len(resolved_component_provenance)}."
        )
    header = ["sample_index", "iid"] + [f"component_{g:03d}" for g in range(len(g_components))]
    write_joined_rows(
        random_comp_path,
        "\t".join(header) + "\n",
        (
            f"{row_idx}\t{iid}\t" + "\t".join(f"{comp[row_idx]:.8e}" for comp in g_components) + "\n"
            for row_idx, iid in enumerate(sample_ids)
        ),
    )

    snp_paths: dict[str, str] = {}
    offsets = None if component_global_offsets is None else np.asarray(component_global_offsets, dtype=np.int64)
    source_indices = (
        None
        if component_source_variant_indices is None
        else [
            np.asarray(idx, dtype=np.int64).reshape(-1)
            for idx in component_source_variant_indices
        ]
    )
    for g_idx, b in enumerate(effects.snp_effects):
        b_np = np.asarray(b, dtype=np.float64).reshape(-1)
        path = out_prefix + f".snp_effects.component_{g_idx:03d}.tsv"
        comp_name = resolved_component_names[g_idx]
        if source_indices is not None:
            src_idx = source_indices[g_idx]
            if src_idx.size != b_np.size:
                raise ValueError(
                    f"component_source_variant_indices[{g_idx}] length mismatch: "
                    f"expected {b_np.size}, got {src_idx.size}."
                )
            if emit_component_name_col:
                write_joined_rows(
                    path,
                    "component_index\tcomponent_name\tcomponent_local_snp_index\tsource_snp_index\tbeta\n",
                    (
                        f"{g_idx}\t{comp_name}\t{local_idx}\t{int(src_idx[local_idx])}\t{val:.8e}\n"
                        for local_idx, val in enumerate(b_np)
                    ),
                )
            else:
                write_joined_rows(
                    path,
                    "component_index\tcomponent_local_snp_index\tsource_snp_index\tbeta\n",
                    (
                        f"{g_idx}\t{local_idx}\t{int(src_idx[local_idx])}\t{val:.8e}\n"
                        for local_idx, val in enumerate(b_np)
                    ),
                )
        elif offsets is not None:
            start = int(offsets[g_idx])
            if emit_component_name_col:
                write_joined_rows(
                    path,
                    "component_index\tcomponent_name\tcomponent_local_snp_index\tglobal_snp_index\tbeta\n",
                    (
                        f"{g_idx}\t{comp_name}\t{local_idx}\t{start + local_idx}\t{val:.8e}\n"
                        for local_idx, val in enumerate(b_np)
                    ),
                )
            else:
                write_joined_rows(
                    path,
                    "component_index\tcomponent_local_snp_index\tglobal_snp_index\tbeta\n",
                    (
                        f"{g_idx}\t{local_idx}\t{start + local_idx}\t{val:.8e}\n"
                        for local_idx, val in enumerate(b_np)
                    ),
                )
        else:
            if emit_component_name_col:
                write_joined_rows(
                    path,
                    "component_index\tcomponent_name\tsnp_index\tbeta\n",
                    (f"{g_idx}\t{comp_name}\t{local_idx}\t{val:.8e}\n" for local_idx, val in enumerate(b_np)),
                )
            else:
                write_joined_rows(
                    path,
                    "component_index\tsnp_index\tbeta\n",
                    (f"{g_idx}\t{local_idx}\t{val:.8e}\n" for local_idx, val in enumerate(b_np)),
                )
        snp_paths[f"component_{g_idx:03d}"] = path

    component_meta = []
    for g_idx, name in enumerate(resolved_component_names):
        component_meta.append(
            {
                "component_index": int(g_idx),
                "component_name": name,
                "n_snps": int(np.asarray(effects.snp_effects[g_idx]).size),
                "snp_effects_path": snp_paths[f"component_{g_idx:03d}"],
                "annotation": resolved_component_annotations[g_idx],
                "provenance": resolved_component_provenance[g_idx],
            }
        )

    meta = {
        "y_mean": float(effects.y_mean),
        "y_scale": float(effects.y_scale),
        "pcg_rel_res": float(effects.pcg_rel_res),
        "pcg_iters": int(effects.pcg_iters),
        "n_components": int(len(effects.snp_effects)),
        "n_samples": int(g_total.size),
        "n_fixed_effects": int(beta.size),
        "components": component_meta,
        "files": {
            "fixed_effects": fixed_path,
            "random_effect": random_path,
            "random_effect_components": random_comp_path,
            "snp_effects": snp_paths,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    return {
        "fixed_effects": fixed_path,
        "random_effect": random_path,
        "random_effect_components": random_comp_path,
        "effect_metadata": meta_path,
        **{f"snp_effects_{k}": v for k, v in snp_paths.items()},
    }


__all__ = ["write_effect_outputs"]
