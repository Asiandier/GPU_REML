# Component Specifications

Component specifications define how SNPs are grouped into variance components
for multi-GRM REML models.

## NPZ Format

Each component is a one-dimensional integer array of zero-based variant indices
in source genotype order:

```python
import numpy as np

np.savez(
    "components.npz",
    arr_0=np.array([0, 1, 2], dtype=np.int64),
    arr_1=np.array([3, 4, 5], dtype=np.int64),
    component_names=np.array(["component_a", "component_b"]),
)
```

Reserved optional keys:

- `component_names`
- `component_annotations_json`
- `component_provenance_json`

The JSON fields should contain one JSON object or `null` per component.

## JSON Format

```json
{
  "components": [
    {
      "name": "component_a",
      "variant_indices": [0, 1, 2],
      "annotation": {"kind": "example"}
    },
    {
      "name": "component_b",
      "variant_indices": [3, 4, 5]
    }
  ]
}
```

## CLI Use

```bash
gpu-reml \
  --bed-prefix /path/to/data \
  --component-spec components.npz \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt
```

Component definitions are interpreted after any sample filtering, but variant
indices remain relative to the source genotype variant order.
