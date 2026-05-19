# Samples

This directory contains sample sequences and runtime sequence folders. Sequence
data used by the pipeline should live under `samples/<sequence_name>/data`.
Generated intermediates such as `state.pkl`, `output/`, and preprocessing
caches should stay out of git.

Expected sequence layout:

```text
samples/<sequence_name>/
  data/
    transforms.json
    images/
    depth/
    instance_masks/
    instance_tree.json
```

`data/depth` is the active dense depth consumed by Clutt3R-Seg. In the paper
pipeline this should be the MVSAnywhere inference depth. The released code
assumes dense depth maps without invalid depth pixels; measured sensor depth with
holes should be replaced by MVSAnywhere or another dense-depth estimate for
initial frames. Update-frame depth is optional and is only used for depth
evaluation when available.

`instance_tree.json` is a precomputed artifact used by the source-available
release. It replaces the internal instance-tree builder, which is not included
in this repository because parts of that implementation depend on closed or
restricted components that cannot be redistributed. The paper describes the
construction procedure at the level intended for reproduction. The artifact
must be generated for the exact masks in `instance_masks/`.
