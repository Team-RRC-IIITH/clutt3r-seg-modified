<div align="center">

# Clutt3R-Seg: Sparse-view 3D Instance Segmentation for Language-grounded Grasping in Cluttered Scenes

[Jeongho Noh](https://jeonghonoh.github.io/)<sup>1</sup>,
[Tai Hyoung Rhee](https://williamrheeth.github.io/)<sup>1</sup>,
[Eunho Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee,+E)<sup>1</sup>,
[Jeongyun Kim](https://jeongyun0609.github.io/)<sup>1</sup>,
[Sunwoo Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee,+S)<sup>2</sup>,
[Ayoung Kim](https://ayoungk.github.io/)<sup>1,†</sup>

<sup>1</sup> Seoul National University &nbsp;&nbsp;
<sup>2</sup> Hyundai Motor Company &nbsp;&nbsp;
<sup>†</sup> Corresponding Author

**ICRA 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2602.11660-b31b1b.svg)](https://arxiv.org/abs/2602.11660)

</div>

Clutt3R-Seg is a zero-shot sparse-view 3D instance segmentation pipeline that builds hierarchy-based, open-vocabulary 3D instances for language-grounded grasping in cluttered scenes.

It groups noisy RGB-D masks across views, resolves over- and under-segmentation through an instance tree, and updates object correspondences after robot interactions without rescanning the full scene.

Paper: [arXiv:2602.11660](https://arxiv.org/abs/2602.11660)

![Clutt3R-Seg pipeline](assets/pipeline.jpg)

## Clone

```bash
git clone https://github.com/jeonghonoh/clutt3r-seg
cd clutt3r-seg
```

## Build

Build the runtime image locally:

```bash
docker build --build-arg INSTALL_DUODUOCLIP=1 -t clutt3r-seg:duoduoclip-local .
```

This local build downloads and installs [DuoduoCLIP](https://github.com/3dlg-hcvc/DuoduoCLIP)
from its upstream repository inside the image. Clutt3R-Seg does not redistribute DuoduoCLIP
source code, checkpoints, or Docker images with DuoduoCLIP preinstalled.

Use of DuoduoCLIP is subject to its upstream license and dependency licenses.
Do not publish or redistribute Docker images with DuoduoCLIP preinstalled unless
you have confirmed that all relevant licenses allow it.

## Data

This release includes sample sequences from
[GraspClutter6D](https://sites.google.com/view/graspclutter6d), along with
custom real-world and synthetic sequences.

Each sequence should follow this layout:

```text
samples/<sequence_name>/
  data/
    transforms.json
    images/
    depth/
    instance_masks/
    instance_tree.json
```

`transforms.json` must contain camera intrinsics and per-frame `file_path`,
`depth_file_path`, and `transform_matrix` entries. Instance masks should be
stored as `mask_<frame_id>_<instance_id>.png`.

`data/depth` should contain dense MVSAnywhere inference depth, as used in the
paper pipeline. Raw measured depth with invalid pixels can break back-projection
and geometry consistency.

`data/instance_tree.json` stores precomputed instance-tree assignments for this
source-available release. The public release does not include the internal
instance-tree builder because parts of that implementation depend on closed or
restricted components that cannot be redistributed. The paper describes the
instance-tree construction procedure at the level intended for reproduction;
new sequences need a compatible precomputed artifact. The artifact must match
the exact `instance_masks/` files used by the run.

For update, the selected update frame must have RGB, instance masks, and
update-frame tree entries in `instance_tree.json`. Update-frame depth is only
used for optional depth evaluation when available.

Included sample sequences:

- `samples/sample_seq1`: custom real-world sequence with more than eight frames;
  supports both initial segmentation and update.
- `samples/sample_seq2`: custom real-world sequence with more than eight frames;
  supports both initial segmentation and update.
- `samples/sample_seq3`: difficult sequence from GraspClutter6D.
- `samples/sample_seq4`: easy sequence from GraspClutter6D.
- `samples/sample_seq5`: custom synthetic sequence captured in Isaac Sim.

See `samples/README.md` for additional sequence layout notes.

## Run

Initial segmentation:

```bash
docker run --rm --gpus all \
  -v "$PWD/samples:/workspace/samples" \
  -v "$PWD/.cache:/workspace/.cache" \
  clutt3r-seg:duoduoclip-local \
  bash scripts/run_initial.sh samples/sample_seq2 0,1,2,3,4,5,6,7 "cracker box"
```

Update segmentation and scene update:

```bash
docker run --rm --gpus all \
  -v "$PWD/samples:/workspace/samples" \
  -v "$PWD/.cache:/workspace/.cache" \
  clutt3r-seg:duoduoclip-local \
  bash scripts/run_update.sh samples/sample_seq2 8 1 "chips can"
```

Run update only after initial segmentation has created
`samples/<sequence_name>/state.pkl`.

Outputs are written under `samples/<sequence_name>/output/`:

- `<target_prompt>.ply`: prompt-matched target object point cloud.
- `updated_scene_<update_num>.ply`: full updated scene from the update step.

The 6-DoF grasp pose estimation stage used in the paper is not included in this
release; exported object point clouds can be used as input to external
grasp-pose estimators.

For an interactive shell:

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  -v "$PWD/.cache:/workspace/.cache" \
  clutt3r-seg:duoduoclip-local \
  bash
```

## License

This repository is released under the Clutt3R-Seg Non-Commercial
Source-Available License. See `LICENSE` for details.

Third-party code, checkpoints, datasets, models, and generated assets are
governed by their own licenses. See `THIRD_PARTY.md` for third-party notices.

## Citation

If you found our work useful, please cite:

```bibtex
@inproceedings{noh2026clutt3rseg,
  title={Clutt3R-Seg: Sparse-view 3D Instance Segmentation for Language-grounded Grasping in Cluttered Scenes},
  author={Noh, Jeongho and Rhee, Tai Hyoung and Lee, Eunho and Kim, Jeongyun and Lee, Sunwoo and Kim, Ayoung},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```
