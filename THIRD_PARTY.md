# Third-Party Code and Models

The Clutt3R-Seg license covers only the Clutt3R-Seg code and documentation in this repository. Third-party code, checkpoints, datasets, and generated assets remain under their own licenses.

This source repository does not vendor DuoduoCLIP source code. Clutt3R-Seg imports it through a thin adapter in `clutt3rseg/clip_backends/duoduo.py`.

- Repository: [3dlg-hcvc/DuoduoCLIP](https://github.com/3dlg-hcvc/DuoduoCLIP)
- Checkpoint: `Four_1to6F_bs1600_LT6.ckpt`

The Docker build can optionally clone and install DuoduoCLIP into a local image with `--build-arg INSTALL_DUODUOCLIP=1`. Do not redistribute that image unless the upstream DuoduoCLIP license and dependency licenses permit it, or you have separate permission. Large model weights, datasets, and generated assets should not be committed to this repository.

MVSAnywhere and Grounded-SAM are treated as external preprocessing dependencies in this release. Run them from their upstream repositories or from local lab copies, then copy their generated outputs into a sequence with `scripts/adopt_preprocess_outputs.py`.

## GraspClutter6D Sample Sequences

Some sample sequences, including `samples/sample_seq3` and
`samples/sample_seq4`, are derived from the GraspClutter6D dataset.

- Project page: [GraspClutter6D](https://sites.google.com/view/graspclutter6d)
- Dataset license: Creative Commons Attribution-ShareAlike 4.0
  (CC BY-SA 4.0)

These dataset materials are not licensed under the Clutt3R-Seg license. They
remain under the upstream GraspClutter6D dataset license. The samples were
converted into the Clutt3R-Seg sample layout and include precomputed artifacts
needed by this release.
