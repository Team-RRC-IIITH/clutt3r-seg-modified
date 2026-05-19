"""Update-frame association and scene update entry point for Clutt3R-Seg."""


from pathlib import Path


import argparse
import pickle
import json
import glob

import numpy as np
import torch
from PIL import Image
from collections import defaultdict


from .utils import (
    refine_masks_by_depth, _bbox, _letterbox,
    export_prompt_pointcloud,

)
from .clip_backends.duoduo import DEFAULT_DUODUO_CHECKPOINT, load_duoduo_clip
from .tree_artifacts import leaf_id as make_leaf_id, load_instance_tree_artifact, load_update_tree

from .scene_updater import instance_wise_scene_update

TOTAL_STAGES = 4


def _print_stage(step: int, message: str) -> None:
    print(f"[update {step}/{TOTAL_STAGES}] {message}", flush=True)


def validate_update_dataset(args) -> None:
    database_root = args.experiment_data_dir
    state_path = database_root / "state.pkl"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing {state_path}. Run initial segmentation before update segmentation."
        )

    scene_data_path = database_root / "data"
    transforms_path = scene_data_path / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing {transforms_path}.")

    meta = json.load(open(transforms_path))
    frames_meta = meta.get("frames", [])
    if args.update_fid < 0 or args.update_fid >= len(frames_meta):
        raise ValueError(
            f"Update frame {args.update_fid} is outside transforms.json range 0..{len(frames_meta) - 1}."
        )

    frame_meta = frames_meta[args.update_fid]
    rgb_path = scene_data_path / Path(frame_meta["file_path"]).with_suffix(".png")
    instance_mask_path = scene_data_path / "instance_masks"
    instance_tree_path = scene_data_path / "instance_tree.json"
    mask_files = list(instance_mask_path.glob(f"mask_{args.update_fid:06d}_*.png"))

    missing = []
    if not rgb_path.exists():
        missing.append(f"missing RGB image: {rgb_path}")
    if not mask_files:
        missing.append(
            f"missing Grounded-SAM masks: {instance_mask_path}/mask_{args.update_fid:06d}_*.png"
        )
    if not instance_tree_path.exists():
        missing.append(f"missing precomputed instance-tree artifact: {instance_tree_path}")

    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Update frame {args.update_fid} is not fully preprocessed.\n  - {details}"
        )


def export_update_target_pointcloud(
    scene_update_data: dict,
    target_id: int,
    output_dir: Path,
    target_prompt: str,
) -> Path:
    if target_id < 0:
        raise ValueError("No prompt-matched target instance was selected for export.")

    instance_pcds = scene_update_data.get("instance_pcds", {})
    if target_id not in instance_pcds:
        raise KeyError(f"Target instance {target_id} is missing from scene update output.")

    points, colors = instance_pcds[target_id]
    return export_prompt_pointcloud(points, colors, output_dir, target_prompt)


def update_segmentation_consistency(
    args,
    data: dict,
    clip,
    G_CLIP_MATCHING_THR: float = 0.65,
):
    _print_stage(3, "Matching update masks to existing instances")
    instance_embeddings = data["instance_embeddings"]
    ground_embedding = data["mean_ground_embedding"]
    node2inst = data["node2inst"]

    database_root = args.experiment_data_dir
    scene_data_path = database_root / "data"
    instance_mask_path = scene_data_path / "instance_masks"
    meta = json.load(open(scene_data_path / "transforms.json"))
    frames_meta = meta["frames"]

    update_indices = args.update_fid
    if isinstance(update_indices, int):
        update_indices = [update_indices]

    update_frames_meta = [frames_meta[i] for i in update_indices]

    inst2bin_mask = {}
    raw_images = {}

    for local_idx, fr in enumerate(update_frames_meta):
        f_idx_global = update_indices[local_idx]
        rgb_path   = scene_data_path / Path(fr["file_path"]).with_suffix(".png")
        if not rgb_path.exists():
            continue
        raw_images[f_idx_global] = Image.open(rgb_path).convert("RGB")
        mask_glob = glob.glob(str((instance_mask_path) / f"mask_{f_idx_global:06d}_*.png"))
        mask_refined = refine_masks_by_depth(mask_glob, None, mode = "naive")

        for lid, mask in mask_refined.items():


            if mask.sum() > 500:
                y0, y1, x0, x1 = _bbox(mask)
                crop_rgb  = raw_images[f_idx_global].crop((x0, y0, x1 + 1, y1 + 1))
                crop_mask = (mask[y0:y1 + 1, x0:x1 + 1] * 255).astype(np.uint8)
                crop_arr  = _letterbox(crop_rgb, crop_mask)
                inst2bin_mask[(f_idx_global, lid)] = crop_arr



    emb_feats = {}

    for (fidx, lid), crop_arr in inst2bin_mask.items():
        assert crop_arr.shape == (224, 224, 3), f"Crop shape mismatch: {crop_arr.shape} for (fidx={fidx}, lid={lid})"

        emb = clip.encode_image(crop_arr[None, ...]).squeeze(0).cpu().numpy()
        emb /= np.linalg.norm(emb)


        if ground_embedding is not None:
            similarity = np.dot(emb, ground_embedding)

            if similarity > G_CLIP_MATCHING_THR:
                continue
        emb_feats[(fidx, lid)] = emb


    tree_artifact = load_instance_tree_artifact(args.experiment_data_dir)
    update_tree = load_update_tree(tree_artifact, args.update_fid)
    leaf_nodes = [
        (fidx, lid)
        for fidx, lid in update_tree["leaf_nodes"]
        if (fidx, lid) in emb_feats
    ]
    parent_of = update_tree["parent_of"]
    descendant_leaves = update_tree["descendant_leaves"]
    if not leaf_nodes:
        raise RuntimeError(
            f"No update leaf nodes from instance_tree.json matched frame {args.update_fid}."
        )
    SIM_HIGH = 0.60
    SIM_LOW  = 0.45


    leaf_best: dict[str, tuple[int,float]] = {}
    for fidx, lid in leaf_nodes:
        leaf_id  = f"L_{fidx:02d}_{lid:02d}"
        emb_leaf = emb_feats[(fidx, lid)]
        best_inst, best_sim = None, 0.0
        for inst_id, emb_init in instance_embeddings.items():
            clip_sim = float(np.dot(emb_leaf, emb_init))


            if clip_sim > best_sim:
                best_inst, best_sim = inst_id, clip_sim
        if best_inst is not None:
            leaf_best[leaf_id] = (best_inst, best_sim)


    matched_leaves: dict[str, tuple[int,float]] = {}
    matched_insts_by_frame: dict[int, set[int]] = defaultdict(set)

    for leaf_id, (inst_id, sim) in sorted(
            leaf_best.items(), key=lambda x: x[1][1], reverse=True):
        fidx = int(leaf_id[2:4])

        if sim >= SIM_HIGH and inst_id not in matched_insts_by_frame[fidx]:
            matched_leaves[leaf_id] = (inst_id, sim)
            matched_insts_by_frame[fidx].add(inst_id)


    tentative_leaves: list[tuple[str, int, float]] = []

    for fidx, lid in leaf_nodes:
        leaf_id = f"L_{fidx:02d}_{lid:02d}"
        if leaf_id in matched_leaves:
            continue

        inst_id, sim_leaf = leaf_best.get(leaf_id, (None, 0.0))
        if sim_leaf < SIM_LOW:

            continue

        tree_parent = parent_of.get(make_leaf_id(fidx, lid))
        if tree_parent is None:

            tentative_leaves.append((leaf_id, inst_id, sim_leaf))
            continue

        p_fidx = int(tree_parent[2:4])
        p_lid = int(tree_parent[5:])
        parent_leaf_id = tree_parent
        descendants = set(descendant_leaves.get(tree_parent, []))
        if any(d in matched_leaves for d in descendants):

            tentative_leaves.append((leaf_id, inst_id, sim_leaf))
            continue


        emb_parent = emb_feats.get((p_fidx, p_lid))
        if emb_parent is None or inst_id is None:
            raise ValueError(f"  [ERROR] parent embedding not found: {tree_parent}")
        clip_p   = float(np.dot(emb_parent, instance_embeddings[inst_id]))


        sim_par  = clip_p

        if inst_id not in matched_insts_by_frame[p_fidx]:
            if SIM_HIGH <= sim_par:

                matched_leaves[parent_leaf_id] = (inst_id, sim_par)
                matched_insts_by_frame[p_fidx].add(inst_id)
            elif SIM_LOW <= sim_par and sim_leaf < sim_par:

                matched_leaves[parent_leaf_id] = (inst_id, sim_par)
                matched_insts_by_frame[p_fidx].add(inst_id)
            else:

                matched_leaves[leaf_id] = (inst_id, sim_leaf)
                matched_insts_by_frame[fidx].add(inst_id)
    tentative_leaves.sort(key=lambda x: x[2], reverse=True)

    for leaf_id, inst_id, sim_leaf in tentative_leaves:
        fidx = int(leaf_id[2:4])


        if inst_id in matched_insts_by_frame[fidx]:
            continue


        matched_leaves[leaf_id] = (inst_id, sim_leaf)
        matched_insts_by_frame[fidx].add(inst_id)

    updated_node2inst = node2inst.copy()
    matching_info = {}
    for node_id, (inst_id, _sim) in matched_leaves.items():
        fidx = int(node_id[2:4])
        lid = int(node_id[5:])
        updated_node2inst[(fidx, lid)] = inst_id

        if inst_id < 99:
            matching_info.setdefault(inst_id, []).append((fidx, lid))

    update_data = {
        "instance_embeddings": data["instance_embeddings"],
        "mean_ground_embedding": data["mean_ground_embedding"],
        "inst_colors_u8": data["inst_colors_u8"],

        "node2inst": updated_node2inst,
        "matching_info": matching_info
    }
    return update_data


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _print_stage(1, "Validating update inputs")
    validate_update_dataset(args)

    _print_stage(2, "Loading state and DuoduoCLIP model")
    state_path = args.experiment_data_dir / "state.pkl"
    with open(state_path, 'rb') as f:
        previous_state = pickle.load(f)

    clip = load_duoduo_clip(
        checkpoint=args.clip_checkpoint,
        device=device,
        duoduo_root=args.duoduo_root,
    )


    updated_data = update_segmentation_consistency(
        args,
        data=previous_state,
        clip=clip,
    )

    if not updated_data["matching_info"]:
        raise ValueError(
            f"No instances were matched for update frame {args.update_fid}. "
            "Check that the update frame has valid Grounded-SAM masks and that they overlap the tracked scene."
        )


    _print_stage(4, "Updating scene, exporting target, and saving state")
    target_prompt_emb = clip.encode_text(args.prompt)
    target_prompt_emb = target_prompt_emb.squeeze(0).cpu().numpy()


    inst2target_sim = {}
    for inst_id in updated_data['matching_info'].keys():
        inst2target_sim[inst_id] = np.dot(
            updated_data['instance_embeddings'][inst_id],
            target_prompt_emb
        )


    target_id = max(inst2target_sim, key=inst2target_sim.get) if inst2target_sim else -1
    if target_id != -1 and inst2target_sim[target_id] < 0.1:
        target_id = -1


    scene_update_data = instance_wise_scene_update(
        args,
        initial_data=previous_state,
        matching_info=updated_data['matching_info'],
        use_original_pcd=True,
        original_data=previous_state.get("original_data"),
        target_id=target_id,
        update=args.update_num,
        visualize=False,
        verbose=False
    )
    export_update_target_pointcloud(
        scene_update_data,
        target_id=target_id,
        output_dir=args.experiment_data_dir / "output",
        target_prompt=args.prompt,
    )

    update_key = f"update_{args.update_num}"
    if 'updates' not in previous_state:
        previous_state['updates'] = {}
    previous_state['updates'][update_key] = {"matching_info": updated_data['matching_info']}

    with open(state_path, 'wb') as f:
        pickle.dump(previous_state, f)


if __name__ == "__main__":
    pa = argparse.ArgumentParser()


    pa.add_argument("--experiment_data_dir", type=Path, required=True, help="Directory where state.pkl is located.")
    pa.add_argument("--update_fid", type=int, required=True, help="Update frame index.")
    pa.add_argument("--update_num", type=int, required=True, help="Update number.")
    pa.add_argument("--prompt", type=str, default="", help="Text prompt for new instance (if any).")
    pa.add_argument("--duoduo-root", "--duoduo_root", dest="duoduo_root", type=Path, default=None,
                   help="Path to an external DuoduoCLIP checkout. Defaults to DUODUOCLIP_ROOT.")
    pa.add_argument("--clip-checkpoint", "--clip_checkpoint", dest="clip_checkpoint", type=str,
                   default=DEFAULT_DUODUO_CHECKPOINT,
                   help="DuoduoCLIP checkpoint filename or path accepted by the upstream wrapper.")

    args = pa.parse_args()

    main(args)
