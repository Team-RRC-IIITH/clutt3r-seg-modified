"""Load precomputed instance-tree artifacts used by the release pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ARTIFACT_NAME = "instance_tree.json"


def leaf_id(frame: int, mask: int) -> str:
    return f"L_{int(frame):02d}_{int(mask):02d}"


def parse_leaf_id(value: str) -> tuple[int, int]:
    try:
        _, frame, mask = value.split("_")
        return int(frame), int(mask)
    except ValueError as exc:
        raise ValueError(f"Invalid leaf id '{value}'. Expected L_XX_YY.") from exc


def load_instance_tree_artifact(experiment_data_dir: Path) -> dict:
    artifact_path = Path(experiment_data_dir) / "data" / ARTIFACT_NAME
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing {artifact_path}. This release does not include the "
            "instance-tree builder; provide a precomputed instance_tree.json "
            "artifact for the sequence."
        )

    with artifact_path.open("r") as f:
        artifact = json.load(f)

    if artifact.get("schema_version") != 1:
        raise ValueError(f"Unsupported instance-tree artifact schema in {artifact_path}.")
    return artifact


def load_initial_leaf2inst(artifact: dict, initial_idx: list[int]) -> dict[tuple[int, int], int]:
    initial = artifact.get("initial", {})
    artifact_idx = initial.get("initial_idx", [])
    if artifact_idx and list(map(int, artifact_idx)) != list(map(int, initial_idx)):
        raise ValueError(
            "The instance-tree artifact was generated for initial_idx="
            f"{artifact_idx}, but the command requested initial_idx={list(initial_idx)}."
        )

    leaf2inst = {}
    for item in initial.get("leaf2inst", []):
        key = (int(item["frame"]), int(item["mask"]))
        leaf2inst[key] = int(item["instance"])

    if not leaf2inst:
        raise ValueError("The instance-tree artifact does not contain initial leaf assignments.")
    return leaf2inst


def load_initial_ground_embedding(artifact: dict) -> np.ndarray | None:
    value = artifact.get("initial", {}).get("mean_ground_embedding")
    if value is None:
        return None

    emb = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm <= 0:
        return None
    return emb / norm


def load_update_tree(artifact: dict, update_frame: int) -> dict:
    update = artifact.get("updates", {}).get(str(int(update_frame)))
    if update is None:
        raise FileNotFoundError(
            f"The instance-tree artifact does not contain update frame {update_frame}."
        )

    leaf_nodes = []
    for item in update.get("leaf_nodes", []):
        leaf_nodes.append((int(item["frame"]), int(item["mask"])))

    parent_of = {
        str(child): str(parent)
        for child, parent in update.get("parent_of", {}).items()
    }
    descendant_leaves = {
        str(parent): [str(leaf) for leaf in leaves]
        for parent, leaves in update.get("descendant_leaves", {}).items()
    }

    if not leaf_nodes:
        raise ValueError(f"No update leaf nodes are stored for frame {update_frame}.")

    return {
        "leaf_nodes": leaf_nodes,
        "parent_of": parent_of,
        "descendant_leaves": descendant_leaves,
    }
