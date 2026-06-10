import argparse
import json
from collections import defaultdict
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def load_instance_tree(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def build_frame_graphs(tree: dict) -> dict[int, nx.DiGraph]:
    initial = tree.get("initial", {})
    frame_graphs: dict[int, nx.DiGraph] = {}
    frame_to_leaves: dict[int, list[dict]] = defaultdict(list)

    for leaf in initial.get("leaf2inst", []):
        frame_to_leaves[int(leaf["frame"])].append(leaf)

    for frame, leaves in sorted(frame_to_leaves.items()):
        graph = nx.DiGraph()
        for leaf in leaves:
            frame_id = int(leaf["frame"])
            mask_id = int(leaf["mask"])
            instance_id = int(leaf["instance"])
            leaf_node = f"leaf_{frame_id}_{mask_id}"
            inst_node = f"inst_{instance_id}"

            graph.add_node(leaf_node, label=f"L:{frame_id}-{mask_id}", kind="leaf")
            graph.add_node(inst_node, label=f"I:{instance_id}", kind="instance")
            graph.add_edge(leaf_node, inst_node)

        frame_graphs[frame] = graph

    return frame_graphs


def draw_frame_graphs(
    frame_graphs: dict[int, nx.DiGraph],
    output_path: Path,
    title: str | None = None,
) -> None:
    num_frames = len(frame_graphs)
    cols = min(4, num_frames)
    rows = ceil(num_frames / cols)
    fig_width = max(12, cols * 4)
    fig_height = max(3 * rows, 6)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    if isinstance(axes, plt.Axes):
        axes = [[axes]]
    else:
        axes = axes.reshape(rows, cols)

    frame_items = list(frame_graphs.items())
    empty_ax = fig.add_subplot(rows, cols, rows * cols)

    for index, (frame, graph) in enumerate(frame_items):
        row = index // cols
        col = index % cols
        ax = axes[row][col]

        leaf_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "leaf"]
        instance_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "instance"]

        pos = {}
        if leaf_nodes:
            for i, node in enumerate(sorted(leaf_nodes)):
                pos[node] = (0.0, 1.0 - (i + 1) / (len(leaf_nodes) + 1))
        if instance_nodes:
            for j, node in enumerate(sorted(instance_nodes)):
                pos[node] = (1.0, 1.0 - (j + 1) / (len(instance_nodes) + 1))

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=leaf_nodes,
            node_color="#dd8452",
            node_size=250,
            alpha=0.9,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=instance_nodes,
            node_color="#4c72b0",
            node_size=450,
            alpha=0.9,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            arrowstyle="-|>",
            arrowsize=8,
            edge_color="#555555",
            alpha=0.7,
        )

        labels = {node: data.get("label", node) for node, data in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)

        ax.set_title(f"Frame {frame}")
        ax.axis("off")

    # hide unused subplots
    for index in range(len(frame_items), rows * cols):
        row = index // cols
        col = index % cols
        axes[row][col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an instance_tree.json artifact.")
    parser.add_argument("json_path", type=Path, help="Path to instance_tree.json")
    parser.add_argument("--output", type=Path, default=Path("instance_tree.png"), help="Output image file path")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    args = parser.parse_args()

    tree = load_instance_tree(args.json_path)
    frame_graphs = build_frame_graphs(tree)

    title = f"Instance Tree: {args.json_path.name}"
    draw_frame_graphs(frame_graphs, args.output, title=title)

    print(f"Saved instance tree visualization to {args.output}")
    if args.show:
        img = plt.imread(args.output)
        plt.figure(figsize=(14, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
