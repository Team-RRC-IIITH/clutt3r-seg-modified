import glob
import re
import pickle
from pathlib import Path

from calculate_similarity import SensorFeatureStore
from hierarchy_based_instance_mask_grouping import Clutt3RSegClustering

def parse_mask_filename(filepath: str):
    """Extracts frame_id and mask_id from mask_XXXXXX_YY.png"""
    filename = Path(filepath).name
    match = re.search(r'mask_(\d{6})_(\d{2})\.png', filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected format.")
    return int(match.group(1)), int(match.group(2))

def main():
    data_root = Path("samples/sample_seq1/data")
    feature_store_path = "feature_store_cache.pkl"
    output_json_path = "instance_tree.json"

    # Hyperparameters
    # tau_spat: Minimum Weighted Jaccard index to merge based purely on 3D geometry
    # tau_sem: Minimum Cosine Similarity to merge based on CLIP visual features
    tau_spat = 0.45 
    tau_sem = 0.70  

    print(f"1. Loading populated SensorFeatureStore from {feature_store_path}...")
    try:
        with open(feature_store_path, 'rb') as f:
            feature_store = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {feature_store_path} not found. Please run Phase 2 first.")
        return

    print("2. Building leaf list from instance masks...")
    mask_files = glob.glob(str(data_root / "instance_masks/mask_*.png"))
    mask_files.sort() 

    leaves = []
    # global_mask_id acts as the unique 'node_id' in the graph
    for global_mask_id, filepath in enumerate(mask_files):
        frame_id, mask_id = parse_mask_filename(filepath)
        leaves.append((global_mask_id, frame_id, mask_id))
        
    print(f"   -> Discovered {len(leaves)} base instance masks.")

    print(f"3. Initializing Clutt3RSegClustering (tau_spat={tau_spat}, tau_sem={tau_sem})...")
    clusterer = Clutt3RSegClustering(tau_spat=tau_spat, tau_sem=tau_sem)

    print("4. Constructing Leaf Graph & calculating pairwise similarities...")
    clusterer.construct_leaf_graph(
        leaves=leaves,
        feature_store=feature_store
    )

    print("5. Executing Two-Stage Similarity Grouping (Agglomerative Clustering)...")
    clusterer.group_by_similarity()

    print(f"6. Exporting results to {output_json_path}...")
    # (Optional) If you have the per-frame hierarchical trees T_f from SAM, load and pass them here
    # mock_per_frame_trees = {"parent_of": {...}, "descendant_leaves": {...}}
    
    clusterer.export_to_json(
        output_path=output_json_path,
        per_frame_trees=None # Replace with mock_per_frame_trees if available
    )
    
    print("Pipeline Complete.")

if __name__ == "__main__":
    main()