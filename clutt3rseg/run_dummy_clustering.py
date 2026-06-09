import numpy as np

# Import your classes from the respective files
from calculate_similarity import SensorFeatureStore
from hierarchy_based_instance_mask_grouping import Clutt3RSegClustering

def main():
    print("1. Initializing Sensor Feature Store...")
    feature_store = SensorFeatureStore(epsilon=1e-6)

    # --- MOCK DATA INSERTION ---
    # In your real pipeline, you would loop through your frames here, 
    # running SAM and CLIP to populate these dictionaries.
    
    # Mock Voxel Weights (w_k)
    feature_store.global_voxel_weights = {101: 0.8, 102: 0.5, 103: 0.2, 104: 0.9}

    # Mock Mask Occupancies (mask_id -> {voxel_id: ratio})
    feature_store.mask_occupancies[1] = {101: 0.9, 102: 0.4}  # Mask 1 in Frame 0
    feature_store.mask_occupancies[2] = {101: 0.85, 103: 0.7} # Mask 2 in Frame 1
    feature_store.mask_occupancies[3] = {104: 0.95}           # Mask 3 in Frame 1 (Unrelated object)

    # Mock CLIP Embeddings (L2 Normalized)
    embed_1 = np.random.rand(512)
    embed_2 = np.random.rand(512)
    embed_3 = np.random.rand(512)
    
    # Simulating that Mask 1 and Mask 2 are semantically similar
    embed_2 = embed_1 + np.random.normal(0, 0.1, 512) 

    feature_store.mask_embeddings[1] = embed_1 / np.linalg.norm(embed_1)
    feature_store.mask_embeddings[2] = embed_2 / np.linalg.norm(embed_2)
    feature_store.mask_embeddings[3] = embed_3 / np.linalg.norm(embed_3)

    print("2. Initializing Clustering Algorithm...")
    # Set your thresholds based on your sensor calibration
    clusterer = Clutt3RSegClustering(tau_spat=0.5, tau_sem=0.75)

    # Define the base leaves: (node_id, frame_id)
    # Mask 1 is from frame 0. Masks 2 and 3 are from frame 1.
    mock_leaves = [(1, 0), (2, 1), (3, 1)] 

    print("3. Building Leaf Graph...")
    # Inject the feature store directly
    clusterer.construct_leaf_graph(
        leaves=mock_leaves,
        feature_store=feature_store  
    )

    print("4. Running Two-Stage Similarity Grouping...")
    clusterer.group_by_similarity()

    # Optional: If you generated T_f (the hierarchical trees) upstream, define them here
    mock_per_frame_trees = {
        "parent_of": {"L_00_01": "L_00_Root"},
        "descendant_leaves": {"L_00_Root": ["L_00_01"]}
    }

    output_file = "instance_tree.json"
    print(f"5. Exporting results to {output_file}...")
    clusterer.export_to_json(output_file, per_frame_trees=mock_per_frame_trees)

if __name__ == "__main__":
    main()