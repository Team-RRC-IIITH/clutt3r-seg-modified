import os
import glob
import re
import cv2
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

from calculate_similarity import SensorFeatureStore

class SpatialHasher:
    """Utility to convert 3D voxel coordinates into unique integer IDs for dictionary hashing."""
    @staticmethod
    def hash_coords(coords: np.ndarray) -> np.ndarray:
        # Standard large primes for 3D spatial hashing
        p1, p2, p3 = 73856093, 19349663, 83492791
        coords = coords.astype(np.int64)
        return np.bitwise_xor(
            np.bitwise_xor(coords[:, 0] * p1, coords[:, 1] * p2), 
            coords[:, 2] * p3
        )
        
class DuoduoCLIPBackends:
    """Mock backend for the CLIP extraction. Replace with your actual model inference."""
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_embedding(self, frame_id: int, mask_id: int, rgb_img: np.ndarray, mask_img: np.ndarray) -> np.ndarray:
        cache_path = self.cache_dir / f"clip_{frame_id:06d}_{mask_id:02d}.npy"
        
        if cache_path.exists():
            return np.load(cache_path)
        
        # TODO: Implement actual cropping and inference here
        # e.g., crop = rgb_img[bounding_box] * mask_img
        # embedding = self.model.encode(crop)
        
        # dummy as of now
        # Simulating normalized output vector (512-dim)
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        np.save(cache_path, embedding)
        return embedding
    
class GenerateInstanceTreePipeline:
    def __init__(self, data_root: str, json_meta_path: str, voxel_size: float = 0.05):
        self.data_root = Path(data_root)
        self.voxel_size = voxel_size
        
        self.feature_store = SensorFeatureStore(epsilon=1e-6)
        self.clip_backend = DuoduoCLIPBackends(cache_dir=self.data_root / "clip_cache")
        
        # Parse the JSON layout to initialize intrinsics and retrieve frame metadata
        self.frames_meta = self._parse_camera_config(json_meta_path)
        
        # Accumulators to calculate global weights w_k at the end
        self.global_voxel_counts: Dict[int, int] = {}
        self.total_scene_points = 0
        
    def _parse_camera_config(self, json_path: str) -> List[Dict[str, any]]:
        """Parses transform.json to construct the intrinsic matrix and extract frames metadata."""
        with open(json_path, 'r') as f:
            meta = json.load(f)
            
        # Construct the 3x3 Camera Matrix K using the parsed properties
        self.intrinsics = np.array([
            [meta['fl_x'],      0.0, meta['cx']],
            [    0.0, meta['fl_y'], meta['cy']],
            [    0.0,      0.0,     1.0]
        ], dtype=np.float32)
        
        print(f"Loaded camera config ({meta['camera_model']}). Dimensions: {meta['w']}x{meta['h']}.")
        return meta['frames']
        
    def parse_mask_filename(self, filepath: str) -> Tuple[int, int]:
        """Extracts frame_id and mask_id from mask_XXXXXX_YY.png"""
        filename = Path(filepath).name
        match = re.search(r'mask_(\d{6})_(\d{2})\.png', filename)
        if not match:
            raise ValueError(f"Filename {filename} does not match expected format.")
        return int(match.group(1)), int(match.group(2))
    
    def project_to_3d_world(self, depth_img: np.ndarray, mask_img: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
        """Projects masked 2D pixels into 3D points using camera intrinsics."""
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        
        # Get coordinates of valid mask pixels with valid depth
        valid_pixels = (mask_img > 0) & (depth_img > 0)
        v, u = np.where(valid_pixels)
        z = depth_img[valid_pixels]
        
        # Pinhole camera projection math
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_cam = np.stack((x, y, z), axis=-1) # Nx3
        
        if len(points_cam) == 0:
            return points_cam
        
        # transform to global world coordinates via matrix multiplication
        # Convert to homogeneous coordinates (Nx4) by appending a column of ones
        ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
        points_hom = np.hstack((points_cam, ones)) # Nx4
        
        # Rigid body transformations: P_world = T_c2w * P_hom
        points_world = (t_c2w @ points_hom.T).T[:, :3] # Nx3
        return points_world
    
    def process_frame_mask(self, frame_meta: Dict[str, any], mask_filepath: str, global_mask_id: int):
        """Processes a single mask, projects to global world coordinates, and aggregates data."""
        frame_id, mask_id = self.parse_mask_filename(mask_filepath)
        
        # Use transform_matrix as a numpy float32 array
        t_c2w = np.array(frame_meta['transform_matrix'], dtype=np.float32)
        
        mask_path = self.data_root / f"instance_masks/mask_{frame_id:06d}_{mask_id:02d}.png"
        depth_path = self.data_root / f"depth/depth_{frame_id:06d}.png"
        rgb_path = self.data_root / f"images/image_{frame_id:06d}.png"
        
        if not depth_path.exists():
            print(f"Warning: Missing depth image at {depth_path}. Skipping mask F{frame_id:02d}-M{mask_id:02d}.")
            return
        if not rgb_path.exists():
            print(f"Warning: Missing RGB image at {rgb_path}. Skipping mask F{frame_id:02d}-M{mask_id:02d}.")
            return
        
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # Depth is often saved as 16-bit PNG (e.g., millimeters). Convert to meters.
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if raw_depth is None:
            print(f"Warning: Failed to read depth image at {depth_path} (corrupted). Skipping mask.")
            return
        depth_img = raw_depth / 1000.0
        
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            print(f"Warning: Failed to read RGB image at {rgb_path} (corrupted). Skipping mask.")
            return
        
        # extract visual features
        embedding = self.clip_backend.get_embedding(frame_id, mask_id, rgb_img, mask_img)
        self.feature_store.mask_embeddings[global_mask_id] = embedding
        
        # Project Points into the Global Scene Frame
        points_3d = self.project_to_3d_world(depth_img, mask_img, t_c2w)
        if len(points_3d) == 0:
            return
        
        #  Discretize Spatial Structure into Super-Voxels
        voxel_coords = np.floor(points_3d / self.voxel_size)
        voxel_ids = SpatialHasher.hash_coords(voxel_coords)
        
        # count points per voxel for this specific mask
        unique_voxels, counts = np.unique(voxel_ids, return_counts=True)
        
        self.feature_store.mask_occupancies[global_mask_id] = {}
        for v_id, count in zip(unique_voxels, counts):
            self.feature_store.mask_occupancies[global_mask_id][v_id] = count
            
            # Update global accumulators for w_k calculation
            self.global_voxel_counts[v_id] = self.global_voxel_counts.get(v_id, 0) + count
            self.total_scene_points += count
            
    def finalize_occupancies_and_weights(self):
        """
        Converts raw point counts into the formal ratios required by the paper:
        o_i(k) [0,1] and normalized w_k.
        """
        print(f"Finalizing map with {len(self.global_voxel_counts)} unique super-voxels...")
        
        for v_id, total_pts in self.global_voxel_counts.items():
            self.feature_store.global_voxel_weights[v_id] = total_pts / self.total_scene_points
            
        for mask_id, voxel_data in self.feature_store.mask_occupancies.items():
            for v_id, mask_pts in voxel_data.items():
                total_pts_in_voxel = self.global_voxel_counts[v_id]
                occupancy_ratio = mask_pts / total_pts_in_voxel
                self.feature_store.mask_occupancies[mask_id][v_id] = occupancy_ratio
                
    def run(self) -> SensorFeatureStore:
        mask_files = glob.glob(str(self.data_root / "instance_masks/mask_*.png"))
        mask_files.sort()
        
        print(f"Found {len(mask_files)} instance masks. Processing...")
        
        # We need a unique global ID across all frames for the clustering graph
        # This assumes mask_id resets per frame. If it doesn't, you can just use mask_id.
        global_mask_id = 0
        
        for idx, frame_meta in enumerate(self.frames_meta):
            mask_pattern = str(self.data_root / f"instance_masks/mask_{idx:06d}_*.png")
            mask_files = glob.glob(mask_pattern)
            mask_files.sort()
            
            for mask_path in mask_files:
                self.process_frame_mask(frame_meta, mask_path, global_mask_id)
                global_mask_id += 1
            
        self.finalize_occupancies_and_weights()
        print("Preprocessing complete. SensorFeatureStore populated.")
        return self.feature_store
    
if __name__ == "__main__":
    pipeline = GenerateInstanceTreePipeline(
        data_root="/scratch2/clutt3r-seg-modified/samples/sample_seq2/data",
        json_meta_path="/scratch2/clutt3r-seg-modified/samples/sample_seq2/data/transforms.json",
        voxel_size=0.05
    )
    
    populated_store = pipeline.run()
    
    output_path = "feature_store_cache.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(populated_store, f)
    print(f"Saved populated global coordinate SensorFeatureStore to {output_path}")