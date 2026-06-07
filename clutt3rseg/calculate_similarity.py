import numpy as np
from typing import Dict, Tuple

class SensorFeatureStore:
    """
    A data store to hold the super-voxel occupancies, global voxel weights, 
    and CLIP embeddings for all masks in the sequence.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        
        # global_voxel_weights[k] = w_k (normalized weight proportional to point count)
        self.global_voxel_weights: Dict[int, float] = {}
        
        # mask_occupancies[mask_id] = {voxel_id: occupancy_ratio_in_range_0_to_1}
        self.mask_occupancies: Dict[int, Dict[int, float]] = {}
        
        # mask_embeddings[mask_id] = numpy array (D,)
        self.mask_embeddings: Dict[int, np.ndarray] = {}
        
    def calc_spat_sim(self, u: int, v: int) -> float:
        """
        Implementation of the Weighted Jaccard Index based on super-voxel occupancy.
        """
        occ_u = self.mask_occupancies.get(u, {})
        occ_v = self.mask_occupancies.get(v, {})
        
        # Optimization: We only need to iterate over voxels that exist in u OR v.
        # If a voxel k is in neither, max(0, 0) = 0 and min(0, 0) = 0, contributing 0.
        active_voxels = set(occ_u.keys()).union(set(occ_v.keys()))
        
        numerator_sum = 0.0
        denominator_sum = 0.0
        
        for k in active_voxels:
            # Get occupancy ratios o_u(k) and o_v(k), defaulting to 0 if the mask isn't in that voxel
            o_u_k = occ_u.get(k, 0.0)
            o_v_k = occ_v.get(k, 0.0)
            
            # the super-voxel weight w_hat_k
            w_k = self.global_voxel_weights.get(k, 1.0)
            
            # calculate the weighted min (intersection) and max (union)
            numerator_sum += w_k * min(o_u_k, o_v_k)
            denominator_sum += w_k * max(o_u_k, o_v_k)
            
        # S_spatial(u, v) = Sum(min) / (Sum(max) + epsilon)
        s_spatial = numerator_sum / (denominator_sum + self.epsilon)
        return float(s_spatial)
    
    def calc_sem_sim(self, u: int, v: int) -> float:
        """
        Implementation of the Semantic Similarity using the inner product of CLIP embeddings.
        Assumes e_u and e_v are already L2-normalized.
        """
        e_u = self.mask_embeddings.get(u)
        e_v = self.mask_embeddings.get(v)
        
        if e_u is None or e_v is None:
            return 0.0
        
        # S_semantic(u, v) = <e_u, e_v> (Dot product of normalized vectors)
        # Note: If your CLIP vectors are NOT normalized beforehand, use: 
        # s_semantic = np.dot(e_u, e_v) / (np.linalg.norm(e_u) * np.linalg.norm(e_v))
        s_semantic = np.dot(e_u, e_v)
        
        # Clip to [-1.0, 1.0] to handle floating point inaccuracies
        return float(np.clip(s_semantic, -1.0, 1.0))