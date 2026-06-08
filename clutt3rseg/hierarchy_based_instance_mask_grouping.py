import heapq
import json
from typing import List, Dict, Tuple, Set, Callable
from calculate_similarity import SensorFeatureStore

class Node:
    def __init__(self, node_id: int, frame_id: int, mask_id: int = None):
        self.node_id = node_id
        self.frame_id = frame_id
        self.leaf_count = 1
        self.active = True
        self.mask_id = mask_id
        
        # Tracking the original base leaves: [{'frame': f, 'mask': m}]
        if mask_id is not None:
            self.leaves = [{"frame": frame_id, "mask": mask_id}]
        else:
            # Super-nodes start empty and are populated during the merge
            self.leaves = []
        
class Clutt3RSegClustering:
    def __init__(self, tau_spat: float, tau_sem: float):
        self.tau_spat = tau_spat
        self.tau_sem = tau_sem
        
        self.nodes: Dict[int, Node] = {}
        
        # adj[u][v] = {'splat': float, 'sem': float}
        self.adj: Dict[int, Dict[int, Dict[str, float]]] = {}
        
        # max-heaps for fast argmax retrival: (-weight, id1, id2)
        self.spat_heap = []
        self.sem_heap = []
        self.next_node_id = 0
        
    def construct_leaf_graph(self, 
                             leaves: List[Tuple[int, int]], # List of (node_id, frame_id)
                             feature_store: "SensorFeatureStore"):
        """
        Lines 4-10: build initial leaf graph from frame trees 
        """
        
        # initialize nodes
        for node_id, frame_id in leaves:
            self.nodes[node_id] = Node(node_id, frame_id, mask_id)
            self.adj[node_id] = {}
            self.next_node_id = max(self.next_node_id, node_id) + 1
            
        node_ids = list(self.nodes.keys())
        
        # iterate over all unique pairs (V choose 2)
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                u, v = node_ids[i], node_ids[j]
                
                # condition : phi(u) != phi(v)
                if self.nodes[u].frame_id != self.nodes[v].frame_id:
                    s_spat = feature_store.calc_spat_sim(u, v)
                    s_sem = feature_store.calc_sem_sim(u, v)
                    
                    self._add_edge(u, v, s_splat, s_sem)
                    
    def _add_edge(self, u: int, v: int, s_splat: float, s_sem: float):
        """Helper to safely add edges to adjacency list and push to heaps."""
        self.adj[u][v] = {'spat': s_spat, 'sem': s_sem}
        self.adj[v][u] = {'spat': s_spat, 'sem': s_sem}
        
        # Python heapq is a min-heap, so we multiply weights by -1 for max-heap behavior
        heapq.heappush(self.spat_heap, (-s_spat, u, v))
        heapq.heappush(self.sem_heap, (-s_sem, u, v))
        
    def group_and_rewire(self, u: int, v: int):
        """
        Lines 22-31: Group u and v into w, and rewire edges using average linkage.
        """
        node_u = self.nodes[u]
        node_v = self.nodes[v]
        
        # create super node w
        w = self.next_node_id
        self.next_node_id += 1
        
        # In a real system, you might want w's frame_id to be a set, 
        # but for clustering purposes, we just need to track the merged leaf count
        node_w = Node(w, frame_id = -1)
        node_w.leaf_count = node_u.leaf_count + node_v.leaf_count
        
        # merge the physical leaf lists
        node_w.leaves = node_u.leaves + node_v.leaves
        
        self.nodes[w] = node_w
        self.adj[w] = {}
        
        # get all unique neighbors of u and v, excluding u and v themselves
        neighbors_u = set(self.adj[u].keys())
        neighbors_v = set(self.adj[v].keys())
        all_neighbors = (neighbors_u | neighbors_v) - {u, v}
        
        for x in all_neighbors:
            # Dynamic Average Linkage Calculation (Math optimization of Lines 26-27)
            # s_new = (count_u * s_ux + count_v * s_vx) / (count_u + count_v)
            
            s_spat_ux = self.adj[u].get(x, {}).get('spat', 0.0)
            s_spat_vx = self.adj[v].get(x, {}).get('spat', 0.0)
            s_spat_w = (node_u.leaf_count * s_spat_ux + node_v.leaf_count * s_spat_vx) / node_w.leaf_count
            
            s_sem_ux = self.adj[u].get(x, {}).get('sem', 0.0)
            s_sem_vx = self.adj[v].get(x, {}).get('sem', 0.0)
            s_sem_w = (node_u.leaf_count * s_sem_ux + node_v.leaf_count * s_sem_vx) / node_w.leaf_count
            
            # add new edge between w and neighbor x
            self._add_edge(w, x, s_spat_w, s_sem_w)
            
            # clean up old edges pointing to u and v from x
            if u in self.adj[x]: del self.adj[x][u]
            if v in self.adj[x]: del self.adj[x][v]
            
        # Deactivate u and v
        node_u.active = False
        node_v.active = False
        del self.adj[u]
        del self.adj[v]
        
    def _process_stage(self, heap_type: str, threshold: float):
        """Processes either the spatial or semantic heap until threshold is reached"""
        target_heap = self.spat_heap if heap_type == 'spat' else self.sem_heap
        
        while target_heap:
            neg_weight, u, v = target_heap[0] 
            weight = -neg_weight
            
            if weight < threshold:
                break
            
            heapq.heappop(target_heap)
            
            # Lazy deletion check: Ensure both nodes haven't been merged already
            if not self.nodes[u].active or not self.nodes[v].active:
                continue
            
            # Double check the edge actually still exists and wasn't overwritten
            if v not in self.adj[u] or self.adj[u][v][heap_type] != weight:
                continue
            
            # Valid top edge found, excute merge
            self.group_and_rewire(u, v)
            
    def group_by_similarity(self):
        """
        Lines 11-21: Two-stage similarity grouping
        """
        # Stage 1: Spatial similarity based grouping
        self._process_stage('spat', self.tau_spat)
        
        # Stage 2: Semantic similarity based grouping
        self._process_stage('sem', self.tau_sem)
        
        # return active components
        active_nodes = [n_id for n_id, n in self.nodes.items() if n.active]
        return active_nodes
    
    def export_to_json(self, output_path: str, per_frame_trees: dict = None):
        """
        Exports the final clustered graph to the Clutt3R-Seg instance_tree.json format.
        
        Args:
            output_path: Path to save the JSON file.
            per_frame_trees: Optional dict containing the input hierarchy 
                             (parent_of and descendant_leaves) to append to the end.
        """
        leaf2inst = []
        initial_idx = []
        
        # filter out deleted nodes, keeping only the final clustered super-nodes
        active_nodes = [node for node in self.nodes.values() if node.active]
        
        # iterate through surviving nodes and assign them a clean, global Instance ID
        for final_instance_id, node in enumerate(active_nodes):
            # record the valid global instance IDs
            initial_idx.append(final_instance_id)
            
            # map every base leaf inside this super-node to this final ID
            for leaf in node.leaves:
                leaf2inst.append({
                    "frame": leaf["frame"],
                    "mask": leaf["mask"],
                    "instance": final_instance_id
                })
                
        output_schema = {
            "schema_version": 1,
            "source": "Custom clustering implementation for instance-tree.",
            "initial": {
                "initial_idx": initial_idx,
                "leaf2inst": leaf2inst
            }
        }
        
        if per_frame_trees:
            if "parent_of" in per_frame_trees:
                output_schema["initial"]["parent_of"] = per_frame_trees["parent_of"]
            if "descedant_leaves" in per_frame_trees:
                output_schema["initial"]["descendant_leaves"] = per_frame_trees["descendant_leaves"]
                
        with open(output_path, 'w') as f:
            json.dump(output_schema, f, indent=2) 
            
        print(f"Successfully exported {len(initial_idx)} unique instances to {output_path}")