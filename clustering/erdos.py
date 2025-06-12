import networkx as nx
from typing import List, Tuple, Dict, Any
import time
import numpy as np
from scipy.sparse.linalg import eigs, eigsh
from sklearn.cluster import KMeans
from .utils import _get_connected_components, _is_clique

MIN_CLUSTER_SIZE = 3

def _spectral_split(graph: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
    """
    Splits a graph component into two partitions using spectral clustering.
    """
    if len(graph) < 2:
        return graph, None

    # Use the normalized Laplacian
    L = nx.normalized_laplacian_matrix(graph)
    
    # Find the eigenvector for the second smallest eigenvalue (Fiedler vector)
    # We ask for the 2 smallest eigenvalues because the smallest is always 0.
    # `which='SM'` asks for the eigenvalues with the smallest magnitude.
    # `eigsh` is for real symmetric matrices, which the Laplacian is.
    try:
        eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
    except Exception:
        # If solver fails, it might be due to disconnected components,
        # which shouldn't happen here but is a safe fallback.
        return graph, None

    # The Fiedler vector is the one corresponding to the second smallest eigenvalue.
    fiedler_vector = eigenvectors[:, 1]
    
    # Partition nodes based on the sign of the Fiedler vector's components.
    partition1_nodes = {node for i, node in enumerate(graph.nodes()) if fiedler_vector[i] >= 0}
    partition2_nodes = {node for i, node in enumerate(graph.nodes()) if fiedler_vector[i] < 0}

    if not partition1_nodes or not partition2_nodes:
        return graph, None

    subgraph1 = graph.subgraph(partition1_nodes).copy()
    subgraph2 = graph.subgraph(partition2_nodes).copy()
    
    return subgraph1, subgraph2

def _recursive_cluster(graph_component: nx.Graph, split_log: list, level: int) -> List[nx.Graph]:
    """
    Implements the recursive spectral clustering logic.
    """
    print(f"\n[Level {level}] Processing component of size {len(graph_component.nodes())}...")
    if len(graph_component) < MIN_CLUSTER_SIZE or _is_clique(graph_component):
        print(f"  - Component is a final cluster (size < {MIN_CLUSTER_SIZE} or is a clique).")
        return [graph_component]

    print("  - Splitting component using spectral method...")
    subgraph1, subgraph2 = _spectral_split(graph_component)

    if subgraph2 is None:
        print("  - Split resulted in no change. Treating component as a final cluster.")
        return [graph_component]

    print(f"  - Split complete. New component sizes: {len(subgraph1.nodes())} and {len(subgraph2.nodes())}")

    if split_log is not None:
        split_log.append({
            'level': level, 'parent': graph_component,
            'children': [subgraph1, subgraph2], 'timestamp': time.perf_counter()
        })

    clusters1 = _recursive_cluster(subgraph1, split_log, level + 1)
    clusters2 = _recursive_cluster(subgraph2, split_log, level + 1)

    return clusters1 + clusters2

def _process_history(graph, final_components, split_log, start_time):
    """
    Processes the raw clustering output to generate a structured history.
    This version ensures that every level is accounted for, even if no splits occurred.
    """
    if not split_log:
        return [{
            'level': 0, 'component_sizes': [len(c.nodes) for c in final_components],
            'largest_component': len(graph.nodes),
            'avg_component_size': np.mean([len(c.nodes) for c in final_components]), 'time': 0,
        }]

    max_level = max(item['level'] for item in split_log)
    history = []
    current_components = [graph]
    
    history.append({
        'level': 0,
        'component_sizes': [len(c.nodes) for c in current_components],
        'largest_component': max(len(c.nodes) for c in current_components) if current_components else 0,
        'avg_component_size': np.mean([len(c.nodes) for c in current_components]) if current_components else 0,
        'time': 0,
    })

    for level in range(max_level + 1):
        splits_at_level = [s for s in split_log if s['level'] == level]
        
        if splits_at_level:
            parents_this_level = {frozenset(s['parent'].nodes()) for s in splits_at_level}
            children_this_level = [c for s in splits_at_level for c in s['children']]
            unsplit_components = [c for c in current_components if frozenset(c.nodes()) not in parents_this_level]
            current_components = unsplit_components + children_this_level

        component_sizes = [len(c.nodes()) for c in current_components]

        splits_up_to_this_level = [s for s in split_log if s['level'] <= level]
        time_val = history[-1]['time']
        if splits_up_to_this_level:
            max_timestamp = max(s['timestamp'] for s in splits_up_to_this_level)
            time_val = max_timestamp - start_time
        
        history.append({
            'level': level + 1,
            'component_sizes': component_sizes,
            'largest_component': max(component_sizes) if component_sizes else 0,
            'avg_component_size': np.mean(component_sizes) if component_sizes else 0,
            'time': time_val,
        })
        
    return history

def cluster(graph: nx.Graph) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """
    Performs community detection on the Erdos graph using recursive spectral clustering.
    """
    start_time = time.perf_counter()
    # --- Step A: Find the largest connected component ---
    print("Finding the largest connected component...")
    components = _get_connected_components(graph)
    if not components:
        return [], []
    largest_component_nodes = max(components, key=len)
    main_graph = graph.subgraph(largest_component_nodes).copy()
    print(f"Clustering will be performed on the largest component with {main_graph.number_of_nodes()} nodes and {main_graph.number_of_edges()} edges.")

    # Step 2: Recursively split the component
    split_log = []
    final_components = _recursive_cluster(main_graph, split_log, level=0)

    # Step 3: Process the results
    final_clusters_nodes = [list(c.nodes()) for c in final_components]
    history = _process_history(main_graph, final_components, split_log, start_time)
    
    return final_clusters_nodes, history 