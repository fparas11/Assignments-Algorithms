import time
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any
from .utils import _get_connected_components, _is_clique

MIN_CLUSTER_SIZE = 3

def _get_diameter_endpoints(graph: nx.Graph) -> Tuple[int, int]:
    """
    Finds endpoints of a diameter of the graph using a 2-BFS approximation.
    This is much faster than the exact algorithm for large graphs.
    """
    if len(graph) < 2:
        return None, None
    
    start_node = list(graph.nodes())[0]
    distances_from_start = nx.single_source_shortest_path_length(graph, start_node)
    u = max(distances_from_start, key=distances_from_start.get)
    distances_from_u = nx.single_source_shortest_path_length(graph, u)
    v = max(distances_from_u, key=distances_from_u.get)
    return u, v

def _bfs_for_augmenting_path(graph, s, t, parent):
    """
    BFS to find an augmenting path in the residual graph.
    """
    visited = {s}
    queue = [(s, float('inf'))]
    parent[s] = -1
    
    while queue:
        u, path_flow = queue.pop(0)
        for v, attr in graph[u].items():
            if v not in visited and attr['weight'] > 0:
                visited.add(v)
                parent[v] = u
                new_path_flow = min(path_flow, attr['weight'])
                if v == t:
                    return new_path_flow
                queue.append((v, new_path_flow))
    return 0

def _edmonds_karp_min_cut(graph, s, t):
    """
    Custom implementation of the Edmonds-Karp algorithm to find the s-t min cut.
    """
    residual_graph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        residual_graph.add_edge(u, v, weight=data.get('weight', 1))
        residual_graph.add_edge(v, u, weight=data.get('weight', 1))
        
    parent = {}
    max_flow = 0
    while (path_flow := _bfs_for_augmenting_path(residual_graph, s, t, parent)):
        max_flow += path_flow
        v_ = t
        while v_ != s:
            u_ = parent[v_]
            residual_graph[u_][v_]['weight'] -= path_flow
            if not residual_graph.has_edge(v_, u_):
                residual_graph.add_edge(v_, u_, weight=0)
            residual_graph[v_][u_]['weight'] += path_flow
            v_ = u_

    # Find the partitions from the residual graph
    reachable = {s}
    queue = [s]
    while queue:
        u = queue.pop(0)
        for v, attr in residual_graph[u].items():
            if v not in reachable and attr['weight'] > 0:
                reachable.add(v)
                queue.append(v)
    
    partition1 = reachable
    partition2 = set(graph.nodes()) - reachable
    
    return max_flow, (partition1, partition2)

def _split_component(graph: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
    """
    Splits a graph component into two partitions using a min-cut algorithm.
    """
    if len(graph) < 2:
        return graph, None

    print("    - Phase 1: Finding approximate diameter endpoints (s, t)...")
    s, t = _get_diameter_endpoints(graph)
    
    if s is None or t is None or s == t:
        print("    - Could not find a meaningful split.")
        return graph, None
    
    print(f"    - Found s={s}, t={t}.")
    print("    - Phase 2: Calculating minimum cut...")
    cut_value, partitions = _edmonds_karp_min_cut(graph, s, t)
    print(f"    - Minimum cut found with value: {cut_value}")
    
    partition1_nodes, partition2_nodes = partitions
    
    if not partition1_nodes or not partition2_nodes:
        return graph, None

    subgraph1 = graph.subgraph(partition1_nodes).copy()
    subgraph2 = graph.subgraph(partition2_nodes).copy()
    
    return subgraph1, subgraph2

def _recursive_cluster(graph_component: nx.Graph, split_log: list, level: int) -> List[nx.Graph]:
    """
    Implements the recursive clustering logic.
    """
    print(f"\n[Level {level}] Processing component of size {len(graph_component.nodes())}...")
    if len(graph_component) < MIN_CLUSTER_SIZE or _is_clique(graph_component):
        print(f"  - Component is a final cluster (size < {MIN_CLUSTER_SIZE} or is a clique).")
        return [graph_component]

    print("  - Splitting component...")
    subgraph1, subgraph2 = _split_component(graph_component)

    if subgraph2 is None:
        print("  - Split resulted in no change. Treating component as a final cluster.")
        return [graph_component]

    print(f"  - Split complete. New component sizes: {len(subgraph1.nodes())} and {len(subgraph2.nodes())}")

    if split_log is not None:
        split_log.append({
            'level': level, 'parent': graph_component,
            'children': [subgraph1, subgraph2], 'timestamp': time.time()
        })

    clusters1 = _recursive_cluster(subgraph1, split_log, level + 1)
    clusters2 = _recursive_cluster(subgraph2, split_log, level + 1)

    return clusters1 + clusters2

def _process_results(graph, final_components, split_log):
    """
    Processes the raw clustering output to generate a structured history
    and a linkage matrix for plotting. This version ensures that every level
    is accounted for in the history, even if no splits occurred.
    """
    if not split_log:
        history = [{
            'level': 0, 'component_sizes': [len(c.nodes) for c in final_components],
            'largest_component': len(graph.nodes),
            'avg_component_size': np.mean([len(c.nodes) for c in final_components]), 'time': 0,
        }]
        return history, None

    start_time = split_log[0]['timestamp']
    max_level = max(item['level'] for item in split_log)
    
    # --- Linkage Matrix Calculation (remains the same) ---
    component_map = {frozenset(c.nodes()): i for i, c in enumerate(final_components)}
    next_cluster_id = len(final_components)
    cluster_counts = {i: 1 for i in range(len(final_components))}
    linkage = []
    
    for level in range(max_level, -1, -1):
        level_splits = [s for s in split_log if s['level'] == level]
        for split in sorted(level_splits, key=lambda x: x['timestamp'], reverse=True):
            child1, child2 = split['children']
            parent = split['parent']
            c1_nodes, c2_nodes = frozenset(child1.nodes()), frozenset(child2.nodes())
            id1, id2 = component_map.get(c1_nodes), component_map.get(c2_nodes)

            if id1 is None or id2 is None: continue

            new_id = next_cluster_id
            component_map[frozenset(parent.nodes())] = new_id
            
            count1, count2 = cluster_counts.get(id1, 1), cluster_counts.get(id2, 1)
            new_count = count1 + count2
            cluster_counts[new_id] = new_count
            next_cluster_id += 1
            
            distance = level + 1
            linkage.append([id1, id2, distance, new_count])

    linkage_matrix = np.array(linkage, dtype=np.double) if linkage else None

    # --- New, Corrected History Generation ---
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
        
    return history, linkage_matrix

def cluster(graph: nx.Graph) -> Tuple[List[nx.Graph], List[Dict[str, Any]], np.ndarray]:
    """
    Performs community detection on the Karate Club graph using a recursive
    s-t min-cut partitioning strategy.
    """
    if not graph or not graph.nodes():
        return [], [], None

    # Step 1: Find largest component
    print("Finding the largest connected component...")
    components = _get_connected_components(graph)
    if not components:
        return [], [], None
    largest_component_nodes = max(components, key=len)
    main_graph = graph.subgraph(largest_component_nodes).copy()
    print(f"Clustering will be performed on the largest component with {main_graph.number_of_nodes()} nodes and {main_graph.number_of_edges()} edges.")

    # Step 2: Recursively split the component
    split_log = []
    final_components = _recursive_cluster(main_graph, split_log, level=0)

    # Step 3: Process the results
    final_clusters_nodes = [list(c.nodes()) for c in final_components]
    history, linkage_matrix = _process_results(graph, final_components, split_log)

    return final_clusters_nodes, history, linkage_matrix