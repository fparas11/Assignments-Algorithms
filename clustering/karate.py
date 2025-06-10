import time
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any

MIN_CLUSTER_SIZE = 3

# ==============================================================================
# SECTION 1: CORE ALGORITHMS FROM PSEUDOCODE
# This section contains direct implementations of the algorithms from report.md
# ==============================================================================

def _bfs_for_path(graph: nx.Graph, s: int, t: int) -> Dict[int, int]:
    """
    Corresponds to: "BFS_για_αποστάσεις"
    A specific BFS implementation to find an augmenting path in a residual graph.
    It returns a dictionary of 'parents' to reconstruct the path.
    A path is found if 't' is a key in the returned dictionary.
    """
    parents = {s: None}
    queue = [s]
    
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        
        for v, attrs in graph[u].items():
            if v not in parents and attrs.get('weight', 0) > 0:
                parents[v] = u
                queue.append(v)
                if v == t:
                    return parents
    return parents

def _bfs_for_distances(graph: nx.Graph, s: int) -> Dict[int, int]:
    """
    Corresponds to: "BFS_για_αποστάσεις" used for diameter calculation.
    A standard BFS that returns a dictionary of distances from source 's'
    to all other reachable nodes.
    """
    distances = {s: 0}
    queue = [s]
    
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        
        for v in graph[u]:
            if v not in distances:
                distances[v] = distances[u] + 1
                queue.append(v)
    return distances

def _get_connected_components(graph: nx.Graph) -> List[set]:
    """
    Corresponds to: "Εύρεση_Μεγαλύτερης_Συνιστώσας" logic from the report (Step A).
    Finds all connected components in the graph using BFS. Each component is
    returned as a set of nodes.
    """
    components = []
    visited_nodes = set()
    for node in graph.nodes():
        if node not in visited_nodes:
            # This node is part of a new, undiscovered component.
            # Start a BFS from this node to find all nodes in its component.
            component_nodes = set()
            queue = [node]
            visited_nodes.add(node)
            
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                component_nodes.add(u)
                for v in graph.neighbors(u):
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        queue.append(v)
            components.append(component_nodes)
    return components

def _bfs_reachable_in_residual(graph: nx.DiGraph, s: int) -> set:
    """
    Finds all nodes reachable from s in a residual graph, following
    only edges with positive capacity (weight > 0). This is the correct
    way to find the s-partition for the min-cut.
    """
    reachable_nodes = set()
    queue = [s]
    visited = {s}
    
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        reachable_nodes.add(u)
        
        # graph[u].items() gives neighbors and their edge attributes
        for v, attrs in graph[u].items():
            if v not in visited and attrs.get('weight', 0) > 0:
                visited.add(v)
                queue.append(v)
                
    return reachable_nodes

def _edmonds_karp_min_cut(graph: nx.Graph, s: int, t: int) -> Tuple[set, set]:
    """
    Corresponds to: "Edmonds_Karp"
    Finds the min-cut partition of the graph.
    """
    # Create the residual graph from the original.
    residual_graph = nx.DiGraph()
    for u, v in graph.edges():
        residual_graph.add_edge(u, v, weight=1)
        residual_graph.add_edge(v, u, weight=1)

    # Loop to find augmenting paths.
    while True:
        parents = _bfs_for_path(residual_graph, s, t)
        
        # If no path found, break the loop.
        if t not in parents:
            break
            
        # Find the path flow (the bottleneck capacity).
        path_flow = float('Inf')
        v = t
        while v != s:
            u = parents[v]
            path_flow = min(path_flow, residual_graph[u][v]['weight'])
            v = u
            
        # Update residual capacities along the path.
        v = t
        while v != s:
            u = parents[v]
            residual_graph[u][v]['weight'] -= path_flow
            
            # Ensure reverse edge exists before updating
            if not residual_graph.has_edge(v, u):
                residual_graph.add_edge(v, u, weight=0)
            residual_graph[v][u]['weight'] += path_flow
            v = u

    # Find the partition from the final residual graph.
    # The 's' partition are all nodes reachable from 's' using edges with
    # capacity > 0. We use our own BFS for this to ensure correctness.
    s_partition = _bfs_reachable_in_residual(residual_graph, s)
    t_partition = set(graph.nodes()) - s_partition
    
    return s_partition, t_partition


# ==============================================================================
# SECTION 2: MAIN CLUSTERING LOGIC FROM PSEUDOCODE
# ==============================================================================

def _is_clique(graph: nx.Graph) -> bool:
    """Checks if the graph is a clique. (Helper for termination condition)."""
    n = len(graph)
    if n < 2: return True
    # In a clique of size N, every node has a degree of N-1.
    return all(d == n - 1 for _, d in graph.degree())

def _split_component(graph: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
    """
    Corresponds to: "Διαμέρισε_τον_Γράφο"
    Splits a graph component into two partitions.
    """
    if len(graph) < 2:
        return graph, None

    # --- Phase 1: Find diameter endpoints (s, t) ---
    # This is the explicit implementation of the pseudocode's diameter search.
    s, t, max_len = None, None, -1
    nodes = list(graph.nodes)
    for i in range(len(nodes)):
        source_node = nodes[i]
        # Use our own BFS implementation instead of the networkx helper.
        distances = _bfs_for_distances(graph, source_node)
        for target_node, length in distances.items():
            if length > max_len:
                max_len = length
                s, t = source_node, target_node
    
    if s is None or t is None or s == t:
        return graph, None
        
    # --- Phase 2: Find the minimum cut partition between s and t ---
    partition1_nodes, partition2_nodes = _edmonds_karp_min_cut(graph, s, t)
    
    # If a partition is empty, the split was not meaningful.
    if not partition1_nodes or not partition2_nodes:
        return graph, None

    subgraph1 = graph.subgraph(partition1_nodes).copy()
    subgraph2 = graph.subgraph(partition2_nodes).copy()
    
    return subgraph1, subgraph2

def _recursive_cluster(graph_component: nx.Graph, split_log: list, level: int) -> List[nx.Graph]:
    """
    A helper function that implements the recursive clustering logic.
    It populates a flat log of all split events.
    """
    # --- Termination Conditions ---
    if len(graph_component) < MIN_CLUSTER_SIZE or _is_clique(graph_component):
        return [graph_component]

    # --- Recursive Step ---
    subgraph1, subgraph2 = _split_component(graph_component)

    if subgraph2 is None:
        return [graph_component]

    # Log the split event for later processing
    if split_log is not None:
        split_log.append({
            'level': level,
            'parent': graph_component,
            'children': [subgraph1, subgraph2],
            'timestamp': time.time()
        })

    # Recurse on the two new sub-components
    clusters1 = _recursive_cluster(subgraph1, split_log, level + 1)
    clusters2 = _recursive_cluster(subgraph2, split_log, level + 1)

    return clusters1 + clusters2

def cluster(graph: nx.Graph) -> Tuple[List[nx.Graph], List[Dict[str, Any]], np.ndarray]:
    """
    Corresponds to: "Αναδρομική_Ομαδοποίηση"
    This function uses a recursive approach to repeatedly split graph components.
    It serves as the entry point that initializes the clustering process on the
    largest connected component of the input graph, and then processes the
    results to build a history and a linkage matrix for plotting.

    Returns:
        - final_clusters: A list of graph objects, one for each final cluster.
        - history: A list of dictionaries with statistics for each level.
        - linkage_matrix: A NumPy array suitable for scipy's dendrogram function.
    """
    # --- Step A: Find the largest connected component ---
    components = _get_connected_components(graph)
    if not components:
        return [], [], None

    main_component = graph.subgraph(max(components, key=len)).copy()
    
    start_time = time.time()
    split_log = []
    final_clusters = _recursive_cluster(main_component, split_log, level=1)
    
    # --- Post-processing 1: Build level-by-level history from the split log ---
    history = []
    history.append({
        'level': 0, 'component_sizes': [len(main_component)],
        'largest_component': len(main_component),
        'avg_component_size': float(len(main_component)), 'time': 0
    })

    if not split_log:
        return final_clusters, history, None

    max_level = max(s['level'] for s in split_log)
    clusters_at_level = {0: [main_component]}
    last_timestamp = start_time

    for level in range(1, max_level + 2):
        prev_level_clusters = clusters_at_level.get(level - 1, [])
        if not prev_level_clusters:
            break

        splits_this_level = [s for s in split_log if s['level'] == level]
        parents_split_at_this_level = {s['parent'] for s in splits_this_level}
        
        current_level_clusters = []
        if splits_this_level:
            last_timestamp = max(s['timestamp'] for s in splits_this_level)
            for s in splits_this_level:
                current_level_clusters.extend(s['children'])
        
        for comp in prev_level_clusters:
            if comp not in parents_split_at_this_level:
                current_level_clusters.append(comp)

        if not current_level_clusters:
            break
            
        clusters_at_level[level] = current_level_clusters
        
        sizes = [len(c) for c in current_level_clusters]
        history.append({
            'level': level, 'component_sizes': sizes,
            'largest_component': max(sizes), 'avg_component_size': np.mean(sizes),
            'time': last_timestamp - start_time
        })

    # --- Post-processing 2: Build linkage matrix for dendrogram ---
    linkage_matrix = None
    
    if final_clusters and split_log:
        # This maps a frozenset of nodes (a cluster) to its linkage ID and
        # its count in terms of the number of *final clusters* it contains.
        cluster_to_info = {
            frozenset(c.nodes()): (i, 1) for i, c in enumerate(final_clusters)
        }
        
        next_cluster_id = len(final_clusters)
        linkage_rows = []
        
        # Sort splits by level, descending, to process from leaves up to the root
        sorted_splits = sorted(split_log, key=lambda s: s['level'], reverse=True)

        for split in sorted_splits:
            level = split['level']
            parent_nodes = frozenset(split['parent'].nodes())
            child1_nodes = frozenset(split['children'][0].nodes())
            child2_nodes = frozenset(split['children'][1].nodes())

            # Get the IDs and leaf counts of the two children being "merged"
            id1, count1 = cluster_to_info[child1_nodes]
            id2, count2 = cluster_to_info[child2_nodes]
            
            # Distance is defined based on the split level
            distance = float(max_level - level + 1)
            
            # The new count is the sum of the child counts (in terms of final clusters)
            new_count = count1 + count2
            
            linkage_rows.append([id1, id2, distance, new_count])
            
            # The parent cluster gets a new ID and the combined count
            cluster_to_info[parent_nodes] = (next_cluster_id, new_count)
            next_cluster_id += 1
                
        if linkage_rows:
            linkage_matrix = np.array(linkage_rows)

    return final_clusters, history, linkage_matrix