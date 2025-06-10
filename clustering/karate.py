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

def cluster(graph: nx.Graph) -> Tuple[List[nx.Graph], List[Dict[str, Any]]]:
    """
    Corresponds to: "Αναδρομική_Ομαδοποίηση"
    This function uses an iterative approach, which is equivalent to recursion
    but avoids Python's recursion depth limits. It repeatedly tries to split
    all current components until no more splits are possible.
    """
    # --- Step A: Find the largest connected component ---
    # This replaces the nx.is_connected and nx.connected_components calls
    # with our own implementation as per the report's Step A.
    components = _get_connected_components(graph)
    if not components:
        return [], []  # Handle empty graph case

    largest_cc_nodes = max(components, key=len)
    graph = graph.subgraph(largest_cc_nodes).copy()
        
    clusters = [graph]
    history = []
    start_time = time.time()
    
    # Log initial state (level 0)
    history.append({
        'level': 0, 'component_sizes': [len(g) for g in clusters],
        'largest_component': max([len(g) for g in clusters]),
        'avg_component_size': np.mean([len(g) for g in clusters]), 'time': 0
    })

    level = 0
    while True:
        level += 1
        new_clusters = []
        did_split = False
        
        # Try to split each component from the previous level.
        for component in clusters:
            # --- Termination Conditions ---
            if len(component) < MIN_CLUSTER_SIZE or _is_clique(component):
                new_clusters.append(component)
                continue

            # --- Recursive Step (implemented as a call) ---
            subgraph1, subgraph2 = _split_component(component)

            if subgraph2 is None:
                new_clusters.append(component)
            else:
                new_clusters.extend([subgraph1, subgraph2])
                did_split = True

        clusters = new_clusters
        
        # Log state for the current level.
        history.append({
            'level': level, 'component_sizes': [len(g) for g in clusters],
            'largest_component': max([len(g) for g in clusters]),
            'avg_component_size': np.mean([len(g) for g in clusters]),
            'time': time.time() - start_time
        })
        
        # If no component was split in this iteration, the process is complete.
        if not did_split:
            break
            
    return clusters, history 