import networkx as nx
from typing import List, Tuple, Dict, Any
import time
import numpy as np

from networkx.algorithms import community
from .utils import _get_connected_components

def cluster(graph: nx.Graph) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """
    Performs community detection on the Powergrid graph using the Girvan-Newman
    method. This is a divisive algorithm that fits the recursive min-cut
    paradigm by progressively removing high-centrality edges.
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

    print("Starting Girvan-Newman algorithm... (this will take a few minutes)")
    
    # Girvan-Newman is very computationally expensive. We will limit the number of
    # levels to a reasonable number to get a result in a feasible amount of time.
    MAX_LEVELS = 8 
    
    partitions = []
    history = []
    gn_iterator = community.girvan_newman(main_graph)
    
    for i in range(MAX_LEVELS):
        print(f"  - Calculating Girvan-Newman level {i + 1}/{MAX_LEVELS}...")
        try:
            partition = next(gn_iterator)
            partitions.append(partition)
            
            sizes = [len(c) for c in partition]
            if not sizes: continue
            
            history.append({
                'level': i,
                'component_sizes': sizes,
                'largest_component': max(sizes),
                'avg_component_size': np.mean(sizes),
                'time': time.perf_counter() - start_time
            })
        except StopIteration:
            print("  - No more partitions could be generated.")
            break
            
    print("Girvan-Newman processing complete.")

    if not partitions:
        print("  - Warning: No partitions were generated. Using first available partition.")
        return [list(c) for c in next(community.girvan_newman(graph))], []

    # The final clustering is the last partition we generated.
    final_clusters_nodes = [list(c) for c in partitions[-1]]
    
    return final_clusters_nodes, history 