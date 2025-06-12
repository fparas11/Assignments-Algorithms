import networkx as nx
from typing import List

def _get_connected_components(graph: nx.Graph) -> List[set]:
    """
    Finds all connected components in the graph using BFS. Each component is
    returned as a set of nodes. This is our custom implementation.
    """
    components = []
    visited_nodes = set()
    for node in graph.nodes():
        if node not in visited_nodes:
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

def _is_clique(graph: nx.Graph) -> bool:
    """Checks if the graph is a clique. (Helper for termination condition)."""
    n = len(graph)
    if n < 2: return True
    return all(d == n - 1 for _, d in graph.degree()) 