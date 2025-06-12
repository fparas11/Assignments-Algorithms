import networkx as nx
from typing import List, Dict, Any

# Import the clustering modules from our package
from clustering import karate, powergrid, erdos
from plotting import generate_plots

def load_karate_graph() -> nx.Graph:
    """Loads the Karate Club graph from its edgelist file."""
    G = nx.Graph()
    with open('input_data/karateclub.txt', 'r') as f:
        next(f)  # Skip header line
        for line in f:
            if not line.strip():
                continue
            u, v = map(int, line.strip().split(','))
            G.add_edge(u, v, weight=1)
    return G

def load_powergrid_graph() -> nx.Graph:
    """Loads the Powergrid graph from its edgelist file."""
    # The file is a simple space-separated edgelist. We load it and then
    # assign a default weight of 1 to every edge for the min-cut algorithm.
    G = nx.read_edgelist(
        'input_data/powergrid.txt',
        create_using=nx.Graph(),
        nodetype=int
    )
    nx.set_edge_attributes(G, 1, name='weight')
    return G

def load_erdos_graph() -> nx.Graph:
    """Loads the Erdos graph from its edgelist file."""
    # The file is a simple tab-separated edgelist. We load it and then
    # assign a default weight of 1 to every edge for the min-cut algorithm.
    G = nx.read_edgelist(
        'input_data/erdos.txt',
        create_using=nx.Graph(),
        nodetype=int,
        delimiter='\t'
    )
    nx.set_edge_attributes(G, 1, name='weight')
    return G

def log_clustering_history(history: List[Dict[str, Any]]) -> None:
    """Logs the clustering history to the console."""
    print("\n----- Clustering History -----")
    
    if not history:
        print("No history was generated.")
        return

    # Create a lookup for cumulative time per level to calculate per-level time
    level_timestamps = {h['level']: h['time'] for h in history}

    for item in history:
        print(f"----- Level {item['level']} -----")
        print(f"  - Components: {len(item['component_sizes'])}")
        print(f"  - Largest Component: {item['largest_component']} nodes")
        print(f"  - Average Size: {item['avg_component_size']:.2f} nodes")
        
        # Calculate the time spent just for this level
        current_level = item['level']
        cumulative_time = item['time']
        prev_level_time = level_timestamps.get(current_level - 1, 0)
        time_for_level = cumulative_time - prev_level_time
        
        print(f"  - Time for this Level: {time_for_level:.4f}s")
        print(f"  - Time Elapsed (Cumulative): {cumulative_time:.4f}s")
        
        # Show all component sizes for this level
        sizes = sorted(item['component_sizes'], reverse=True)
        # To avoid spamming the console, only show all sizes if there are <= 10 components
        if len(sizes) > 10:
            print(f"  - Component Sizes: {sizes[:5]}... (and {len(sizes) - 5} more)")
        else:
            print(f"  - All Component Sizes: {sizes}")

def main():
    """Main function to run the clustering and plotting for all datasets."""
    datasets = ['karate', 'powergrid', 'erdos']

    for dataset in datasets:
        print(f"\n\n{'='*20} Running clustering for the '{dataset}' dataset... {'='*20}")

        graph = None
        cluster_func = None

        if dataset == 'karate':
            graph = load_karate_graph()
            cluster_func = karate.cluster
        elif dataset == 'powergrid':
            graph = load_powergrid_graph()
            cluster_func = powergrid.cluster
        elif dataset == 'erdos':
            graph = load_erdos_graph()
            cluster_func = erdos.cluster

        if graph is None:
            print(f"Could not load graph for dataset: {dataset}")
            continue

        # Run the clustering algorithm
        results = cluster_func(graph)

        # The 'karate' dataset returns a linkage matrix, the others do not.
        if dataset == 'karate':
            final_clusters, history, linkage_matrix = results
        else:
            final_clusters, history = results
            linkage_matrix = None # Ensure linkage_matrix is defined

        # Log the results to the console if a history was generated
        if history:
            log_clustering_history(history)
            print("\nClustering complete. Generating plots...")
            generate_plots(graph, history, linkage_matrix, final_clusters, dataset)
        
        print(f"\nFound {len(final_clusters)} final clusters for '{dataset}'.")
        print("Final cluster sizes:", sorted([len(c) for c in final_clusters], reverse=True))

if __name__ == '__main__':
    main() 