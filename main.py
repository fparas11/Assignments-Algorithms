import argparse
import networkx as nx

# Import the clustering modules from our package
from clustering import karate, powergrid, erdos
from plotting import generate_plots

def load_karate_graph() -> nx.Graph:
    """Loads the Karate Club graph from the text file."""
    # The file is a simple edgelist with a header.
    # We will build the graph manually to correctly handle the header.
    G = nx.Graph()
    with open('input_data/karateclub.txt', 'r') as f:
        next(f) # Skip header line
        for line in f:
            # handle potential empty lines at the end of the file
            if not line.strip():
                continue
            u, v = map(int, line.strip().split(','))
            # Add weight=1, which will be used as 'capacity' in the min-cut algorithm
            G.add_edge(u, v, weight=1)
    return G

def log_clustering_history(history: list):
    """Prints the clustering history in a structured format."""
    print("\n" + "="*50)
    print(" " * 10 + "Clustering Execution Summary")
    print("="*50)
    
    if not history:
        print("No history data to display.")
        return

    for entry in history:
        level = entry['level']
        sizes = sorted(entry['component_sizes'], reverse=True)
        num_components = len(sizes)
        largest_comp = entry['largest_component']
        avg_size = entry['avg_component_size']
        exec_time = entry['time']

        print(f"\n----- Level {level} -----")
        print(f"  - Components: {num_components}")
        print(f"  - Largest Component: {largest_comp} nodes")
        print(f"  - Average Size: {avg_size:.2f} nodes")
        print(f"  - Time Elapsed: {exec_time:.4f}s")
        if num_components > 1:
            print(f"  - All Component Sizes: {sizes}")
            
    print("\n" + "="*50)

def main():
    """Main function to run the clustering and plotting."""
    parser = argparse.ArgumentParser(description="Graph Clustering Algorithms")
    parser.add_argument(
        'dataset', 
        choices=['karate', 'powergrid', 'erdos'],
        help='The dataset to run the clustering algorithm on.'
    )
    #args = parser.parse_args()
    dataset = 'karate'

    print(f"Running clustering for the '{dataset}' dataset...")

    graph = None
    cluster_func = None

    if dataset == 'karate':
        graph = load_karate_graph()
        cluster_func = karate.cluster
    elif dataset == 'powergrid':
        # Placeholder: a real implementation would load a powergrid graph
        # graph = load_powergrid_graph() 
        cluster_func = powergrid.cluster
    elif dataset == 'erdos':
        # Placeholder: a real implementation would generate an Erdos-Renyi graph
        # graph = nx.erdos_renyi_graph(n=100, p=0.1)
        cluster_func = erdos.cluster

    if graph is None and dataset != 'powergrid' and dataset != 'erdos':
        print(f"Could not load graph for dataset: {dataset}")
        return

    # Run the clustering algorithm
    results = cluster_func(graph)

    linkage_matrix = None
    if dataset == 'karate':
        final_clusters, history, linkage_matrix = results
    else:
        final_clusters, history = results

    # Log the results to the console
    if history:
        log_clustering_history(history)

    # Generate plots if history is available
    if history:
        print("\nClustering complete. Generating plots...")
        generate_plots(history, linkage_matrix, final_clusters, dataset)
        print(f"\nFound {len(final_clusters)} final clusters.")
        print("Final cluster sizes:", [len(c) for c in final_clusters])
    else:
        print("Clustering function did not return data for plotting.")

if __name__ == '__main__':
    main() 