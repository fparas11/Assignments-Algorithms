import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import networkx as nx
import pandas as pd

# Set a consistent and professional style for the plots
sns.set_theme(style="whitegrid")

def _setup_plot(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Helper function to set up plot titles and labels."""
    if title:
        ax.set_title(title, fontsize=16, weight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', labelsize=10)

def plot_dendrogram(linkage_matrix: np.ndarray, final_clusters: List[nx.Graph], output_dir: str) -> None:
    """
    Generates and saves a dendrogram plot from a linkage matrix.
    """
    if linkage_matrix is None or len(linkage_matrix) == 0:
        print("Skipping dendrogram plot: no linkage data available.")
        return
        
    if not final_clusters:
        print("Skipping dendrogram plot: final_clusters not available for labeling.")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    
    # This function will be called for each leaf of the dendrogram.
    # The leaf ID `i` corresponds to the index in our final_clusters list.
    def leaf_label_func(i):
        # Make sure the index is within the bounds of the list
        if i < len(final_clusters):
            cluster_size = len(final_clusters[i])
            return f"C{i}\n(size={cluster_size})"
        return str(i)

    dendrogram(
        linkage_matrix,
        ax=ax,
        orientation='top',
        leaf_label_func=leaf_label_func,
        leaf_font_size=8.,
        show_leaf_counts=False, # Our custom labels are better
    )
    
    _setup_plot(ax, 'Hierarchical Clustering Dendrogram', 'Cluster/Node Index', 'Distance (Proportional to Recursion Depth)')
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    filename = os.path.join(output_dir, 'dendrogram.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_component_sizes(history: List[Dict[str, Any]], output_dir: str, dataset_name: str):
    """
    Plots the number of components, average component size, and largest
    component size over the recursion levels.
    """
    if not history:
        return
        
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # Plot 1: Number of Components
    sns.lineplot(data=df, x='level', y=df['component_sizes'].apply(len), ax=axes[0], marker='o')
    _setup_plot(axes[0], "Number of Components per Recursion Level", None, "Component Count")

    # Plot 2: Average Component Size
    sns.lineplot(data=df, x='level', y='avg_component_size', ax=axes[1], marker='o')
    _setup_plot(axes[1], "Average Component Size per Recursion Level", None, "Average Size (nodes)")

    # Plot 3: Largest Component Size
    sns.lineplot(data=df, x='level', y='largest_component', ax=axes[2], marker='o')
    _setup_plot(axes[2], "Largest Component Size per Recursion Level", "Recursion Level", "Largest Component Size (nodes)")

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    filename = f"{output_dir}/{dataset_name}_component_stats.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def plot_execution_time(history: List[Dict[str, Any]], output_dir: str, dataset_name: str):
    """
    Plots the execution time per recursion level, showing all levels explicitly.
    """
    if not history:
        print(f"Cannot plot execution time for {dataset_name}: history is empty.")
        return

    # The history contains cumulative time. We need to calculate the time for each level.
    level_times_map = {}
    last_cumulative_time = 0
    for item in history:
        level = item['level']
        # Time for this level is the increase from the last cumulative time recorded.
        time_for_level = item['time'] - last_cumulative_time
        level_times_map[level] = time_for_level
        last_cumulative_time = item['time']
    
    if not level_times_map:
        return  # No data to plot

    # We want to show all levels from 0 up to the maximum level that occurred.
    max_level = max(level_times_map.keys())
    plot_levels = list(range(max_level + 1))
    plot_times = [level_times_map.get(l, 0) for l in plot_levels]

    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.barplot(x=plot_levels, y=plot_times, palette="mako", ax=ax)
    title = "Execution Time per Recursion Level"

    _setup_plot(ax, title, "Recursion Level", "Time Taken (seconds)")
    
    # Annotate the bars for clarity
    for p in ax.patches:
        bar_height = p.get_height()
        if bar_height > 1e-4:
            ax.annotate(f"{bar_height:.4f}", 
                        (p.get_x() + p.get_width() / 2., bar_height),
                        ha='center', va='center', fontsize=9, color='black', 
                        xytext=(0, 8), textcoords='offset points')

    # Set the x-axis to have a tick for every single level.
    # This ensures no levels are skipped in the visualization.
    ax.set_xticks(plot_levels)
    ax.tick_params(axis='x', rotation=45, labelsize=8) # Rotate for readability

    fig.tight_layout()
    filename = f"{output_dir}/execution_time.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def plot_community_graph(graph: nx.Graph, partition: List[List[int]], output_dir: str, dataset_name: str):
    """
    Visualizes the graph with nodes colored by their community assignment.
    This is suitable for non-hierarchical or non-binary divisive clustering
    results.
    """
    if not partition or not any(partition) or not graph.nodes():
        print(f"Cannot plot community graph for {dataset_name}: partition or graph is empty.")
        return

    fig, ax = plt.subplots(figsize=(15, 15))
    
    print(f"Calculating graph layout for {dataset_name}... (this may take a moment)")
    iterations = 50 if len(graph) < 1000 else 20
    try:
        pos = nx.spring_layout(graph, iterations=iterations, seed=42)
        print("Layout calculation complete.")
    except Exception as e:
        print(f"Could not generate spring layout, falling back to random. Error: {e}")
        pos = nx.random_layout(graph, seed=42)

    color_map = {node: i for i, comm in enumerate(partition) for node in comm}
    colors = [color_map.get(node, -1) for node in graph.nodes()]

    nx.draw_networkx_edges(graph, pos, alpha=0.5, ax=ax)
    nodes = nx.draw_networkx_nodes(
        graph, pos, node_color=colors, cmap=plt.cm.jet, node_size=50, ax=ax
    )
    if nodes:
        nodes.set_edgecolor('black')

    ax.set_title(f"Community Detection for {dataset_name.capitalize()}", fontsize=20)
    fig.tight_layout()
    filename = f"{output_dir}/community_graph.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def generate_plots(
    graph: nx.Graph,
    history: List[Dict[str, Any]],
    linkage_matrix: np.ndarray,
    final_clusters: List[List[int]],
    dataset_name: str
):
    """
    Generates and saves all relevant plots for a clustering result.
    """
    output_dir = f"output_plots/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving plots for '{dataset_name}' to '{output_dir}/'")

    if history:
        plot_component_sizes(history, output_dir, dataset_name)
        plot_execution_time(history, output_dir, dataset_name)
    
    if linkage_matrix is not None:
        plot_dendrogram(linkage_matrix, final_clusters, output_dir)
    
    # For datasets without a dendrogram, plot the final community graph.
    if linkage_matrix is None:
        plot_community_graph(graph, final_clusters, output_dir, dataset_name) 