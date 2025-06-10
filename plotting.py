import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os

# Set a consistent and professional style for the plots
sns.set_theme(style="whitegrid")

def _setup_plot(title: str, xlabel: str, ylabel: str, ax) -> None:
    """Helper function to set up plot titles and labels."""
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

def plot_component_evolution(history: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Generates a plot showing the size of each component at each level of recursion.
    This creates a single figure with multiple subplots.
    """
    levels = len(history)
    fig, axes = plt.subplots(levels, 1, figsize=(10, 3 * levels), sharex=True)
    if levels == 1: # If only one level, axes is not a list
        axes = [axes]

    fig.suptitle('Evolution of Component Sizes per Recursion Level', fontsize=20, weight='bold')

    for i, data in enumerate(history):
        level = data['level']
        sizes = sorted(data['component_sizes'], reverse=True)
        
        sns.barplot(x=list(range(len(sizes))), y=sizes, ax=axes[i], palette="viridis")
        _setup_plot(f'Level {level}', 'Component Index', 'Number of Nodes', axes[i])
        axes[i].bar_label(axes[i].containers[0])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, '1_component_evolution.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_avg_size(history: List[Dict[str, Any]], output_dir: str) -> None:
    """Plots the average component size at each recursion level."""
    levels = [h['level'] for h in history]
    avg_sizes = [h['avg_component_size'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=levels, y=avg_sizes, marker='o', ax=ax, color='b')
    _setup_plot('Average Component Size vs. Recursion Level', 'Recursion Level', 'Average Size (Nodes)', ax)
    
    filename = os.path.join(output_dir, '2_average_component_size.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_largest_size(history: List[Dict[str, Any]], output_dir: str) -> None:
    """Plots the size of the largest component at each recursion level."""
    levels = [h['level'] for h in history]
    largest_sizes = [h['largest_component'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=levels, y=largest_sizes, marker='o', ax=ax, color='r')
    _setup_plot('Largest Component Size vs. Recursion Level', 'Recursion Level', 'Size of Largest Component (Nodes)', ax)

    filename = os.path.join(output_dir, '3_largest_component_size.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_time_per_level(history: List[Dict[str, Any]], output_dir: str) -> None:
    """Plots the execution time for each individual recursion level."""
    # Extract cumulative times and corresponding levels.
    levels = [h['level'] for h in history]
    cumulative_times = [h['time'] for h in history]

    # Calculate per-level execution time by finding the difference
    # between consecutive cumulative times.
    level_times = []
    for i in range(1, len(cumulative_times)):
        time_for_level = cumulative_times[i] - cumulative_times[i-1]
        level_times.append(time_for_level)

    # The levels for plotting are 1, 2, 3, etc.
    plot_levels = levels[1:]

    if not plot_levels: # Nothing to plot if only level 0 exists.
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # A bar plot is better for visualizing discrete time intervals per level.
    sns.barplot(x=plot_levels, y=level_times, ax=ax, palette="mako")
    
    _setup_plot(
        'Execution Time per Recursion Level',
        'Recursion Level',
        'Time Taken (seconds)',
        ax
    )
    # Add labels to the bars for clarity
    ax.bar_label(ax.containers[0], fmt='%.4f')
    
    # Use a new filename to avoid confusion with the old cumulative plot.
    filename = os.path.join(output_dir, '4_execution_time_per_level.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def generate_plots(history: List[Dict[str, Any]], dataset_name: str) -> None:
    """Main function to generate all required plots."""
    if not history or len(history) < 2:
        print("Not enough history data to generate plots.")
        return

    output_dir = f"plots_{dataset_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_component_evolution(history, output_dir)
    plot_avg_size(history, output_dir)
    plot_largest_size(history, output_dir)
    plot_time_per_level(history, output_dir) 