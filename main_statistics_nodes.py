# main_statistics_nodes.py

import networkx as nx
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from flow_network import FlowNetwork
from scipy.stats import spearmanr, ConstantInputWarning
import warnings

if __name__ == "__main__":
    # Number of iterations for statistical testing
    num_iterations = 2

    # Centrality measures to consider
    centrality_measures = {
        'betweenness': 'Betweenness',
        'pagerank': 'PageRank',
        'closeness': 'Harmonic Closeness',
        'in_degree': 'In-Degree',
        'out_degree': 'Out-Degree',
        'flow_betweenness': 'Flow Betweenness',
        'in_degree_betweenness_ratio': 'In-Degree/Betweenness'
    }

    # Dictionary to store Spearman correlations for each centrality measure over all iterations
    correlations_over_iterations = {measure: [] for measure in centrality_measures}

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")

        # Create an instance of FlowNetwork
        fn = FlowNetwork()

        # IT REALLY DEPENDS ON GRAPH STRUCTURE
        # Generate a small-world graph
        fn.generate_small_world_graph(
            num_nodes=100,
            k=3,
            p=0.05,
            capacity_range=(5, 15),
            node_failure_percentage=0.2,
            edge_failure_percentage=0
        )

        # Add Super Source and Super Sink
        fn.add_super_source_sink()

        # Compute the initial maximum flow
        initial_flow = fn.compute_actual_flow()

        # Compute centrality measures
        centralities = fn.compute_centrality_measures_nodes()

        # List to store flow increases
        flow_increases = {}
        nodes = [n for n in fn.graph.nodes() if n not in [fn.super_source, fn.super_sink]]

        # For each node, repair it if failed and compute the flow increase
        for node in nodes:
            if fn.graph.nodes[node].get('node_failure', False):
                # Create a copy of the network
                fn_temp = FlowNetwork()
                fn_temp.graph = copy.deepcopy(fn.graph)
                fn_temp.add_super_source_sink()

                # Repair the node
                fn_temp.repair_node(node)

                # Compute the new flow
                new_flow = fn_temp.compute_actual_flow()
                flow_increase = new_flow - initial_flow
                # Only consider nodes with positive flow increase
                if flow_increase > 1e-6:
                    flow_increases[node] = flow_increase

        # Skip iteration if no nodes have positive flow increase
        if not flow_increases:
            print("No nodes with positive flow increase in this iteration. Skipping...")
            continue

        # Prepare data for correlation
        repaired_nodes = list(flow_increases.keys())
        flow_increase_values = [flow_increases[node] for node in repaired_nodes]
        print(flow_increase_values)

        # Extract centrality values for repaired nodes
        centrality_values = {
            measure: [centralities[measure][node] for node in repaired_nodes]
            for measure in centrality_measures
        }

        # Function to compute Spearman correlation with handling of constant arrays
        def compute_spearman_correlation(x, y, name):
            if len(set(y)) <= 1:
                # Cannot compute Spearman correlation with constant input
                return np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=ConstantInputWarning)
                    corr, _ = spearmanr(x, y)
                return corr

        # Compute Spearman correlations for each centrality measure
        for measure, label in centrality_measures.items():
            corr = compute_spearman_correlation(flow_increase_values, centrality_values[measure], label)
            correlations_over_iterations[measure].append(corr)

    # Aggregate results by computing the average Spearman correlation for each centrality measure
    average_correlations = {}
    for measure in centrality_measures:
        # Remove NaN values before computing the average
        valid_correlations = [corr for corr in correlations_over_iterations[measure] if not np.isnan(corr)]
        if valid_correlations:
            average_correlation = np.mean(valid_correlations)
        else:
            average_correlation = np.nan
        average_correlations[measure] = average_correlation

    print("\nAverage Spearman Correlations over all iterations:")
    # Build a list of tuples: (internal_measure_key, display_name, average_correlation)
    sorted_average_correlations = sorted(
        ((measure, centrality_measures[measure], average_correlations[measure])
         for measure in centrality_measures),
        key=lambda x: abs(x[2]) if not np.isnan(x[2]) else -1,
        reverse=True
    )

    for measure_key, display_name, avg_corr in sorted_average_correlations:
        if np.isnan(avg_corr):
            print(f"{display_name}: Average Spearman correlation is undefined (insufficient data)")
        else:
            print(f"{display_name}: Average Spearman correlation = {avg_corr:.4f}")

    # Extract measures (display names) and average correlation values
    measures = [display_name for measure_key, display_name, avg_corr in sorted_average_correlations if
                not np.isnan(avg_corr)]
    avg_corr_values = [avg_corr for measure_key, display_name, avg_corr in sorted_average_correlations if
                       not np.isnan(avg_corr)]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(measures, avg_corr_values, color='skyblue')
    plt.xlabel('Centrality Measures')
    plt.ylabel('Average Spearman Correlation')
    plt.title('Average Spearman Correlation between Centrality Measures and Flow Increase')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('average_centrality_correlations.png')
    print("\nAverage correlation plot saved to 'average_centrality_correlations.png'")
    plt.close()
