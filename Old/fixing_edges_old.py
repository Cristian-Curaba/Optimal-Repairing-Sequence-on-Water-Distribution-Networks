import networkx as nx
import numpy as np
import random
from scipy.stats import spearmanr


class FlowNetwork:
    def __init__(self):
        self.graph = None  # The NetworkX graph
        self.super_source = 'SuperSource'
        self.super_sink = 'SuperSink'

    def generate_random_graph(self, n_nodes=100, k=4, p=0.1, seed=None):
        """
        Generates a small-world random graph using the Watts-Strogatz model
        and converts it to a directed graph with random edge directions.
        """
        # Generate an undirected Watts-Strogatz small-world graph
        G = nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)

        # Create a directed graph with random edge directions
        DG = nx.DiGraph()
        for u, v in G.edges():
            if random.random() < 0.5:
                DG.add_edge(u, v)
            else:
                DG.add_edge(v, u)

        # Assign random capacities (flows) to each edge
        for (u, v) in DG.edges():
            DG.edges[u, v]['capacity'] = random.uniform(1, 10)

        self.graph = DG

    def assign_random_recovery(self, percentage_not_fixed=0.1, mean=0.5, variance=0.1):
        """
        Assigns random recovery values to edges.
        """
        num_edges = self.graph.number_of_edges()
        num_not_fixed = int(percentage_not_fixed * num_edges)
        edges = list(self.graph.edges())

        # Randomly select edges to be not fully recovered
        not_fixed_edges = random.sample(edges, num_not_fixed)

        # Assign recovery = 1 to all edges initially
        for (u, v) in self.graph.edges():
            self.graph.edges[u, v]['recovery'] = 1.0

        # Assign recovery values < 1 to the selected edges
        for (u, v) in not_fixed_edges:
            recovery = max(0.0, min(1.0, random.gauss(mean, variance)))
            self.graph.edges[u, v]['recovery'] = recovery

    def add_super_source_sink(self):
        """
        Adds a Super Source node connected to all source nodes (nodes with no incoming edges)
        and a Super Sink node connected to all sink nodes (nodes with no outgoing edges).
        """
        DG = self.graph

        # Identify source nodes (nodes with no incoming edges)
        sources = [n for n in DG.nodes() if DG.in_degree(n) == 0]

        # Identify sink nodes (nodes with no outgoing edges)
        sinks = [n for n in DG.nodes() if DG.out_degree(n) == 0]

        # Add Super Source and connect it to all source nodes
        DG.add_node(self.super_source)
        for s in sources:
            DG.add_edge(self.super_source, s, capacity=float('inf'), recovery=1.0)

        # Add Super Sink and connect all sink nodes to it
        DG.add_node(self.super_sink)
        for t in sinks:
            DG.add_edge(t, self.super_sink, capacity=float('inf'), recovery=1.0)

        self.graph = DG

    def compute_actual_flow(self):
        """
        Computes the maximum flow from Super Source to Super Sink, considering
        capacities modified by recovery factors.
        """
        G = self.graph

        # Create a copy of the graph for flow calculations
        flow_graph = nx.DiGraph()
        flow_graph.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            capacity = data.get('capacity', 1.0)
            recovery = data.get('recovery', 1.0)

            #  Modify the capacity by the recovery factor
            actual_capacity = capacity * recovery
            flow_graph.add_edge(u, v, capacity=actual_capacity)

        # Compute maximum flow
        try:
            flow_value, flow_dict = nx.maximum_flow(
                flow_graph, self.super_source, self.super_sink)
        except nx.NetworkXUnbounded:
            flow_value = float('inf')
        return flow_value

    def evaluate_node_repairs(self, nodes_to_evaluate):
        """
        Evaluates the effect of repairing each node on the maximum flow.
        Returns a dictionary mapping each node to the increase in flow.
        """
        G = self.graph
        original_recoveries = {}

        # Compute original maximum flow
        original_flow = self.compute_actual_flow()

        flow_increases = {}

        for node in nodes_to_evaluate:
            # Store original recovery values for the node's edges
            connected_edges = list(G.in_edges(node)) + list(G.out_edges(node))
            original_recoveries[node] = {}
            for u, v in connected_edges:
                original_recoveries[node][(u, v)] = G.edges[u, v]['recovery']
                # Set recovery to 1 (repair the edge)
                G.edges[u, v]['recovery'] = 1.0

            # Compute new maximum flow
            new_flow = self.compute_actual_flow()
            flow_increase = new_flow - original_flow
            flow_increases[node] = flow_increase

            # Restore original recovery values
            for u, v in connected_edges:
                G.edges[u, v]['recovery'] = original_recoveries[node][(u, v)]

        return flow_increases

    def compute_centrality_measures(self):
        """
        Computes centrality measures weighted by actual capacities.
        Returns a dictionary of centrality measures for all nodes.
        """
        G = self.graph.copy()

        # Remove Super Source and Super Sink for centrality computations
        if self.super_source in G:
            G.remove_node(self.super_source)
        if self.super_sink in G:
            G.remove_node(self.super_sink)

        # Compute actual capacities
        for u, v, data in G.edges(data=True):
            capacity = data.get('capacity', 1.0)
            recovery = data.get('recovery', 1.0)
            actual_capacity = capacity * recovery
            data['actual_capacity'] = actual_capacity

        # Use actual capacities as edge weights
        weight = 'actual_capacity'

        # Centrality measures that use edge weights:
        centrality_measures = {}

        # Betweenness Centrality
        betweenness = nx.betweenness_centrality(G, weight=weight, normalized=True)
        centrality_measures['Betweenness Centrality'] = betweenness

        # Closeness Centrality
        if nx.is_strongly_connected(G):
            closeness = nx.closeness_centrality(G, distance=weight)
        else:
            # Compute for largest strongly connected component
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            closeness = {}
            if len(largest_scc) > 1:
                G_scc = G.subgraph(largest_scc)
                closeness_scc = nx.closeness_centrality(G_scc, distance=weight)
                # Assign closeness values
                closeness = {node: closeness_scc.get(node, 0.0) for node in G.nodes()}
            else:
                # All nodes have zero closeness centrality if not connected
                closeness = {node: 0.0 for node in G.nodes()}
        centrality_measures['Closeness Centrality'] = closeness

        # PageRank
        pagerank = nx.pagerank(G, weight=weight)
        centrality_measures['PageRank'] = pagerank

        # Current-Flow Betweenness Centrality
        undirected_G = G.to_undirected()
        try:
            current_flow_betweenness = nx.current_flow_betweenness_centrality(undirected_G, weight=weight,
                                                                              normalized=True)
        except nx.NetworkXError:
            current_flow_betweenness = {node: 0.0 for node in G.nodes()}
        centrality_measures['Current-Flow Betweenness'] = current_flow_betweenness

        return centrality_measures

    def evaluate_centrality_correlation(self):
        """
        Evaluates the correlation between centrality measures and flow increases
        for nodes connected to broken edges.
        """
        import warnings
        from scipy.stats import spearmanr, ConstantInputWarning

        G = self.graph

        # Identify nodes connected to at least one broken edge (recovery < 1)
        broken_edges = [(u, v) for u, v, data in G.edges(data=True) if data['recovery'] < 1.0]
        nodes_with_broken_edges = set()
        for u, v in broken_edges:
            nodes_with_broken_edges.update([u, v])
        nodes_with_broken_edges = list(nodes_with_broken_edges)

        if not nodes_with_broken_edges:
            print("No nodes with broken edges found.")
            return None

        # Compute centrality measures
        centrality_measures = self.compute_centrality_measures()

        # Filter centrality measures for nodes with broken edges
        centrality_data = {}
        for measure_name, measure_dict in centrality_measures.items():
            centrality_data[measure_name] = {node: measure_dict[node] for node in nodes_with_broken_edges}

        # Evaluate node repairs and get flow increases
        flow_increases = self.evaluate_node_repairs(nodes_with_broken_edges)

        # Remove nodes with zero flow increase
        nodes_to_keep = [node for node in nodes_with_broken_edges if flow_increases[node] != 0]
        if len(nodes_to_keep) < 5:
            print("Not enough nodes with variable flow increases.")
            return None

        flow_increase_values = np.array([flow_increases[node] for node in nodes_to_keep])

        correlations = {}
        for measure_name, measure_dict in centrality_data.items():
            centrality_values = np.array([measure_dict[node] for node in nodes_to_keep])

            # Check for constant arrays
            if np.all(centrality_values == centrality_values[0]):
                print(f"{measure_name}: Centrality values are constant; cannot compute correlation.")
                continue
            if np.all(flow_increase_values == flow_increase_values[0]):
                print(f"Flow increases are constant; cannot compute correlation for {measure_name}.")
                continue

            # Compute Spearman correlation with exception handling
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=ConstantInputWarning)
                try:
                    corr_coef, p_value = spearmanr(centrality_values, flow_increase_values)
                    correlations[measure_name] = corr_coef
                    print(f"{measure_name}: Spearman Correlation = {corr_coef:.4f}, p-value = {p_value:.4f}")
                except ConstantInputWarning:
                    print(f"{measure_name}: Constant input; cannot compute correlation.")
                    continue

        if not correlations:
            print("No valid correlations could be computed.")
            return None

        return correlations


if __name__ == "__main__":
    num_iterations = 150
    total_correlations = {
        'Betweenness Centrality': [],
        'Closeness Centrality': [],
        'PageRank': [],
        'Current-Flow Betweenness': []
    }
    successful_iterations = 0

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1} ---")
        # Create an instance of FlowNetwork
        fn = FlowNetwork()

        # Generate a random graph
        fn.generate_random_graph(n_nodes=50, k=4, p=0.1)

        # Assign random recovery functions with adjusted parameters
        fn.assign_random_recovery(percentage_not_fixed=0.3, mean=0.05, variance=0.01)

        # Add Super Source and Super Sink
        fn.add_super_source_sink()

        # Evaluate centrality measures and get correlations
        correlations = fn.evaluate_centrality_correlation()
        if correlations is None:
            continue  # Skip iteration if no valid correlations

        successful_iterations += 1

        # Accumulate the correlations
        for measure_name, corr in correlations.items():
            total_correlations[measure_name].append(corr)

    if successful_iterations == 0:
        print("No successful iterations with valid correlations.")
    else:
        # Compute average correlation coefficients
        average_correlations = {measure: np.mean(values) for measure, values in total_correlations.items()}

        # Print average correlations
        print(f"\n--- Average Spearman Correlations over {successful_iterations} Iterations ---")
        for measure_name, avg_corr in average_correlations.items():
            print(f"{measure_name}: Average Spearman Correlation = {avg_corr:.4f}")

        # Identify the best centrality measure(s)
        max_avg_corr = max(average_correlations.values())
        best_measures = [measure for measure, corr in average_correlations.items() if corr == max_avg_corr]

        print(f"\nBest Centrality Measure(s) on Average: {', '.join(best_measures)} with average Spearman correlation {max_avg_corr:.4f}")
