# flow_network.py
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import numpy as np

class FlowNetwork:
    def __init__(self):
        self.graph = None  # The NetworkX graph
        self.super_source = 'SuperSource'
        self.super_sink = 'SuperSink'

    def assign_recovery_values(self, node_recovery_dict=None, edge_recovery_dict=None):
        """
        Assigns specific recovery values to nodes and edges, and stores initial recovery values.
        """
        if self.graph is None:
            print("No graph to assign recovery values.")
            return

        G = self.graph

        # Initialize recovery attributes if not present, and store initial values
        for (u, v) in G.edges():
            if 'edge_recovery' not in G.edges[u, v]:
                G.edges[u, v]['edge_recovery'] = 1.0
            G.edges[u, v]['edge_failure'] = G.edges[u, v]['edge_recovery'] < 1.0
            # Store initial recovery and failure status
            G.edges[u, v]['initial_edge_recovery'] = G.edges[u, v]['edge_recovery']
            G.edges[u, v]['initial_edge_failure'] = G.edges[u, v]['edge_failure']

        for node in G.nodes():
            if 'node_recovery' not in G.nodes[node]:
                G.nodes[node]['node_recovery'] = 1.0
            G.nodes[node]['node_failure'] = G.nodes[node]['node_recovery'] < 1.0
            # Store initial recovery and failure status
            G.nodes[node]['initial_node_recovery'] = G.nodes[node]['node_recovery']
            G.nodes[node]['initial_node_failure'] = G.nodes[node]['node_failure']

        # Assign node recovery values
        if node_recovery_dict:
            for node, recovery in node_recovery_dict.items():
                recovery = max(0.0, min(1.0, recovery))  # Ensure recovery is between 0 and 1
                G.nodes[node]['node_recovery'] = recovery
                G.nodes[node]['node_failure'] = recovery < 1.0
                # Store initial recovery and failure status
                G.nodes[node]['initial_node_recovery'] = recovery
                G.nodes[node]['initial_node_failure'] = recovery < 1.0

        # Assign edge recovery values
        if edge_recovery_dict:
            for (u, v), recovery in edge_recovery_dict.items():
                if (u, v) in G.edges:
                    recovery = max(0.0, min(1.0, recovery))  # Ensure recovery is between 0 and 1
                    G.edges[u, v]['edge_recovery'] = recovery
                    G.edges[u, v]['edge_failure'] = recovery < 1.0
                    # Store initial recovery and failure status
                    G.edges[u, v]['initial_edge_recovery'] = recovery
                    G.edges[u, v]['initial_edge_failure'] = recovery < 1.0

    def add_super_source_sink(self):
        """
        Adds a Super Source and Super Sink to the graph.
        """
        if self.graph is None:
            print("No graph to add super source/sink.")
            return

        G = self.graph

        # If SuperSource or SuperSink already in graph, remove them to avoid duplicates
        if self.super_source in G:
            G.remove_node(self.super_source)
        if self.super_sink in G:
            G.remove_node(self.super_sink)

        # Identify source nodes (nodes with no incoming edges)
        sources = [n for n in G.nodes() if G.in_degree(n) == 0]

        # Identify sink nodes (nodes with no outgoing edges)
        sinks = [n for n in G.nodes() if G.out_degree(n) == 0]

        # Add Super Source and connect it to all source nodes
        G.add_node(self.super_source)
        for s in sources:
            if s != self.super_sink:  # Ensure we do not create a loop if both same
                G.add_edge(self.super_source, s, capacity=float('inf'), edge_recovery=1.0)

        # Add Super Sink and connect all sink nodes to it
        G.add_node(self.super_sink)
        for t in sinks:
            if t != self.super_source:  # Ensure we do not create a loop if both same
                G.add_edge(t, self.super_sink, capacity=float('inf'), edge_recovery=1.0)

    def compute_actual_flow(self):
        """
        Computes the maximum flow from Super Source to Super Sink, considering
        capacities modified by recovery factors.
        """
        if self.graph is None:
            print("No graph to compute actual flow.")
            return 0.0

        G = self.graph

        # Create a list of operational nodes (node_recovery > 0)
        operational_nodes = [n for n in G.nodes() if G.nodes[n].get('node_recovery', 1.0) > 0.0]

        if self.super_source not in operational_nodes or self.super_sink not in operational_nodes:
            # If super source or super sink are not operational, flow is 0
            return 0.0

        # Create a copy of the graph for flow calculations
        flow_graph = nx.DiGraph()
        flow_graph.add_nodes_from(operational_nodes)
        for u, v, data in G.edges(data=True):
            if u not in operational_nodes or v not in operational_nodes:
                continue  # Skip edges connected to failed nodes or non-operational nodes

            capacity = data.get('capacity', 1.0)
            edge_recovery = data.get('edge_recovery', 1.0)
            node_recovery_source = G.nodes[u].get('node_recovery', 1.0)
            node_recovery_target = G.nodes[v].get('node_recovery', 1.0)

            # Modify the capacity by the recovery factors
            actual_capacity = capacity * edge_recovery * node_recovery_source * node_recovery_target
            if actual_capacity > 0:
                flow_graph.add_edge(u, v, capacity=actual_capacity)

        # Check if there is a path from Super Source to Super Sink
        try:
            if not nx.has_path(flow_graph, self.super_source, self.super_sink):
                return 0.0
        except nx.NodeNotFound:
            return 0.0

        # Compute maximum flow using the Edmonds-Karp algorithm
        try:
            flow_value, flow_dict = nx.maximum_flow(
                flow_graph, self.super_source, self.super_sink, flow_func=nx.algorithms.flow.edmonds_karp)
        except (nx.NetworkXUnbounded, nx.NetworkXError, ValueError, IndexError) as e:
            # Handle errors due to disconnections or other issues
            print(f"Error computing flow: {e}")
            flow_value = 0.0
        return flow_value

    def repair_node(self, node):
        """
        Repairs a node by setting its recovery to 1.
        """
        if self.graph is None:
            print("No graph to repair node.")
            return
        G = self.graph
        if node in G.nodes() and G.nodes[node].get('node_failure', False):
            G.nodes[node]['node_recovery'] = 1.0
            G.nodes[node]['node_failure'] = False

    def generate_small_world_graph(self, num_nodes=30, k=3, p=0.05, capacity_range=(5, 15),
                                   node_failure_percentage=0.2, edge_failure_percentage=0.2):
        """
        Generates a small-world network with capacities and recovery values.

        Parameters:
        - num_nodes: Number of nodes in the graph.
        - k: Each node is connected to k nearest neighbors in ring topology.
        - p: The probability of rewiring each edge.
        - capacity_range: Tuple specifying the range of capacities for edges.
        - node_failure_percentage: The percentage of nodes to be failed (recovery < 1.0).
        - edge_failure_percentage: The percentage of edges to be failed (recovery < 1.0).
        """
        import random

        # Generate a small-world network using Watts-Strogatz model
        G_undirected = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p, seed=None)

        # Convert to a directed graph
        G_directed = nx.DiGraph()
        G_directed.add_nodes_from(G_undirected.nodes())

        # Randomly assign directions to edges
        for u, v in G_undirected.edges():
            if random.choice([True, False]):
                G_directed.add_edge(u, v)
            else:
                G_directed.add_edge(v, u)

        # Assign capacities to edges
        for u, v in list(G_directed.edges()):
            capacity = random.uniform(*capacity_range)
            G_directed.edges[u, v]['capacity'] = capacity
            # Initialize edge recovery to 1.0 (fully recovered)
            G_directed.edges[u, v]['edge_recovery'] = 1.0
            G_directed.edges[u, v]['edge_failure'] = False
            G_directed.edges[u, v]['initial_edge_recovery'] = 1.0
            G_directed.edges[u, v]['initial_edge_failure'] = False

        # Assign node recovery values
        for node in G_directed.nodes():
            # Initialize node recovery to 1.0 (fully recovered)
            G_directed.nodes[node]['node_recovery'] = 1.0
            G_directed.nodes[node]['node_failure'] = False
            G_directed.nodes[node]['initial_node_recovery'] = 1.0
            G_directed.nodes[node]['initial_node_failure'] = False

        # Ensure super_source and super_sink are not failed
        # We'll add them later with add_super_source_sink

        # Fail a percentage of nodes (excluding super_source and super_sink if they are present)
        valid_nodes_for_failure = [n for n in G_directed.nodes() if n not in [self.super_source, self.super_sink]]
        num_nodes_to_fail = int(node_failure_percentage * len(valid_nodes_for_failure))
        if num_nodes_to_fail > 0 and num_nodes_to_fail <= len(valid_nodes_for_failure):
            nodes_to_fail = random.sample(valid_nodes_for_failure, num_nodes_to_fail)
            for node in nodes_to_fail:
                # Assign a recovery value less than 1.0 (e.g., 0.0)
                G_directed.nodes[node]['node_recovery'] = 0.0
                G_directed.nodes[node]['node_failure'] = True
                G_directed.nodes[node]['initial_node_recovery'] = 0.0
                G_directed.nodes[node]['initial_node_failure'] = True

        # Fail a percentage of edges (excluding edges connecting super_source or super_sink if present)
        edges = list(G_directed.edges())
        edges_for_failure = [(u, v) for (u, v) in edges if
                             u not in [self.super_source, self.super_sink] and v not in [self.super_source,
                                                                                         self.super_sink]]
        num_edges_to_fail = int(edge_failure_percentage * len(edges_for_failure))
        if num_edges_to_fail > 0 and num_edges_to_fail <= len(edges_for_failure):
            edges_to_fail = random.sample(edges_for_failure, num_edges_to_fail)
            for u, v in edges_to_fail:
                # Assign a recovery value less than 1.0 (e.g., 0.0)
                if u in G_directed.nodes() and v in G_directed[u]:
                    G_directed.edges[u, v]['edge_recovery'] = 0.0
                    G_directed.edges[u, v]['edge_failure'] = True
                    G_directed.edges[u, v]['initial_edge_recovery'] = 0.0
                    G_directed.edges[u, v]['initial_edge_failure'] = True

        self.graph = G_directed
        self.add_super_source_sink()

        # If baseline flow is zero, it might be due to disconnection after failures. Let's confirm if there's any path.
        baseline_flow = self.compute_actual_flow()
        if baseline_flow == 0.0:
            print(
                "Warning: Baseline flow is zero after generating and failing nodes/edges. The network may be disconnected or super_source/super_sink are isolated.")
            # (Optional) Attempt to fix or adjust node/edge recovery to ensure a path from super_source to super_sink.
            # This step depends on whether we want to guarantee a path or not.

    def repair_edge(self, u, v):
        """
        Repairs an edge by setting its recovery to 1.
        """
        G = self.graph
        if G.edges[u, v].get('edge_failure', False):
            G.edges[u, v]['edge_recovery'] = 1.0
            G.edges[u, v]['edge_failure'] = False

    def draw_graph(self, filename='flow_network.png', graph=None, directory='images'):
        """
        Draws the graph and saves it to a file in the specified directory.

        Parameters:
        - filename: Name of the image file.
        - graph: The graph to be drawn. If None, uses self.graph.
        - directory: The directory where the image file will be saved.
        """
        if graph is None:
            graph = self.graph

        G = graph

        # Compute positions for the nodes
        pos = nx.spring_layout(G, seed=42)

        # Define node colors
        node_colors = []
        for node in G.nodes():
            if node == self.super_source or node == self.super_sink:
                node_colors.append('white')
            else:
                initial_failure = G.nodes[node].get('initial_node_failure', False)
                current_failure = G.nodes[node].get('node_failure', False)

                if initial_failure and current_failure:
                    # Node is still failed
                    recovery = G.nodes[node].get('node_recovery', 1.0)
                    cmap = cm.get_cmap('autumn')
                    color = cmap(1.0 - recovery)  # Map recovery to color
                    node_colors.append(color)
                elif initial_failure and not current_failure:
                    # Node was failed but is now fixed
                    node_colors.append('lightblue')
                elif not initial_failure:
                    # Node was unaffected
                    node_colors.append('lightgreen')
                else:
                    # Should not reach here
                    node_colors.append('gray')

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)

        # Define edge colors
        edge_colors = []
        for u, v in G.edges():
            if u == self.super_source or v == self.super_sink:
                edge_colors.append('black')
                continue

            initial_failure = G.edges[u, v].get('initial_edge_failure', False)
            current_failure = G.edges[u, v].get('edge_failure', False)

            if initial_failure and current_failure:
                # Edge is still failed
                recovery = G.edges[u, v].get('edge_recovery', 1.0)
                cmap = cm.get_cmap('autumn')
                color = cmap(1.0 - recovery)  # Map recovery to color
                edge_colors.append(color)
            elif initial_failure and not current_failure:
                # Edge was failed but is now fixed
                edge_colors.append('lightblue')
            elif not initial_failure:
                # Edge was unaffected
                edge_colors.append('lightgreen')
            else:
                # Should not reach here
                edge_colors.append('gray')

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowstyle='->', arrowsize=10)

        # Draw labels for Super Source and Super Sink
        labels = {node: node if node in [self.super_source, self.super_sink] else '' for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        # Prepare edge labels with actual capacities
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            capacity = data.get('capacity', 1.0)
            edge_recovery = data.get('edge_recovery', 1.0)
            node_recovery_source = G.nodes[u].get('node_recovery', 1.0)
            node_recovery_target = G.nodes[v].get('node_recovery', 1.0)
            actual_capacity = capacity * edge_recovery * node_recovery_source * node_recovery_target
            edge_labels[(u, v)] = f"{actual_capacity:.1f}"

        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

        plt.title("Flow Network with Actual Capacities")
        plt.axis('off')

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Combine directory and filename to create the full path
        filepath = os.path.join(directory, filename)

        # Save the figure to a file
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Graph saved to {filepath}")
        plt.close()

    def compute_centrality_measures_nodes(self):
        """
        Computes various centrality measures, including the In-Degree/Betweenness Centrality Ratio.
        """
        G = self.graph.copy()

        # Calculate the mean capacity to use as the default weight
        capacities = [d.get('capacity', None) for _, _, d in G.edges(data=True)]
        mean_capacity = np.mean([c for c in capacities if c is not None]) if capacities else 1.0

        # Use capacities as weights
        for u, v in G.edges():
            G.edges[u, v]['weight'] = G.edges[u, v].get('capacity', mean_capacity)

        # Ensure the graph is directed
        if not G.is_directed():
            G = G.to_directed()

        # Compute centralities
        try:
            # Weighted In-Degree and Out-Degree Centrality
            in_degree_centrality = {n: sum(d.get('weight', mean_capacity) for _, _, d in G.in_edges(n, data=True)) for n
                                    in G.nodes()}
            out_degree_centrality = {n: sum(d.get('weight', mean_capacity) for _, _, d in G.out_edges(n, data=True)) for
                                     n in G.nodes()}

            # Weighted Betweenness Centrality
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)

            # Harmonic Closeness Centrality
            closeness_centrality = nx.harmonic_centrality(G, distance='weight')

            # PageRank Centrality
            pagerank_centrality = nx.pagerank(G, weight='weight', alpha=0.85)

            # Flow Betweenness Centrality
            edge_betweenness_centrality = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
            flow_betweenness_centrality = {}
            for node in G.nodes():
                flow_betweenness_centrality[node] = sum(
                    eb for (u, v), eb in edge_betweenness_centrality.items() if u == node or v == node)

            # In-Degree/Betweenness Centrality Ratio
            in_degree_betweenness_ratio = {}
            for node in G.nodes():
                betweenness = betweenness_centrality.get(node, 0)
                if betweenness != 0:
                    ratio = in_degree_centrality[node] / betweenness
                else:
                    # Handle zero betweenness centrality
                    ratio = float('inf')
                in_degree_betweenness_ratio[node] = ratio

            centralities = {
                'in_degree': in_degree_centrality,
                'out_degree': out_degree_centrality,
                'betweenness': betweenness_centrality,
                'closeness': closeness_centrality,
                'pagerank': pagerank_centrality,
                'flow_betweenness': flow_betweenness_centrality,
                'in_degree_betweenness_ratio': in_degree_betweenness_ratio
            }
        except nx.NetworkXError as e:
            print(f"Error computing centralities: {e}")
            centralities = {}

        return centralities

    def evaluate_resilience(self):
        """
        Evaluates the resilience of the network by calculating the initial maximum flow,
        then computing the increase in flow when repairing each failed node individually.
        """
        import copy

        if self.graph is None:
            print("No graph to evaluate resilience.")
            return

        # Create a temporary copy of the network for each node repair simulation
        fn_temp = FlowNetwork()
        fn_temp.graph = copy.deepcopy(self.graph)
        # Ensure the super source and sink are added in the temporary network
        fn_temp.add_super_source_sink()
        # Set all nodes as fully functional initially
        fn_temp.assign_recovery_values()  # Method that assigns initial recovery values (assumed implemented elsewhere)
        # Compute the initial maximum flow with the current state of the network
        initial_flow = fn_temp.compute_actual_flow()
        print(f"Initial maximum flow: {initial_flow}")
        # List to store flow increases
        flow_increases = {}
        nodes = [n for n in fn_temp.graph.nodes() if n not in [fn_temp.super_source, fn_temp.super_sink]]

        # Iterate over each node to evaluate flow increase when repaired
        for node in nodes:

            # Temporarily fail the current node to evaluate impact
            fn_temp.assign_recovery_values({node: 0.0})
            # Compute the flow with the node in a failed state
            failed_flow = fn_temp.compute_actual_flow()
            flow_reduction = initial_flow - failed_flow

            # Repair the node and compute the increased flow
            fn_temp.repair_node(node)

            # Only consider nodes with positive flow increase when repaired
            if flow_reduction > 1e-6:
                flow_increases[node] = flow_reduction

        # Skip evaluation if no nodes have positive flow increase
        if not flow_increases:
            print("No nodes with positive flow increase. Evaluation complete.")
            return

        # Prepare data for analysis
        repaired_nodes = list(flow_increases.keys())
        flow_increase_values = [flow_increases[node] for node in repaired_nodes]
        print(f"Flow increases for repaired nodes: {flow_increase_values}")

        # Print nodes ordered by flow increase
        print("\nNodes ordered by flow increase after repair:")
        for node, increase in sorted(flow_increases.items(), key=lambda x: x[1], reverse=True):
            print(f"Node {node}: Flow inc   rease = {increase}")


        G = fn_temp.graph.to_undirected()
        # 1. Algebraic Connectivity (Fiedler Value)
        fiedler_value = nx.algebraic_connectivity(G, method='lanczos', tol=1e-6) if nx.is_connected(
            G) else 0.0

        # 2. Distribution Analysis: Betweenness Centrality, Degree, and Flow Increase

        def compute_kl_divergence(dist):
            n = len(dist)
            if n == 0:
                return 0.0
            dist_sum = sum(dist)
            p = [d / dist_sum for d in dist]
            q = [1.0 / n] * n
            kl_div = sum(p[i] * np.log2(p[i] / q[i]) for i in range(n) if p[i] > 0)
            return kl_div

        # Betweenness Centrality Distribution
        betweenness_centralities = nx.betweenness_centrality(G)
        betweenness_values = list(betweenness_centralities.values())
        betweenness_kl = compute_kl_divergence(betweenness_values)

        # Degree Distribution
        degrees = [G.degree(n) for n in G.nodes()]
        degree_kl = compute_kl_divergence(degrees)

        flow_increase_kl = compute_kl_divergence(flow_increase_values) if flow_increase_values else 0.0

        # 3. Network Diameter
        if nx.is_connected(G):
            diameter = nx.diameter(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            diameter = nx.diameter(G.subgraph(largest_cc))

        # 4. Average Path Length
        avg_path_length = nx.average_shortest_path_length(G.subgraph(largest_cc)) if not nx.is_connected(
            G) else nx.average_shortest_path_length(G)

        # 5. Clustering Coefficient
        clustering_coefficient = nx.average_clustering(G)

        # 6. Assortativity Coefficient
        assortativity_coefficient = nx.degree_assortativity_coefficient(G)

        # 7. Edge and Node Redundancy
        edges_count = G.number_of_edges()
        nodes_count = G.number_of_nodes()
        edge_redundancy = (edges_count - (nodes_count - 1)) / edges_count if edges_count > nodes_count - 1 else 0.0

        articulation_points = list(nx.articulation_points(G))
        node_redundancy = (nodes_count - len(articulation_points)) / nodes_count

        # 8. Edge Density
        max_edges = nodes_count * (nodes_count - 1) / 2 if not G.is_directed() else nodes_count * (nodes_count - 1)
        edge_density = edges_count / max_edges if max_edges > 0 else 0.0

        # Print the results
        print("\nResilience Measures:")
        print(f"1. Algebraic Connectivity (Fiedler Value): {fiedler_value:.4f}")
        print(f"2. Distribution Analysis:")
        print(f"   - Betweenness Centrality KL Divergence: {betweenness_kl:.4f}")
        print(f"   - Degree Distribution KL Divergence: {degree_kl:.4f}")
        print(f"   - Flow Increase Distribution KL Divergence: {flow_increase_kl:.4f}")
        print(f"3. Network Diameter: {diameter}")
        print(f"4. Average Path Length: {avg_path_length:.4f}")
        print(f"5. Clustering Coefficient: {clustering_coefficient:.4f}")
        print(f"6. Assortativity Coefficient: {assortativity_coefficient:.4f}")
        print(f"7. Edge Redundancy: {edge_redundancy:.4f}")
        print(f"8. Node Redundancy: {node_redundancy:.4f}")
        print(f"9. Edge Density: {edge_density:.4f}")

        # Optional: Plot distributions for visual reference
        self.plot_distribution('betweenness_centrality_distribution.png', betweenness_values,
                               'Betweenness Centrality')
        self.plot_distribution('degree_distribution.png', degrees, 'Degree')
        if flow_increase_values:
            self.plot_distribution('flow_increase_distribution.png', flow_increase_values, 'Flow Increase')
        else:
            print("No values to plot for Flow Increase distribution.")

    def plot_distribution(self, filename, values, label):
        """
        Plots the distribution of given values and saves it to a file.
        """
        if len(values) == 0:
            print(f"No values to plot for {label} distribution.")
            return

        directory = 'images'
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'{label} Distribution')
        plt.xlabel(label)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(filepath)
        print(f"{label} distribution plot saved to '{filepath}'")
        plt.close()

def generate_water_resource_graph(num_nodes, avg_degree=2, rewiring_prob=0.05, seed=None):
    np.random.seed(seed)

    # Step 1: Create an initial regular ring lattice (to mimic structured WDS layout)
    G = nx.Graph()
    for i in range(num_nodes):
        for j in range(1, avg_degree // 2 + 1):
            G.add_edge(i, (i + j) % num_nodes)
            G.add_edge(i, (i - j) % num_nodes)

    # Step 2: Rewire edges with a low probability to add randomness (small-world effect)
    edges = list(G.edges())
    for u, v in edges:
        if np.random.rand() < rewiring_prob:
            G.remove_edge(u, v)
            new_node = np.random.choice([n for n in range(num_nodes) if not G.has_edge(u, n) and n != u])
            G.add_edge(u, new_node)

    # Step 3: Apply Poisson degree distribution for initial degrees
    poisson_degree_target = np.random.poisson(avg_degree, num_nodes)
    for node in G.nodes():
        current_degree = G.degree[node]
        target_degree = poisson_degree_target[node]

        # Adjust the node's degree to match Poisson target
        if current_degree < target_degree:
            while G.degree[node] < target_degree:
                potential_neighbors = [n for n in range(num_nodes) if not G.has_edge(node, n) and n != node]
                if potential_neighbors:
                    new_neighbor = np.random.choice(potential_neighbors)
                    G.add_edge(node, new_neighbor)
                else:
                    break

    # Step 4: Calculate neighborhood degree for additional insights
    neighborhood_degrees = {}
    for node in G.nodes():
        neighborhood_degrees[node] = sum(G.degree[n] for n in G.neighbors(node))

    # Outputting the graph and neighborhood degree information
    return G, neighborhood_degrees