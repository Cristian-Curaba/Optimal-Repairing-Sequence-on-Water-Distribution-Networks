# main_test_cases.py

import networkx as nx
import copy
import os

# Files
from flow_network import FlowNetwork
from test_cases import test_cases

if __name__ == "__main__":
    print("Available Test Cases:")
    for i, (label, case_data) in enumerate(test_cases.items(), start=1):
        print(f"{i}. {label} - {case_data['description']}")

    # Optionally, prompt the user to select test cases
    selection = input("\nEnter the numbers of the test cases you want to run (e.g., 1,3,5) or press Enter to run all: ")
    if selection.strip():
        selected_indices = [int(num.strip()) for num in selection.split(',') if num.strip().isdigit()]
    else:
        selected_indices = list(range(1, len(test_cases) + 1))  # Run all test cases

    # Run selected test cases
    for index in selected_indices:
        label = list(test_cases.keys())[index - 1]
        case_data = test_cases[label]
        print(f"\nRunning {label}: {case_data['description']}")

        # Create an instance of FlowNetwork
        fn = FlowNetwork()

        # Create the graph
        G = nx.DiGraph()
        graph_data = case_data.get('graph_data', {})

        # If graph_data is provided
        if graph_data:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
        else:
            # Automatically generate a random graph if no data is provided
            fn.generate_random_graph()
            G = fn.graph

        fn.graph = G

        # Assign recovery values
        node_recovery_dict = case_data.get('node_recovery_dict', None)
        edge_recovery_dict = case_data.get('edge_recovery_dict', None)
        fn.assign_recovery_values(node_recovery_dict=node_recovery_dict, edge_recovery_dict=edge_recovery_dict)

        # Add Super Source and Super Sink
        fn.add_super_source_sink()

        # Make a deep copy of the graph before repairs
        graph_before = copy.deepcopy(fn.graph)

        # Compute the actual flow before repairs
        flow_before = fn.compute_actual_flow()
        print(f"Maximum flow before repairs: {flow_before}")

        # Perform repairs if specified in the test case data
        repairs = case_data.get('repairs', {})
        for repair_node in repairs.get('nodes', []):
            fn.repair_node(repair_node)
            print(f"Repaired node {repair_node}")

        for repair_edge in repairs.get('edges', []):
            fn.repair_edge(*repair_edge)
            print(f"Repaired edge {repair_edge}")

        # Compute the actual flow after repairs
        flow_after = fn.compute_actual_flow()
        print(f"Maximum flow after repairs: {flow_after}")

        # Prepare directory path using case_id (label)
        case_id = label.replace(' ', '_')
        directory = os.path.join('images', case_id)

        # Draw the graphs
        filename_before = f"{case_id}_before.png"
        fn.draw_graph(filename=filename_before, graph=graph_before, directory=directory)

        filename_after = f"{case_id}_after.png"
        fn.draw_graph(filename=filename_after, directory=directory)
