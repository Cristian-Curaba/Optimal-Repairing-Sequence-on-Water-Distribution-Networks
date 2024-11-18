# test_cases.py

test_cases = {
    'Test Case 1': {
        'description': 'No Failures',
        'graph_data': {
            'nodes': [1, 2, 3, 4],
            'edges': [
                (1, 2, {'capacity': 10}),
                (2, 3, {'capacity': 5}),
                (3, 4, {'capacity': 15}),
                (1, 3, {'capacity': 10}),
                (2, 4, {'capacity': 5})
            ]
        },
        'node_recovery_dict': None,
        'edge_recovery_dict': None,
        'repairs': {
            'nodes': [],
            'edges': []
        }
    },
    'Test Case 2': {
        'description': 'All Nodes Have Failures',
        'graph_data': {
            'nodes': [1, 2, 3, 4],
            'edges': [
                (1, 2, {'capacity': 10}),
                (2, 3, {'capacity': 5}),
                (3, 4, {'capacity': 15}),
                (1, 3, {'capacity': 10}),
                (2, 4, {'capacity': 5})
            ]
        },
        'node_recovery_dict': {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
        'edge_recovery_dict': None,
        'repairs': {
            'nodes': [2, 3],
            'edges': []
        }
    },
    'Test Case 3': {
        'description': 'All Edges Have Failures',
        'graph_data': {
            'nodes': [1, 2, 3, 4],
            'edges': [
                (1, 2, {'capacity': 10}),
                (2, 3, {'capacity': 5}),
                (3, 4, {'capacity': 15}),
                (1, 3, {'capacity': 10}),
                (2, 4, {'capacity': 5})
            ]
        },
        'node_recovery_dict': None,
        'edge_recovery_dict': {
            (1, 2): 0.5,
            (2, 3): 0.5,
            (3, 4): 0.5,
            (1, 3): 0.5,
            (2, 4): 0.5
        },
        'repairs': {
            'nodes': [],
            'edges': [(1, 2), (2, 4)]
        }
    },
    'Test Case 4': {
        'description': 'Combination of Node and Edge Failures',
        'graph_data': {
            'nodes': [1, 2, 3, 4],
            'edges': [
                (1, 2, {'capacity': 10}),
                (2, 3, {'capacity': 5}),
                (3, 4, {'capacity': 15}),
                (1, 3, {'capacity': 10}),
                (2, 4, {'capacity': 5})
            ]
        },
        'node_recovery_dict': {2: 0.5, 4: 0.7},
        'edge_recovery_dict': {(1, 2): 0.6, (3, 4): 0.8},
        'repairs': {
            'nodes': [2],
            'edges': [(1, 2)]
        }
    },

    'Test Case 5': {
        'description': 'Repairing Nodes and Edges',
        'graph_data': {
            'nodes': [1, 2, 3, 4],
            'edges': [
                (1, 2, {'capacity': 10}),
                (2, 3, {'capacity': 5}),
                (3, 4, {'capacity': 15}),
                (1, 3, {'capacity': 10}),
                (2, 4, {'capacity': 5})
            ]
        },
        'node_recovery_dict': {2: 0.5, 3: 0.6},
        'edge_recovery_dict': {(1, 2): 0.7, (2, 3): 0.5},
        'repairs': {
            'nodes': [2],
            'edges': [(1, 2)]
        }
    }
    # Continue adding more test cases and their respective repairs
}

test_cases['Test Case 6'] = {
    'description': 'Large Graph with Mixed Failures',
    'graph_data': {
        'nodes': list(range(1, 31)),  # Nodes from 1 to 30
        'edges': [
            # Adding edges to create a connected graph
            # You can customize the edge connections and capacities
            (1, 2, {'capacity': 10}),
            (2, 3, {'capacity': 8}),
            (3, 4, {'capacity': 5}),
            (4, 5, {'capacity': 15}),
            (5, 6, {'capacity': 10}),
            (6, 7, {'capacity': 12}),
            (7, 8, {'capacity': 7}),
            (8, 9, {'capacity': 9}),
            (9, 10, {'capacity': 11}),
            (10, 11, {'capacity': 6}),
            (11, 12, {'capacity': 13}),
            (12, 13, {'capacity': 8}),
            (13, 14, {'capacity': 10}),
            (14, 15, {'capacity': 9}),
            (15, 16, {'capacity': 7}),
            (16, 17, {'capacity': 12}),
            (17, 18, {'capacity': 6}),
            (18, 19, {'capacity': 5}),
            (19, 20, {'capacity': 14}),
            (20, 21, {'capacity': 9}),
            (21, 22, {'capacity': 8}),
            (22, 23, {'capacity': 11}),
            (23, 24, {'capacity': 7}),
            (24, 25, {'capacity': 10}),
            (25, 26, {'capacity': 12}),
            (26, 27, {'capacity': 6}),
            (27, 28, {'capacity': 8}),
            (28, 29, {'capacity': 9}),
            (29, 30, {'capacity': 5}),
            # Adding some additional connections for complexity
            (5, 15, {'capacity': 5}),
            (10, 20, {'capacity': 5}),
            (15, 25, {'capacity': 5}),
            (20, 30, {'capacity': 5}),
            (1, 10, {'capacity': 7}),
            (2, 20, {'capacity': 6}),
            (3, 30, {'capacity': 8}),
        ]
    },
    'node_recovery_dict': {
        # Assign random recovery values to some nodes to simulate failures
        5: 0.6,
        12: 0.5,
        18: 0.7,
        25: 0.4
    },
    'edge_recovery_dict': {
        # Assign random recovery values to some edges to simulate failures
        (7, 8): 0.5,
        (14, 15): 0.6,
        (21, 22): 0.5,
        (28, 29): 0.7,
        (15, 25): 0.5,
        (3, 30): 0.6
    },
    'repairs': {
        'nodes': [5, 18],
        'edges': [(7, 8), (15, 25)]
    }
}

test_cases['Test Case 7'] = {
    'description': 'Large Graph with Disconnection',
    'graph_data': {
        'nodes': list(range(1, 31)),
        'edges': [
            # Creating two clusters connected by a single edge
            # Cluster 1
            (1, 2, {'capacity': 10}),
            (2, 3, {'capacity': 8}),
            (3, 4, {'capacity': 5}),
            (4, 5, {'capacity': 15}),
            (5, 6, {'capacity': 10}),
            (6, 7, {'capacity': 12}),
            (7, 8, {'capacity': 7}),
            (8, 9, {'capacity': 9}),
            (9, 10, {'capacity': 11}),
            # Cluster 2
            (11, 12, {'capacity': 13}),
            (12, 13, {'capacity': 8}),
            (13, 14, {'capacity': 10}),
            (14, 15, {'capacity': 9}),
            (15, 16, {'capacity': 7}),
            (16, 17, {'capacity': 12}),
            (17, 18, {'capacity': 6}),
            (18, 19, {'capacity': 5}),
            (19, 20, {'capacity': 14}),
            (20, 21, {'capacity': 9}),
            (21, 22, {'capacity': 8}),
            (22, 23, {'capacity': 11}),
            (23, 24, {'capacity': 7}),
            (24, 25, {'capacity': 10}),
            (25, 26, {'capacity': 12}),
            (26, 27, {'capacity': 6}),
            (27, 28, {'capacity': 8}),
            (28, 29, {'capacity': 9}),
            (29, 30, {'capacity': 5}),
            # Connecting edge
            (10, 11, {'capacity': 10})
        ]
    },
    'node_recovery_dict': {},
    'edge_recovery_dict': {
        # Simulate disconnection by failing the connecting edge
        (10, 11): 0.0
    },
    'repairs': {
        'nodes': [],
        'edges': [(10, 11)]
    }
}

test_cases['Test Case 8'] = {
    'description': 'Medium Graph with Infinite Capacities',
    'graph_data': {
        'nodes': [1, 2, 3, 4, 5, 6, 7],
        'edges': [
            (1, 2, {'capacity': float('inf')}),
            (2, 3, {'capacity': 5}),
            (3, 4, {'capacity': 15}),
            (4, 5, {'capacity': 10}),
            (5, 6, {'capacity': 7}),
            (6, 7, {'capacity': float('inf')})
        ]
    },
    'node_recovery_dict': {
        3: 0.5,
        5: 0.5
    },
    'edge_recovery_dict': {},
    'repairs': {
        'nodes': [3],
        'edges': []
    }
}
