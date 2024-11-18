# main_graph_analysis.py
import networkx as nx
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from flow_network import FlowNetwork
from scipy.stats import spearmanr, ConstantInputWarning
import warnings

if __name__ == "__main__":
    fn = FlowNetwork()
    fn.generate_small_world_graph(
        num_nodes=30,
        k=4,
        p=0.1,
        capacity_range=(5, 15),
        node_failure_percentage=0,
        edge_failure_percentage=0
    )
    fn.evaluate_resilience()
