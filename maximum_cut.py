# Copyright 2019 D-Wave Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------ Import necessary packages ----

# pip install streamlit
# streamlit run maximum_cut.py

from collections import defaultdict

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

import random
import streamlit as st


# ------- Set up our graph -------

def generate_random_network(n):
    """
    Generates a random network of n nodes.
    Each node has up to 10 random connections to other nodes.
    
    :param n: Number of nodes in the network
    :return: A list of tuples representing the edges of the network
    """
    edges = set()

    for i in range(1, n + 1):
        # Determine the number of connections for this node (up to 10)
        num_connections = random.randint(1, 10)

        # Create connections
        connections = random.sample(range(1, n + 1), num_connections)

        # Ensure no self-loop and add the edges to the set
        for connection in connections:
            if i != connection:
                edge = tuple(sorted((i, connection)))
                edges.add(edge)

    return list(edges)

# Streamlit interface
st.title('Maximum Cut Problem on a Random Network')

# Sidebar for user input
n = st.sidebar.number_input('Enter the number of nodes:', min_value=10, max_value=200, value=20, step=1)
run_button = st.sidebar.button('Generate Random Network')

# This block runs when the 'Generate Random Network' button is clicked
if run_button:
    # Generate the random network
    network_edges = generate_random_network(n)

    # Create empty graph
    G = nx.Graph()

    # Add edges to the graph (also adds nodes)
    G.add_edges_from(network_edges)

    # ------- Set up our QUBO dictionary -------

    # Initialize our Q matrix
    Q = defaultdict(int)

    # Update Q matrix for every edge in the graph
    for i, j in G.edges:
        Q[(i,i)]+= -1
        Q[(j,j)]+= -1
        Q[(i,j)]+= 2

    # ------- Run our QUBO on the QPU -------
    # Set up QPU parameters
    chainstrength = 8
    numruns = 10

    # Run the QUBO on the solver from your config file
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(Q,
                                chain_strength=chainstrength,
                                num_reads=numruns,
                                label='Example - Maximum Cut')

    # ------- Print results to user -------
    print('-' * 60)
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
    print('-' * 60)
    for sample, E in response.data(fields=['sample','energy']):
        S0 = [k for k,v in sample.items() if v == 0]
        S1 = [k for k,v in sample.items() if v == 1]
        print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int(-1*E))))

    # ------- Display results to user -------
    # Grab best result
    # Note: "best" result is the result with the lowest energy
    # Note2: the look up table (lut) is a dictionary, where the key is the node index
    #   and the value is the set label. For example, lut[5] = 1, indicates that
    #   node 5 is in set 1 (S1).
    lut = response.first.sample

    # Interpret best result in terms of nodes and edges
    S0 = [node for node in G.nodes if not lut[node]]
    S1 = [node for node in G.nodes if lut[node]]
    cut_edges = [(u, v) for u, v in G.edges if lut[u]!=lut[v]]
    uncut_edges = [(u, v) for u, v in G.edges if lut[u]==lut[v]]

    # Display best result
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='r')
    nx.draw_networkx_nodes(G, pos, nodelist=S1, node_color='c')
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
    nx.draw_networkx_labels(G, pos)

    filename = "maxcut_plot.png"
    plt.savefig(filename, bbox_inches='tight')
    print("\nYour plot is saved to {}".format(filename))

        # Display results
    st.write('-' * 60)
    st.write('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0', 'Set 1', 'Energy', 'Cut Size'))
    st.write('-' * 60)
    for sample, E in response.data(fields=['sample', 'energy']):
        S0 = [k for k, v in sample.items() if v == 0]
        S1 = [k for k, v in sample.items() if v == 1]
        st.write('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0), str(S1), str(E), str(int(-1 * E))))

    # Plotting (Streamlit handles plotting slightly differently)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='r')
    nx.draw_networkx_nodes(G, pos, nodelist=S1, node_color='c')
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
    nx.draw_networkx_labels(G, pos)

    st.pyplot(plt)
