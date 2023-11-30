

# ------ Import necessary packages ----

# pip install dash dash-cytoscape

# pip install plotly

# streamlit run maximum_cut.py

import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import networkx as nx
from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import random


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

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Maximum Cut Problem Visualization"

# Define CSS styles for dark mode
styles = {
    'container': {
        'padding': '10px',
        'font-family': 'Arial, sans-serif',
        'background-color': '#121212',  # Dark background color
        'color': '#FFFFFF',            # Light text color
        'min-height': '100vh',         # Full height container
        'display': 'flex',
        'flex-direction': 'column',
        'align-items': 'center'
    },
    'title': {
        'textAlign': 'center',
        'color': '#FFFFFF'
    },
    'input': {
        'width': '200px',
        'padding': '10px',
        'margin': '10px',
        'color': '#000000'
    },
    'button': {
        'background-color': '#1E90FF', # Button color
        'color': '#FFFFFF',            # Button text color
        'border': 'none',
        'padding': '10px 20px',
        'text-align': 'center',
        'text-decoration': 'none',
        'display': 'inline-block',
        'font-size': '16px',
        'margin': '4px 2px',
        'cursor': 'pointer',
        'border-radius': '5px'
    },
    'graph-container': {
        'height': '1000px',
        'width': '1200px',
        'margin-top': '5px'
    }
}

# Define the layout of the app
app.layout = html.Div(style=styles['container'], children=[
    html.H1('Maximum Cut Problem on a Random Network', style=styles['title']),
    dcc.Markdown("**The Maximum Cut Problem, in the context of disrupting enemy communication networks as a part of electronic or cyber warfare, involves strategically targeting certain nodes (like servers, communication hubs, or relay stations) to maximally disrupt the network's functionality**", style={'textAlign': 'center', 'color': '#FFFFFF'}),
    dcc.Markdown("Network Modeling, Identifying Key Nodes, Maximizing Impact, Operational Efficiency, Dynamic Adaptation", style={'textAlign': 'center', 'color': '#FFFFFF'}),
    html.Div([
        dcc.Input(id='num-nodes', type='number', min=10, max=200, value=20, step=1, style=styles['input']),
        html.Button('Generate Random Network', id='generate-button', style=styles['button'])
    ], style={'textAlign': 'center'}),
    dcc.Graph(id='network-graph', style=styles['graph-container'])
])

# Callback to update the network graph
@app.callback(
    Output('network-graph', 'figure'),
    [Input('generate-button', 'n_clicks')],
    [dash.dependencies.State('num-nodes', 'value')],
)

def update_graph(n_clicks, num_nodes):
    if n_clicks is None or num_nodes is None or num_nodes < 1:
        # Prevent update before the button is clicked for the first time
        return go.Figure()

    # Generate the random network
    network_edges = generate_random_network(num_nodes)

    # Create empty graph
    G = nx.Graph()
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


 # -- UI -- 

    # Interpret best result in terms of nodes and edges
    lut = response.first.sample
    S0 = [node for node in G.nodes if not lut[node]]
    S1 = [node for node in G.nodes if lut[node]]

    # Generate 3D positions for each node using a 3D layout
    pos = nx.spring_layout(G, dim=3, seed=42)

    # Create Plotly 3D scatter plot for the nodes
    node_x, node_y, node_z, node_color = [], [], [], []

    # Create edge traces with specific colors for each set
    edge_trace_red = go.Scatter3d(
        x=[], y=[], z=[], 
        mode='lines', 
        line=dict(color='darkred', width=2)
    )
    edge_trace_blue = go.Scatter3d(
        x=[], y=[], z=[], 
        mode='lines', 
        line=dict(color='darkblue', width=2)
    )

    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        # Assign edge color based on the set of the first node
        if edge[0] in S0:
            edge_trace_red['x'] += (x0, x1, None)
            edge_trace_red['y'] += (y0, y1, None)
            edge_trace_red['z'] += (z0, z1, None)
        else:
            edge_trace_blue['x'] += (x0, x1, None)
            edge_trace_blue['y'] += (y0, y1, None)
            edge_trace_blue['z'] += (z0, z1, None)

    node_x, node_y, node_z, node_color = [], [], [], []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_color.append('red' if node in S0 else 'blue')

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, 
        mode='markers', 
        marker=dict(size=5, color=node_color)
    )

    fig = go.Figure(data=[edge_trace_red, edge_trace_blue, node_trace])

    # Update layout for dark mode and remove gridlines
    fig.update_layout(
        title='Maximum Cut Chart',
        paper_bgcolor='rgba(0,0,0,0)',  
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, showgrid=False, zeroline=False),  # Hide axis background, ticks, gridlines
            yaxis=dict(showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
            xaxis_title='',
            yaxis_title='',
            zaxis_title=''
        ),
        scene_aspectmode='cube'
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

