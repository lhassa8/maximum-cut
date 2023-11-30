

# ------ Import necessary packages ----

# pip install dash dash-cytoscape

# pip install plotly

# streamlit run maximum_cut.py

import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import random
import time


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
    },
        'input-container': {
        'margin': '10px 0',  # Add margin to top and bottom
        'display': 'flex',
        'justify-content': 'space-evenly',  # Evenly space items
        'align-items': 'center'
    },
    'timer': {
        'textAlign': 'center',
        'color': '#FFFFFF',
        'marginTop': '10px'  # Add margin to the top of the timer
    }
}

# Define the layout of the app
app.layout = html.Div(style=styles['container'], children=[
    html.H1('Maximum Cut Problem on a Network', style=styles['title']),
    dcc.Markdown("The Maximum Cut Problem, in this context finds the most effective way to disrupt or weaken the network.", style={'textAlign': 'left', 'color': '#FFFFFF'}),
    dcc.Markdown("Network Modeling, Identifying Key Nodes, Maximizing Impact, Operational Efficiency, Dynamic Adaptation", style={'textAlign': 'left', 'color': '#FFFFFF'}),
   
    html.Div(style=styles['input-container'], children=[
        dcc.Input(id='num-nodes', type='number', min=10, max=200, value=10, step=1, style=styles['input']),  # Default value set to 10
        html.Button('Generate with Edges', id='generate-with-edges-button', style=styles['button']),
        html.Button('Generate without Edges', id='generate-without-edges-button', style=styles['button']),
    ]),
    
    html.Span(id='timer-output', style=styles['timer']),
    dcc.Graph(id='network-graph', style=styles['graph-container'])
])

# Callback to update the network graph
@app.callback(
    Output('network-graph', 'figure'),
    Output('timer-output', 'children'),
    Input('generate-with-edges-button', 'n_clicks'),
    Input('generate-without-edges-button', 'n_clicks'),
    State('num-nodes', 'value')
)

def update_graph(with_edges_clicks, without_edges_clicks, num_nodes):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if button_id in ['generate-with-edges-button', 'generate-without-edges-button']:
        start_time = time.time()

        edges_visible = (button_id == 'generate-with-edges-button')
        fig = regenerate_graph(num_nodes, edges_visible)

        elapsed_time = time.time() - start_time
        time_output = f"Graph generated in {elapsed_time:.2f} seconds"
        return fig, time_output

    return go.Figure(), ""


def regenerate_graph(num_nodes, edges_visible):
    if num_nodes is None or num_nodes < 1:
        return go.Figure()

    # Generate the random network
    network_edges = generate_random_network(num_nodes)

    # Create empty graph
    G = nx.Graph()
    G.add_edges_from(network_edges)

    # Initialize Q matrix and other variables for QUBO
    Q = defaultdict(int)
    for i, j in G.edges:
        Q[(i, i)] += -1
        Q[(j, j)] += -1
        Q[(i, j)] += 2

    # Run the QUBO on the solver
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(Q, chain_strength=8, num_reads=10)
    
    # Interpret best result in terms of nodes and edges
    lut = response.first.sample
    S0 = {node for node in G.nodes if not lut[node]}
    S1 = {node for node in G.nodes if lut[node]}

    # Count cross-set edges for each node
    cross_set_edges_count = {}
    for node in G.nodes:
        cross_set_edges_count[node] = sum(1 for neighbor in G[node] if (neighbor in S1 if node in S0 else neighbor in S0))

    # Identify top 3 nodes with most cross-set edges
    top_3_nodes = sorted(cross_set_edges_count, key=cross_set_edges_count.get, reverse=True)[:3]


    # Generate 3D positions for each node using a 3D layout
    pos = nx.spring_layout(G, dim=3, seed=42)

    # Create Plotly 3D scatter plot for the nodes
    node_x, node_y, node_z, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_color.append('red' if node in S0 else 'blue')
        node_size.append(15 if node in top_3_nodes else 7)  # Larger size for top 3 nodes

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, 
        mode='markers', 
        marker=dict(
            size=node_size, 
            color=node_color,
            line=dict(width=0)  # Set line width to 0 to remove the border
        )
    )

    fig = go.Figure(data=[node_trace])

    # Add edges if they are set to be visible
    if edges_visible:
        edge_trace_red = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='darkred', width=2))
        edge_trace_blue = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='darkblue', width=2))

        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            if edge[0] in S0:
                edge_trace_red['x'] += (x0, x1, None)
                edge_trace_red['y'] += (y0, y1, None)
                edge_trace_red['z'] += (z0, z1, None)
            else:
                edge_trace_blue['x'] += (x0, x1, None)
                edge_trace_blue['y'] += (y0, y1, None)
                edge_trace_blue['z'] += (z0, z1, None)

        fig.add_trace(edge_trace_red)
        fig.add_trace(edge_trace_blue)

    # Update layout for dark mode and remove gridlines
    fig.update_layout(title='Maximum Cut Chart', paper_bgcolor='rgba(0,0,0,0)',  
                      plot_bgcolor='rgba(0,0,0,0)',
                      scene=dict(xaxis=dict(showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
                                 yaxis=dict(showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
                                 zaxis=dict(showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
                                 xaxis_title='', yaxis_title='', zaxis_title=''),
                      scene_aspectmode='cube')

    return fig




# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

