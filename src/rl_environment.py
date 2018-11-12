"""
This module is for working with reinforcement learning agents
for computing the tree decomposition of the expression graphs
"""
import numpy as np
import networkx as nx
import src.graph_model as gm
import copy
import random

MAX_STATE_SIZE = 15


def sparse_graph_adjacency(G, max_size, node_to_row, weight='weight'):
    """Return the graph adjacency matrix as a SciPy sparse matrix.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the adjacency matrix.

    max_size : int
        Matrix size. May be larger than the number of nodes. Has to
        be compatible with the node_to_idx mapping.

    node_to_row : dict
        The mapping between graph nodes and rows/columns in the
        the adjacency matrix

    Returns
    -------
    M : scipy.sparse
        Zero padded adjacency matrix
    """
    from scipy import sparse

    nodelist = list(G)

    if not set(nodelist).issubset(node_to_row):
        msg = "`nodelist` is not a subset of the `node_to_row` dictionary."
        raise nx.NetworkXError(msg)

    index = {node: node_to_row[node] for node in nodelist}
    coefficients = zip(*((index[u], index[v], d.get(weight, 1))
                         for u, v, d in G.edges(nodelist, data=True)
                         if u in index and v in index))
    try:
        row, col, data = coefficients
    except ValueError:
        # there is no edge in the subgraph
        row, col, data = [], [], []

    # symmetrize matrix
    d = data + data
    r = row + col
    c = col + row
    # selfloop entries get double counted when symmetrizing
    # so we subtract the data on the diagonal
    selfloops = list(nx.selfloop_edges(G, data=True))
    if selfloops:
        diag_index, diag_data = zip(*((index[u], -d.get(weight, 1))
                                      for u, v, d in selfloops
                                      if u in index and v in index))
        d += diag_data
        r += diag_index
        c += diag_index
    M = sparse.coo_matrix((d, (r, c)), shape=(max_size, max_size))
    return M


def print_int_matrix(matrix):
    """
    Prints integer matrix in a readable form
    """
    for row in matrix:
        line = ' '.join(f'{e:d}' if e != 0 else '-' for e in row)
        print(line)


def print_int_tril_matrix(matrix):
    """
    Prints a lower triangular integer matrix in a
    readable form
    """
    from math import sqrt
    size = int(-0.5 + sqrt(0.25+2*len(matrix)))

    idx = 0
    for ii in range(size):
        n_elem = ii + 1
        next_idx = idx + n_elem
        line = ' '.join(f'{e:d}' if e != 0 else '-' for e in
                        matrix[idx:next_idx])
        print(line)
        idx = next_idx


def contraction_cost_flops(graph, node):
    """
    Cost function that uses flops contraction cost
    """
    memory, flops = gm.get_cost_by_node(graph, node)
    return flops


def contraction_cost_memory(graph, node):
    """
    Cost function that uses memory contraction cost
    """
    memory, flops = gm.get_cost_by_node(graph, node)
    return memory


def degree_cost(graph, node):
    """
    Cost function that calculates degree
    """
    return graph.degree(node)


class Environment:
    """
    Creates an environment to train the agents
    """
    def __init__(self, filename,
                 cost_function=contraction_cost_flops,
                 simple_graph=False):
        """
        Creates an environment for the model from file

        Parameters
        ----------
        filename : str
               file to load
        cost_function : function, optional
               function (networkx.Graph, int)->int which
               evaluates the cost of selecting a node.
               Default `contraction_cost_flops`
        simple_graph : bool
               If the graph should be generated as simple graph
               (no selfloops and no parallel edges).
        """
        n_qubits, initial_graph = gm.read_graph(filename)

        if initial_graph.number_of_nodes() > MAX_STATE_SIZE:
            raise ValueError(
                f'Graph is larger than the maximal state size:' +
                f' {MAX_STATE_SIZE}')

        if simple_graph:
            initial_graph = nx.Graph(initial_graph)
            initial_graph.remove_edges_from(
                initial_graph.selfloop_edges(data=False))

        self.initial_graph = initial_graph
        self.cost_function = cost_function

        self.reset()

    def reset(self):
        """
        Resets the state of the environment. The graph is
        randomly permutted and a new adjacency matrix is generated
        """

        n_nodes = self.initial_graph.number_of_nodes()

        graph_indices = np.random.permutation(range(1, n_nodes+1))
        entry_indices = np.random.choice(MAX_STATE_SIZE, n_nodes,
                                         replace=False)
        # Build mappings idx -> node
        node_to_row = dict(zip(graph_indices, entry_indices))
        row_to_node = dict(zip(entry_indices, graph_indices))

        # Build mapping triangular index -> node
        row, col = np.tril_indices(MAX_STATE_SIZE)
        # n*(n+1)//2 number of edges that selfloops are allowed.
        tril_to_row = dict(zip(
            range(MAX_STATE_SIZE*(MAX_STATE_SIZE+1) // 2), row))
        idx_to_node = {tril_idx: row_to_node[tril_to_row[tril_idx]]
                       for tril_idx in
                       range(MAX_STATE_SIZE*(MAX_STATE_SIZE+1) // 2)
                       if tril_to_row[tril_idx] in entry_indices
        }

        # Build adjacency matrix and pack it to lower triangular
        graph = copy.deepcopy(self.initial_graph)
        adj_matrix = np.asarray(
            sparse_graph_adjacency(
                graph, MAX_STATE_SIZE, node_to_row).todense()
            )

        state = adj_matrix[np.tril_indices_from(adj_matrix)]
        # state = adj_matrix

        # Store state and useful mappings
        self.node_to_row = node_to_row
        self.idx_to_node = idx_to_node
        self.tril_indices = (row, col)
        self.graph = graph
        self.state = state

    def step(self, index):
        """
        Takes 1 step in the graph elimination environment

        Parameters
        ----------
        index : int
              index in the state matrix to eliminate.
        """
        node = self.idx_to_node[index]

        # Calculate cost function
        cost = self.cost_function(self.graph, node)

        # Update state
        gm.eliminate_node(self.graph, node)
        complete = self.graph.number_of_nodes() == 0

        adj_matrix = np.asarray(
            sparse_graph_adjacency(self.graph, MAX_STATE_SIZE,
                                   self.node_to_row).todense()
        )
        self.state = adj_matrix[self.tril_indices]
        # self.state = adj_matrix

        return cost, complete


if __name__ == '__main__':
    environment = Environment('inst_2x2_7_0.txt')
    environment.reset()

    costs = []
    steps = []
    complete = False
    while not complete:
        print_int_tril_matrix(environment.state)
        print()

        row, *_ = np.nonzero(environment.state)
        # row, col = np.nonzero(environment.state)
        cost, complete = environment.step(row[0])

        steps.append(environment.idx_to_node[row[0]])
        costs.append(cost)

    print(' Strategy\n Node | Cost:')
    print('-'*24)
    print('\n'.join('{:5} | {:5}'.format(step, cost)
                    for step, cost in zip(steps, costs)))
    print('-'*24)
    print('Total cost: {}'.format(sum(costs)))


def wrap_general_graph_for_qtree(graph):
    """
    Modifies a general networkx graph to be compatible with
    graph functions from qtree. Basically, we just renumerate nodes
    from 1 and set attributes.

    Parameters
    ----------
    graph : networkx.Graph or networkx.Multigraph
            Input graph
    Returns
    -------
    new_graph : type(graph)
            Modified graph
    """
    # relabel nodes starting from 1
    label_dict = dict(zip(
        range(graph.number_of_nodes()),
        range(1, graph.number_of_nodes()+1)
    ))

    # Add unique hash tags to edges
    new_graph = nx.relabel_nodes(graph, label_dict, copy=True)
    for edge in new_graph.edges():
        new_graph.edges[edge].update({'hash_tag': hash(random.random())})
    return new_graph


def generate_random_graph(n_nodes, n_edges):
    """
    Generates a random graph with n_nodes and n_edges. Edges are
    selected randomly from a uniform distribution over n*(n-1)/2
    possible edges

    Parameters
    ----------
    n_nodes : int
          Number of nodes
    n_edges : int
          Number of edges
    Returns
    -------
    graph : networkx.Graph
          Random graph usable by graph_models
    """

    # Create a disconnected graph
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))

    # Add edges
    row, col = np.tril_indices(n_nodes)
    idx_to_pair = dict(zip(
        range(int(n_nodes*(n_nodes+1)//2)),
        zip(row, col)
    ))

    edge_indices = np.random.choice(
        range(int(n_nodes*(n_nodes+1)//2)),
        n_edges,
        replace=False
    )
    graph.add_edges_from(idx_to_pair[idx] for idx in edge_indices)

    return wrap_general_graph_for_qtree(graph)
