import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it


from src.optimizer import Var, Tensor


from numpy.compat import basestring
from numpy.core.numeric import asarray, asanyarray


einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)


def _parse_einsum_input(operands):
    """
    A reproduction of einsum c side einsum parsing in python.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> __parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b])

    >>> __parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b])
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], basestring):
        subscripts = operands[0].replace(" ", "")
        operands = [asanyarray(v) for v in operands[1:]]

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol."
                                 % s)
    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [asanyarray(v) for v in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists"
                                    " must contain "
                                    "either int or Ellipsis")
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists"
                                    " must contain "
                                    "either int or Ellipsis")
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (
            subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(
            ".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) -
                                         set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol."
                                 % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear"
                             " in the input" % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must"
                         " be equal to the number of operands.")

    return (input_subscripts, output_subscript, operands)


def einsum2graph(subscripts, *operands):
    """
    Construct a graph of a tensor contraction from the
    input to Numpy's einsum.

    Parameters:
    -----------
    subsrcipts : str
        set of indices in Einstein notation
        i.e. 'ij,jk,klm,lm->i'

    operands: list of array_like
        tensor

    Returns:
    --------
    graph : networkx.MultiGraph
            Graph which corresponds to the tensors contraction
    free_variables: list of Var
            Indices of the result. These variables will not be contracted
    data_dict : dict
            Dictionary containing Numpy tensors
    """
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4, 4)
    subscripts = '...a,...a->...'
    operands = (a, b)

    # Python side parsing
    input_subscripts, output_subscript, operands = _parse_einsum_input(
        [subscripts, *operands])
    subscripts = input_subscripts + '->' + output_subscript

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    # input_sets = [set(x) for x in input_list]
    # output_set = set(output_subscript)
    # indices = set(input_subscripts.replace(',', ''))

    # Get length of each unique dimension and
    # ensure all dimensions are correct
    dimension_dict = {}
    broadcast_indices = [[] for x in range(len(input_list))]
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s "
                             "does not contain the "
                             "correct number of indices for operand %d."
                             % (input_subscripts[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = sh[cnum]

            # Build out broadcast indices
            if dim == 1:
                broadcast_indices[tnum].append(char)

            if char in dimension_dict.keys():
                # For broadcasting cases we always
                # want the largest dim size
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError("Size of label '%s' for"
                                     " operand %d (%d) "
                                     "does not match previous terms (%d)."
                                     % (char, tnum,
                                        dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim

    # create MultiGraph
    graph = nx.MultiGraph()

    # Indices are represented by integers in the graph
    # Add nodes to the graph
    name_to_idx = {}
    for ii, (idx_name, size) in enumerate(dimension_dict.items()):
        graph.add_node(ii, name=idx_name, size=size)
        name_to_idx[idx_name] = ii

    # Add edges between nodes and build data dictionary
    data_dict = {}
    for tensor_num, str_indices in enumerate(input_list):
        tensor_indices = tuple(name_to_idx[idx_name]
                               for idx_name in str_indices)
        tensor_name = f'T_{tensor_num}'
        data_hash = tensor_num
        edges = it.combinations(tensor_indices, 2)
        graph.add_edges_from(
            edges, tensor={'name': tensor_name, 'indices': tensor_indices,
                           'data_hash': data_hash}
            )
        data_dict[data_hash] = operands[tensor_num]

    free_variables = [Var(name_to_idx[idx_name],
                          name=idx_name,
                          size=dimension_dict[idx_name]) for idx_name
                      in output_subscript]
    return graph, free_variables, data_dict


def graph_to_tnet(graph):
    """
    Construct a classical graph from graph-model

    Parameters:
    -----------
    graph : nx.MultiGraph
        A graph corresponding to graph-model.
        Edges of the graph have parameters
        'tensor_name' : str
            and
        'tensor_value': array

    Returns:
    --------
    new_graph : nx.MultiGraph
    """

    # Create new graph for mapping
    new_graph = nx.MultiGraph()

    # create list for storing all tensors in the graph
    list_of_tensors = []

    # dict for storing nodes of new graph and its data
    nodes_dict = {}

    for node in graph.nodes():
        tensor_neighbors = []
        nodes_neighbors = graph.neighbors(node)

        for neighbor in nodes_neighbors:
            tensor_name = graph.get_edge_data(
                node, neighbor)[0]['tensor_name']
            tensor_value = graph.get_edge_data(
                node, neighbor)[0]['tensor_value']
            tensor_neighbors.append(tensor_name)

            if tensor_name in list_of_tensors:
                continue
            else:
                list_of_tensors.append(tensor_name)
                num = len(list_of_tensors)
                new_graph.add_node(tensor_name, tensor=tensor_value)
        new_edges = it.combinations(set(tensor_neighbors), 2)
        new_graph.add_edges_from(new_edges)

    return new_graph
