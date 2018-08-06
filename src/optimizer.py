import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#from .Tensor import Tensor
from .operators import *
from .einsum.Expression import Expression
from .einsum.Variable import Variable
from .einsum.Tensor import Tensor

class Bucket():
    tensors = []
    def __init__(self, var):
        self.var = var
    def set_var(self,var):
        self.var = var
    def append(self,t):
        self.tensors.append(t)
    def __iadd__(self,t):
        self.append(t)
    def process(self):
        result = self.tensors[0].matrix
        for op in self.tensors:
            result = np.multiply(result,op.matrix,
                                axes=[(-1,),(index),()]
                                )



def find_tensors_by_var(tensors,var):
    ts = []
    for t in tensors:
        if var in t.variables:
         ts.append(t)
    return ts

def naive_eliminate(graph,tensors):
    for variable in graph.nodes():
        log.info(f"eliminating {variable} var")
        _tensors = find_tensors_by_var(tensors,variable)
        l = len(_tensors)
        v = str(([t.variables for t in _tensors]))
        log.debug(f'need to multiply {l} tensors {v}')
        product = _tensors[0]
        tensors.remove(_tensors[0])
        for t in _tensors[1:]:
            product = product.multiply(t,variable)
            tensors.remove(t)
        new_tensor = product.sum(over=variable)
        tensors.append(new_tensor)
        log.debug('new tensor'+str(new_tensor))
    log.info('DONE'+str(tensors))

    T = tensors[0]
    vs = T.variables
    indexes = []
    for i in range(-1,-len(vs),-1):
        indexes.append(vs.index(i))
    indexes.append(vs.index(0))
    print(indexes)
    ordered = T._tensor.transpose(indexes)
    #fl = tensors[0]._tensor.flatten()
    print(ordered)
    fl = ordered[:,0].flatten()

    if len(fl)<100:
        print('me',fl.round(2))
    return  fl[0]


def circ2graph(circuit):
    g = nx.Graph()
    tensors2vars = []
    circuit.reverse()

    qubit_count = len(circuit[0])
    print(qubit_count)

    # we start from 0 here to avoid problems with quickbb
    tensors =[]
    expr = Expression()
    vari = [Variable(0)]
    free_vars = [vari[0]]

    # Process first layer
    for i in range(1, qubit_count+1):
        op = circuit[0][i-1]
        g.add_node(i)
        tensor =Tensor(op.tensor)
        # 0 is inital index
        vari.append(Variable(i))
        print('i',i,'vari',vari)
        tensor.add_variables(vari[0],vari[i])
        tensor.name=op.name
        expr+=tensor

    current_var = qubit_count
    variable_col= list(range(1,qubit_count+1))
    print(circuit)

    for layer in circuit[1:-1]:
        for op in layer:
            tensor = Tensor(op.tensor)
            if not op.diagonal:
                # Non-diagonal gate adds a new variable and
                # an edge to graph
                g.add_node(current_var+1)
                g.add_edge(
                    variable_col[op._qubits[0]],
                    current_var+1 )
                vari.append(Variable(current_var+1))
                tensor.add_variables(
                    vari[variable_col[op._qubits[0]]],
                    vari[current_var+1] )
                current_var += 1

                variable_col[op._qubits[0]] = current_var

            elif isinstance(op,cZ):
                # cZ connects two variables with an edge
                i1 = variable_col[op._qubits[0]]
                i2 = variable_col[op._qubits[1]]
                g.add_edge(i1,i2)
                tensor.add_variables(vari[i1],vari[i2])
            # tensors2 vars is a list of tensos
            # which leads to variables it operates on
            else:
                tensor.add_variables( vari[current_var])
            tensor.name=op.name
            expr+=tensor
    i = 1
    # Will fail if thee is 
    if len(circuit[-1])!=qubit_count:
        log.warn("use max gates count on last layer")
    for op in circuit[-1]:
        tensor = Tensor(op.tensor)
        xvar = Variable(-i)
        vari.append(xvar)
        free_vars.append(xvar)
        tensor.add_variables(vari[variable_col[i-1]],xvar)
        expr += tensor
        i+=1
    log.info(f"there are {len(tensors)} tensors")
    expr.free_vars = free_vars

    v = g.number_of_nodes()
    e = g.number_of_edges()
    print(g)
    log.info(f"Generated graph with {v} nodes and {e} edges")
    log.info(f"last index contains from {variable_col}")

    aj = nx.adjacency_matrix(g)
    matfile = 'adjacency_graph.mat'
    np.savetxt(matfile ,aj.toarray(),delimiter=" ",fmt='%i')
    with open(matfile,'r') as fp:
        s = fp.read()
        #s = s.replace(' ','-')
        #print(s.replace('0','-'))

    plt.figure(figsize=(10,10))
    nx.draw(g,with_labels=True)
    plt.savefig('graph.eps')
    
    return g,expr

