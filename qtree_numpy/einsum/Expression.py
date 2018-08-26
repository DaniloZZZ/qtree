import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import matplotlib.pyplot as plt

from .Variable import Variable
from .Tensor import Tensor
from .Optimiser import GreedyOptimiser
from . import quickbb_api as qbb
import logging
log = logging.getLogger('qtree')

class Expression:
    def __init__(self,
                 graph_model_plot=None,
                 save_graphs=False):
        """Set Expression parameters and
        set empty lists for tensors and vars.

        Parameters
        ----------
        graph_model_plot: str, default None
            a path to file where to save a plot of graphical model
        """
        self.graph_model_plot=graph_model_plot
        self.save_graphs = save_graphs
        self._tensors = []
        self._variables = []
        self._graph = nx.Graph()
        self._graph_layout =None

    def __iadd__(self,tensor):
        """Append a tensor to Expression and add it's Variables
        if they're not already in set. Also, __update_graph()
        """
        if not isinstance(tensor,Tensor):
            raise Exception("expected Tensor but got",
                            tensor)
        self._tensors.append(tensor)
        self._variables+=tensor.variables
        self._variables = list(set(self._variables))
        #print('addwd vars to exp',self._variables)
        self.__update_graph(tensor)
        return self

    def __update_graph(self,tensor):
        for v in tensor.variables:
            self._graph.add_node(v._id)
        if len(tensor.variables)==2:
            self._graph.add_edge(
                tensor.variables[0]._id,
                tensor.variables[1]._id)
        elif len(tensor.variables)>2:
            vs = [ v._id for v in tensor.variables]
            pairs = [(vs[i],vs[j]) for i in range(len(vs)) for j in range(i)]
            #print(vs,pairs)
            self._graph.add_edges_from(pairs)
            # TODO make it work for more than 2 variables
            #raise Exception('found a tensor with more than 2 vars.')

    def set_tensors(self,tensors):
        """A wrapper of `__iadd__()` for adding list of tensors
        Parameters
        ----------
        tensors: list [Tensor]
        """
        for t in tensors:
            self += t

    def set_order_from_qbb(self):
        """Runs QuickBB and sets order of elimination
        Mutates `self._graph`: removes free vars
        Creates a plot to file `self.graph_model_plot` if defined
        Creates a config file for quickbb and some output files
        """
        cnffile = 'quickbb.cnf'
        graph = self._graph
        graph.remove_nodes_from([ v._id for v in self._variables if v.fixed])
        if self.graph_model_plot:
            self.__draw_graph(self.graph_model_plot)
        qbb.gen_cnf(cnffile,graph)
        ordering = qbb.run_quickbb(cnffile)
        print("Ordering from QuickBB is",ordering)
        self.set_order(ordering)

    def __draw_graph(self,path):
        if not self._graph_layout:
            self._graph_layout = nx.spectral_layout(self._graph)
            print (self._graph_layout)
        ids = [v._id for v in self._variables]
        layout = {i:pos for i,pos in self._graph_layout.items() if i in ids}
        plt.figure(figsize=(10,10))
        nx.draw(self._graph,
                pos=layout,
                node_color=np.array(list(self._graph.nodes())),
                node_size=400,
                cmap=plt.cm.Blues,
                with_labels=True,
               )
        plt.savefig(path)

    def set_order(self,order):
        """Set order of elimination.
        Every variable is assigned an index, then

        Parameters
        ----------
        order: list of integers
            indexes of variables, representing the desired order
        """
        self.ordering=order
        for v in self._variables:
            if v._id in order:
                v.idx = order.index(v._id)
        self._variables = sorted(
            self._variables, key=lambda v: v.idx)

    def fix_vars_for_parallel(self,rank,nproc):
        # Here we assume that every variable has 2 vals:0 and 1
        # TODO: Support arbitary variable space size
        # var count is minimum k so that 2^k>nproc
        values = self.__get_values_for_fix(rank,nproc)
        variables = self.__paralleled_vars
        for i in range(len(values)):
            variables[i].fix(values[i])
        log.info('process with id %i evaluates with vals %s',
                 rank,values)
        self.__paralleled_vars = self.__paralleled_vars[:len(values)]
        print('parallised vars',self.__paralleled_vars)

    def _get_ids_for_fix(self,nproc):
        var_count=_next_exp_of2(nproc)
        opt = GreedyOptimiser(num_items=var_count)
        self._graph.remove_nodes_from([ v._id for v in self._variables if v.fixed])
        def cost_func(nodes):
            g = self._graph
            cost = 0
            for n in g.nodes():
                for _n in g.neighbors(n):
                    if _n not in nodes:
                        cost+=1
            return cost
        nodes = opt.optimise(list(self._graph.nodes()),cost_func)
        print(opt.process)
        print('greeder returned',nodes)
        return nodes

    def __get_values_for_fix(self,rank,nproc):
        """
        desired output for nproc=4: 00,01,10,11
        desired output for nproc=5: 000,001,01,10,11
        """
        var_count=_next_exp_of2(nproc)
        leaf_count = np.power(2,var_count)
        non_merged_leafs = nproc - ( leaf_count-nproc )
        if rank<non_merged_leafs:
            val_str = bin(rank)[2:].zfill(var_count)
        else:
            val_str = bin(rank+(rank-non_merged_leafs))[2:-1].zfill(var_count-1)
        values = [int(x) for x in val_str]
        return values

    def get_var_id(self,i):
        """Get Variables with given integer id
        returns all vars with var._id == i

        Parameters
        ----------
        i: int
            id of variable
        Returns
        ----------
        res: list [Variable]
        """
        res = []
        for v in self._variables:
            if v._id==i:
                res.append(v)
        return res

    def parallel_evaluate(self):
        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        rank = comm.Get_rank()
        log.info('Evaluating the expression: %s',str(self))
        if rank==0:
            start_time = time.time()
            self.set_order_from_qbb()
            print("qbb-- %s seconds --" % (time.time() - start_time))
            ordering = comm.bcast(self.ordering,root=0)
            nodes = self._get_ids_for_fix(nproc)
            self.__paralleled_vars = [v for v in self._variables if v._id in nodes]
            parv = comm.bcast(nodes,root=0)
        else:
            ordering=None
            ordering = comm.bcast(ordering,root=0)
            self.set_order(ordering)
            nodes= None
            nodes= comm.bcast(nodes,root=0)
            self.__paralleled_vars = [v for v in self._variables if v._id in nodes]
        self.fix_vars_for_parallel(rank,nproc)
        start_time = time.time()
        for t in self._tensors:
            t.slice_if_fixed()
        log.info('Expression is now:%s',str(self))
        # Iterate over only non-free vars
        vs = [v for v in self._variables if not v.fixed]
        res = self._variable_eliminate(vs)
        if rank==0:
            results = [res]
            for i in range(1,nproc):
                res_ = comm.recv(source=i,tag=42)
                log.debug('received result ',i,res_)
                results.append(res_)
            log.info('results', results)
            res = sum([r[0]._tensor for r in results])
        else:
            req = comm.send(res ,dest=0,tag=42)
        self.eval_time = time.time() - start_time
        print("eval%s--- %s seconds --" % (rank,self.eval_time))
        if rank==0:
            return [Tensor(res)]

    def evaluate(self,parallel=False):
        """Evaluate the Expression by summing over non-free vars
        Uses variable elimination algorithm.
        You should call Variable.fix() in advance to set up
        variables of resulting tensor.
        Mutates the instance: removes all variables except fixed
        and removes all tensors except the resulting one

        Returns
        ----------
        self._tensors : list [Tensor]
            If everything went OK, list contains one tensor
            with rank equal to fixed variables count
            If not, warning is printed (Check if graph connected)
        """
        # self.variables are expected to be sorted
        log.info('Evaluating the expression: %s',str(self))
        self.set_order_from_qbb()
        log.info('Slicing by fixed vars %s',[x for x in self._variables if x.fixed])
        for t in self._tensors:
            t.slice_if_fixed()
        log.info('Expression is now:%s',str(self))
        # Iterate over only non-free vars
        vs = [v for v in self._variables if not v.fixed]
        start_time = time.time()
        r = self._variable_eliminate(vs)
        self.eval_time = time.time() - start_time
        print("Eval time-- %s seconds --" % (self.eval_time))
        return r

    def _variable_eliminate(self,vs):
        tensor_sizes = []
        for var in vs:
            log.debug('expr %s',self)
            tensors_of_var = [t for t in self._tensors
                              if var in t.variables]
            log.info('Eliminating %s %i \twith %i tensors, ranks:%s',
                     var,var.idx,len(tensors_of_var),
                    str([t.rank for t in tensors_of_var]),
                    )
            log.debug('tensors of var: \n%s', tensors_of_var)
            tensor_of_var = tensors_of_var[0].merge_with(
                tensors_of_var[1:])
            #tensor_of_var.diagonalize_if_dupl()
            new_t = tensor_of_var.sum_by(var)
            tensor_sizes.append(new_t.rank)
            log.debug('tensor after sum:\n%s',new_t)
            new_expr_tensors = []
            for t in self._tensors:
                if t not in tensors_of_var:
                    new_expr_tensors.append(t)
                else:
                    del t
            self._tensors = new_expr_tensors
            self._tensors.append(new_t)
            self._graph.remove_nodes_from([var._id])
            self.__update_graph(new_t)
            if self.save_graphs :
                self.__draw_graph('./graphs/graph_%i.png'%var.idx)
        if len(self._tensors)>1:
            x = 1
            for t in self._tensors:
                x*=t._tensor
                # TODO: check the rank here, if >0 then something went wrong
            self._tensors=[Tensor(x)]
        print("DONE varelim max ts:",max(tensor_sizes))
        print("sum ts",sum(tensor_sizes))
        return self._tensors

    def __repr__(self):
        return ' '.join([str(t) for t in self._tensors])

def _next_exp_of2(n):
    n-=1
    e = 0
    while n>=1:
        n/=2
        e+=1
    return e

