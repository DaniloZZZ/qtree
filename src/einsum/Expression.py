import numpy as np
from .Variable import Variable
from .Tensor import Tensor
import logging
log = logging.getLogger('qtree')

class Expression:
    def __init__(self):
        self._tensors = []
        self._variables = []

    def __iadd__(self,tensor):
        if not isinstance(tensor,Tensor):
            raise Exception("expected Tensor but got",
                            tensor)
        self._tensors.append(tensor)
        self._variables+=tensor.variables
        self._variables = list(set(self._variables))
        print('addwd vars to exp',self._variables)
        return self
    def set_tensors(self,tensors):
        for t in tensors:
            self += t

    def set_order(self,order):
        """ Set index for every variable for ordering
        :param order: list of integers - ids of variables
        """
        i = 0
        for id in order:
            # set index i for variable with id id
            for v in self._variables:
                if v._id==id:
                    v.idx = i
            i+=1
        self._variables = sorted(
            self._variables, key=lambda v: v.idx)
    def get_var_id(self,i):
        res = []
        for v in self._variables:
            if v._id==i:
                res.append(v)
        return res

    def evaluate(self,free_vars=None):
        """ Evaluate the Expression by variable elimination
        algorithm
        :param free_vars: list of Variables - free vars
        :return: a Tensor with order=len(free_vars)
        """
        # variables are expexted to be sorted
        vs = []
        print('evalung',str(self))
        if not free_vars:
            free_vars = self.free_vars
        for v in self._variables:
            if v not in free_vars:
                vs.append(v)
        for var in vs:
            print('eliminating Variable',var)
            print('expr',self)
            tensors_of_var = [t for t in self._tensors
                              if var in t.variables]
            print('tensors of var:\n',
                  tensors_of_var,
                 )
            tensor_of_var = tensors_of_var[0].merge_with(
                tensors_of_var[1:])
            print(
                  '\n tensor after merge:\n',
                  tensor_of_var.__repr__())
            tensor_of_var.diagonalize_if_dupl()
            new_t = tensor_of_var.sum_by(var)
            print('tensor after sum:\n',new_t)
            new_expr_tensors = []
            for t in self._tensors:
                if t not in tensors_of_var:
                    new_expr_tensors.append(t)
                else:
                    del t
            self._tensors = new_expr_tensors
            self._tensors.append(new_t)
        return self._tensors
    def __repr__(self):
        return '.'.join([str(t) for t in self._tensors])
