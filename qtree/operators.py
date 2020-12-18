"""
This module implements quantum gates from the CMON set of Google
"""
import numpy as np
import re
import cirq

from fractions import Fraction
from qtree.logger_setup import log
from functools import partial

import qtree.system_defs as defs
import uuid


class Gate:
    """
    Base class for quantum gates.

    Attributes
    ----------
    name: str
            The name of the gate

    parameters: dict
             Parameters used by the gate (may be empty)

    qubits: tuple
            Qubits the gate acts on

    changed_qubits : tuple
            Tuple of ints which states what qubit's bases are changed
            (along which qubits the gate is not diagonal).

    cirq_op: Cirq.GridQubit
            Cirq 2D gate. Used for unit tests. Optional

    Methods
    -------
    gen_tensor(): numpy.array
            The gate tensor. For each qubit a gate
            either introduces a new variable (non-diagonal gate, like X)
            or does not (diagonal gate, like T). Multiqubit gates
            can be diagonal on some of the variables, and not diagonal on
            others (like ccX). The order of dimensions IS ALWAYS
            (new_a, a, b_new, b, c, d_new, d, ...)

    dagger():
            Class method that returns a daggered class

    dagger_me():
            Changes the instance's gen_tensor inplace

    is_parametric(): bool
            Returns False for gates without parameters
    """

    def __init__(self, *qubits):
        self._qubits = tuple(qubits)
        # supposedly unique id for an instance
        self._parameters = { }
        self._check_qubit_count(qubits)
        self.name = type(self).__name__

    def _check_qubit_count(self, qubits):
        n_qubits = len(self.gen_tensor().shape) - len(
            self._changes_qubits)
        # return back the saved version

        if len(qubits) != n_qubits:
            raise ValueError(
                "Wrong number of qubits for gate {}:\n"
                "{}, required: {}".format(
                    self.name, len(qubits), n_qubits))

    @classmethod
    def dag_tensor(cls, inst):
        return cls.gen_tensor(inst).conj().T

    @classmethod
    def dagger(cls):
        # This thing modifies the base class itself.
        orig = cls.gen_tensor
        def conj_tensor(self):
            t = orig(self)
            return t.conj().T
        cls.gen_tensor = conj_tensor
        cls.__name__ += '.dag'
        return cls

    def dagger_me(self):
        # Maybe the better way is to create a separate object
        # Warning: dagger().dagger().dagger() will define many things
        self.gen_tensor = partial(self.dag_tensor, self)
        self.name += '+'
        return self


    def gen_tensor(self):
        raise NotImplementedError()

    @property
    def parameters(self):
        return self._parameters

    def is_parametric(self):
        return len(self.parameters) > 0

    @property
    def qubits(self):
        return self._qubits

    @property
    def changed_qubits(self):
        return tuple(self._qubits[idx] for idx in self._changes_qubits)

    def to_cirq_1d_circ_op(self):
        return self.cirq_op(
            *[cirq.LineQubit(qubit) for qubit in self._qubits]
        )

    def __str__(self):
        return ("{}".format(self.name) +
                "({})".format(','.join(map(str, self._qubits)))
        )

    def __repr__(self):
        return self.__str__()


class ParametricGate(Gate):
    """
    Gate with parameters.

    Attributes
    ----------
    name: str
            The name of the gate

    parameters: dict
             Parameters used by the gate (may be empty)

    qubits: tuple
            Qubits the gate acts on

    changed_qubits : tuple
            Tuple of ints which states what qubit's bases are changed
            (along which qubits the gate is not diagonal).

    cirq_op: Cirq.GridQubit
            Cirq 2D gate. Used for unit tests. Optional

    Methods
    -------
    gen_tensor(\\**parameters={}): numpy.array
            The gate tensor. For each qubit a gate
            either introduces a new variable (non-diagonal gate, like X)
            or does not (diagonal gate, like T). Multiqubit gates
            can be diagonal on some of the variables, and not diagonal on
            others (like ccX). The order of dimensions IS ALWAYS
            (new_a, a, b_new, b, c, d_new, d, ...)

    is_parametric(): bool
            Returns True
    """
    def __init__(self, *qubits, **parameters):
        self._qubits = tuple(qubits)
        # supposedly unique id for an instance
        self._parameters = parameters
        self._check_qubit_count(qubits)
        self.name = type(self).__name__

    def _check_qubit_count(self, qubits):
        # fill parameters and save a copy
        filled_parameters = {}
        for par, value in self._parameters.items():
            if isinstance(value, placeholder):
                filled_parameters[par] = np.zeros(value.shape)
            else:
                filled_parameters[par] = value
        parameters = self._parameters

        # substitute parameters by filled parameters
        # to evaluate tensor shape
        self._parameters = filled_parameters
        n_qubits = len(self.gen_tensor().shape) - len(
            self._changes_qubits)
        # return back the saved version
        self._parameters = parameters

        if len(qubits) != n_qubits:
            raise ValueError(
                "Wrong number of qubits: {}, required: {}".format(
                    len(qubits), n_qubits))

    def gen_tensor(self, **parameters):
        if len(parameters) == 0:
            return self._gen_tensor(**self._parameters)
        else:
            return self._gen_tensor(**parameters)

    def __str__(self):
        par_str = (",".join("{}={}".format(
            param_name,
            '?.??' if isinstance(param_value, placeholder)
            else '{:.2f}'.format(float(param_value)))
                            for param_name, param_value in
                            sorted(self._parameters.items(),
                                   key=lambda pair: pair[0])))

        return ("{}".format(self.name) + "[" + par_str + "]" +
                "({})".format(','.join(map(str, self._qubits))))


def op_scale(factor, operator, name):
    """
    Scales a gate class by a scalar. The resulting class
    will have a scaled tensor

    It is not recommended to use this many times because of
    possibly low performance

    Parameters
    ----------
    factor: float
          scaling factor
    operator: class
          operator to modify
    name: str
          Name of the new class
    Returns
    -------
    class
    """
    def gen_tensor(self):
        return factor * operator.gen_tensor(operator)

    attr_dict = {attr: getattr(operator, attr) for attr in dir(operator)}
    attr_dict['gen_tensor'] = gen_tensor

    return type(name, (operator, ), attr_dict)


class M(Gate):
    """
    Measurement gate. This is essentially the identity operator, but
    it forces the introduction of a variable in the graphical model
    """
    def gen_tensor(self):
        return np.array([[1, 0], [0, 1]], dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )
    cirq_op = cirq.I


class I(Gate):
    def gen_tensor(self):
        return np.array([1, 1], dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = tuple()
    cirq_op = cirq.I


class H(Gate):
    """
    Hadamard gate
    """
    def gen_tensor(self):
        return 1/np.sqrt(2) * np.array([[1,  1],
                                        [1, -1]],
                                       dtype=defs.NP_ARRAY_TYPE)
    _changes_qubits = (0, )
    cirq_op = cirq.H


class Z(Gate):
    """
    :math:`Z`-gate
    """
    def gen_tensor(self):
        return np.array([1, -1],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = tuple()
    cirq_op = cirq.Z



class cZ(Gate):
    """
    Controlled :math:`Z` gate
    """
    def gen_tensor(self):
        return np.array([[1, 1],
                         [1, -1]],
                        dtype=defs.NP_ARRAY_TYPE)
    _changes_qubits = tuple()
    cirq_op = cirq.CZ


class T(Gate):
    """
    :math:`T`-gate
    """
    def gen_tensor(self):
        return np.array([1, np.exp(1.j*np.pi/4)],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = tuple()
    cirq_op = cirq.T


class Tdag(Gate):
    """
    :math:`T` inverse gate
    """
    def gen_tensor(self):
        pass
        return np.array([1, np.exp(-1.j*np.pi/4)],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = tuple()
    cirq_op = cirq.inverse(cirq.T)


class S(Gate):
    """
    :math:`S`-gate
    """
    def gen_tensor(self):
        return np.array([1, 1j],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = tuple()
    cirq_op = cirq.S


class Sdag(Gate):
    """
    :math:`S` inverse gate
    """
    def gen_tensor(self):
        return np.array([1, -1j],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = tuple()
    cirq_op = cirq.inverse(cirq.S)


class X_1_2(Gate):
    """
    :math:`X^{1/2}`
    gate
    """
    def gen_tensor(self):
        return (Fraction(1, 2) *
                np.array([[1 + 1j, 1 - 1j],
                          [1 - 1j, 1 + 1j]])
        ).astype(defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )

    def cirq_op(self, x): return cirq.X(x)**0.5


class Y_1_2(Gate):
    r"""
    :math:`Y^{1/2}` gate
    """
    def gen_tensor(self):
        return (Fraction(1, 2) *
                np.array([[1 + 1j, -1 - 1j],
                          [1 + 1j, 1 + 1j]])
        ).astype(defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )

class W_1_2(Gate):
    r"""
    :math:`W^{1/2}` gate
    """
    def gen_tensor(self):
        sqj = np.sqrt(1j)
        nsqj = np.sqrt(-1j)
        return (1/np.sqrt(2)*
                np.array([[1 , -sqj],
                          [nsqj, 1]])
        ).astype(defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )

    def cirq_op(self, x): return cirq.Y(x)**0.5


class X(Gate):
    def gen_tensor(self):
        return np.array([[0, 1],
                         [1, 0]],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )

    def cirq_op(self, x): return cirq.X(x)


class cX(Gate):
    def gen_tensor(self):
        return np.array([[[1., 0.],
                          [0., 1.]],
                         [[0., 1.],
                          [1., 0.]]])

    _changes_qubits = (1, )
    cirq_op = cirq.CNOT


class ccX(Gate):
    def gen_tensor(self):
        #TODO: tensor shapes
        return np.array([])

    _changes_qubits = (2, )
    cirq_op = cirq.CCNOT


class Y(Gate):
    def gen_tensor(self):
        return np.array([[0, -1j],
                         [1j, 0]],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )

    def cirq_op(self, x): return cirq.Y(x)


class cY(Gate):
    def gen_tensor(self):
        return np.array([[[1., 0.],
                          [0., 1.]],
                         [[0., -1j],
                          [1j, 0.]]])

    _changes_qubits = (0, )

    def cirq_op(self, x): pass


class YPhase(ParametricGate):
    """A gate that rotates around the Y axis of the Bloch sphere.

        The unitary matrix of ``YPowGate(exponent=t)`` is:

            [[g·c, -g·s],
             [g·s, g·c]]

        where:

            c = cos(π·t/2)
            s = sin(π·t/2)
            g = exp(i·π·t/2).
    """

    _changes_qubits = (0, )

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along Y axis"""
        alpha = parameters['alpha']

        c = np.cos(np.pi * alpha / 2)
        s = np.sin(np.pi * alpha / 2)
        g = np.exp(1j * np.pi * alpha / 2)

        return np.array([[g * c, -1 * g * s],
                         [g * s, g * c]])

    def cirq_op(self, x): return cirq.YPowGate(
            exponent=float(self._parameters['alpha']))(x)


def ry(parameters: [float], *qubits):
    """Arbitrary :math:`Y` rotation"""

    return YPhase(*qubits, alpha=parameters[0]/np.pi)


class ZPhase(ParametricGate):
    """Arbitrary :math:`Z` rotation
    [[1, 0],
    [0, g]],  where

    g = exp(i·π·t)
    """

    _changes_qubits = tuple()
    parameter_count = 1

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along Z axis"""
        alpha = parameters['alpha']
        return np.array([1., np.exp(1j * np.pi * alpha)])

    def cirq_op(self, x): return cirq.ZPowGate(
            exponent=float(self._parameters['alpha']))(x)


def rz(parameters: [float], *qubits):
    """Arbitrary :math:`Z` rotation"""

    return ZPhase(*qubits, alpha=parameters[0]/np.pi)


def rx(parameters: [float], *qubits):
    """Arbitrary :math:`X` rotation"""

    return XPhase(*qubits, alpha=parameters[0]/np.pi)


class XPhase(ParametricGate):
    """Arbitrary :math:`X` rotation
    [[g·c, -g·s],
    [g·s, g·c]], where

    c = cos(π·alpha/2), s = sin(π·alpha/2), g = exp(i·π·alpha/2).
    """

    _changes_qubits = (0, )
    parameter_count = 1

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along X axis"""
        alpha = parameters['alpha']

        c = np.cos(np.pi*alpha/2)
        s = np.sin(np.pi*alpha/2)
        g = np.exp(1j*np.pi*alpha/2)

        return np.array([[g*c, -1j*g*s],
                         [-1j*g*s, g*c]])

    def cirq_op(self, x): return cirq.XPowGate(
            exponent=float(self._parameters['alpha']))(x)

class fSim(ParametricGate):
    _changes_qubits = (0, 1)
    parameter_count = 2

    @staticmethod
    def _gen_tensor(**parameters):
        alpha, beta = parameters['alpha'], parameters['beta']
        c = np.cos(alpha)
        s = np.sin(alpha)
        g = np.exp(-1j*beta)

        abcd = np.array([[[[1, 0],
                           [0, 0]],

                          [[0, c],
                           [-1j*s, 0]]],


                         [[[0, -1j*s],
                           [c, 0]],

                          [[0, 0],
                           [0, g]]]])
        acbd = np.swapaxes(abcd, 1, 2)
        return acbd

class SWAP(Gate):
    ## This gate is a snowflake for graph model
    ## it swaps qubits
    _changes_qubits = tuple()
    @staticmethod
    def gen_tensor():
        return np.array([[1,1],[1,1]])



class U(ParametricGate):
    """ Arbitrary single qubit unitary operator
    U(t, p, l) =
    [exp(-j*(p+l)/2)*c, -exp(-j*(p-l)/2)*s],
    [exp( j*(p-l)/2)*s,  exp( j*(p+l)/2)*c]

    where c = cos(t/2)
          s = sin(t/2)
    """

    _changes_qubits = (0,)

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along X axis"""
        theta = parameters['theta']
        phi = parameters['phi']
        lambda_param = parameters['lambda_param']

        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        mat_00 = np.exp(-1j * (phi + lambda_param) / 2) * c
        mat_01 = -np.exp(-1j * (phi - lambda_param) / 2) * s
        mat_10 = np.exp(1j * (phi - lambda_param) / 2) * s
        mat_11 = np.exp(1j * (phi + lambda_param) / 2) * c

        return np.array([[mat_00, mat_01],
                         [mat_10, mat_11]])

    def cirq_op(self, x): pass


def u3(parameters: [float], *qubits):
    """Arbitrary single qubit rotation"""

    return U(*qubits, theta=parameters[0], phi=parameters[1], lambda_param=parameters[2])


def u2(parameters: [float], *qubits):
    """Qiskit rotation operation"""

    return U(*qubits, theta=np.pi/2, phi=parameters[0], lambda_param=parameters[1])


def u1(parameters: [float], *qubits):
    """Qiskit rotation operation"""

    return U(*qubits, theta=0, phi=0, lambda_param=parameters[0])


LABEL_TO_GATE_DICT = {
    'i': I,
    'h': H,
    't': T,
    'z': Z,
    'cz': cZ,
    'cX': cX,
    'rz': ZPhase,
    'rX': XPhase,
    'x': X,
    'y': Y,
    'x_1_2': X_1_2,
    'y_1_2': Y_1_2,
    'hz_1_2': W_1_2,
    'fs': fSim
}

def read_circuit_file(filename, max_depth=None):
    log.info("reading file {}".format(filename))
    with open(filename, "r") as fp:
        n, qc = read_circuit_stream(fp, max_depth=max_depth)
    return n, qc

def read_circuit_stream(stream, max_depth=None):
    """
    Read circuit file and return quantum circuit in the
    form of a list of lists

    Parameters
    ----------
    filename : str
             circuit file in the format of Sergio Boixo
    max_depth : int
             maximal depth of gates to read

    Returns
    -------
    qubit_count : int
            number of qubits in the circuit
    circuit : list of lists
            quantum circuit as a list of layers of gates
    """

    operation_search_patt = r'(?P<operation>' + r'|'.join(LABEL_TO_GATE_DICT.keys()) + r')(?P<qubits>( \d+(?!\.))+)'
    params_search_patt_1 = r'(?P<operation>' + r'|'.join(LABEL_TO_GATE_DICT.keys()) + r')(?P<qubits>( \d+)+) (?P<alpha>-?\d+\.\d+)$'
    params_search_patt_2 = r'(?P<operation>' + r'|'.join(LABEL_TO_GATE_DICT.keys()) + r')(?P<qubits>( \d+)+) (?P<alpha>-?\d+\.\d+) (?P<beta>-?\d+\.\d+)$'

    circuit = []
    circuit_layer = []

    qubit_count = int(stream.readline())
    log.info("There are {:d} qubits in circuit".format(qubit_count))
    n_ignored_layers = 0
    current_layer = 0

    for idx, line in enumerate(stream):
        if not line or line =='\n':
            break

        m = re.search(r'(?P<layer>[0-9]+) (?=[a-z])', line)
        if m is None:
            raise Exception("file format error at line {}: {}".format(idx, line))
        # Read circuit layer by layer
        layer_num = int(m.group('layer'))

        if max_depth is not None and layer_num > max_depth:
            n_ignored_layers = layer_num - max_depth
            continue

        if layer_num > current_layer:
            circuit.append(circuit_layer)
            circuit_layer = []
            current_layer = layer_num

        op_str = line[m.end():]
        m = re.search(operation_search_patt, op_str)
        if m is None:
            raise Exception("file format error in {}".format(op_str))

        op_identif = m.group('operation')

        q_idx = tuple(int(qq) for qq in m.group('qubits').split())
        op_cls = LABEL_TO_GATE_DICT[op_identif]
        # A conditional on using simplification of fsim gate.
        if op_identif=='fso':
            if False:
                circuit_layer.append(cZ(*q_idx))
                circuit_layer.append(H(q_idx[0]))
                circuit_layer.append(H(q_idx[1]))
                circuit_layer.append(cZ(*q_idx))
            else:
                circuit_layer.append(cZ(*q_idx))
                circuit_layer.append(SWAP(*q_idx))
        else:
            if issubclass(op_cls, ParametricGate):
                if op_cls.parameter_count==1:
                    m = re.search(params_search_patt_1, op_str)
                    if m is None:
                        raise Exception(f'Could not find parameter for gate. `{line})')
                    alpha = m.group('alpha')
                    op = op_cls(*q_idx, alpha=float(alpha))
                elif op_cls.parameter_count==2:
                    m = re.search(params_search_patt_2, op_str)
                    alpha = m.group('alpha')
                    beta = m.group('beta')
                    op = op_cls(*q_idx, alpha=float(alpha), beta=float(beta))
            else:
                op = op_cls(*q_idx)
            circuit_layer.append(op)

    circuit.append(circuit_layer)  # last layer

    if n_ignored_layers > 0:
        log.info("Ignored {} layers".format(n_ignored_layers))

    return qubit_count, circuit

def circuit_to_text(circuit, n_qubits):
    reverse_dict = {v:k for k,v in LABEL_TO_GATE_DICT.items()}
    file = f"{n_qubits}\n"
    for i, layer in enumerate(circuit):
        for gate in layer:
            qubit_label=reverse_dict[type(gate)]
            qubits = ' '.join([str(q) for q in gate.qubits])
            params = gate._parameters
            param_str = ' '.join([str(x) for x in params.values()])
            if param_str:
                file += f"{i} {qubit_label} {qubits} {param_str}\n"
            else:
                file += f"{i} {qubit_label} {qubits} \n"
    return file


def read_qasm_file(filename, max_ins=None):
    """
    Read circuit file in the QASM format and return
    quantum circuit in the form of a list of lists

    Parameters
    ----------
    filename : str
             circuit file in the QASM format
    max_ins : int
             maximal depth of instructions to read

    Returns
    -------
    qubit_count : int
            number of qubits in the circuit
    circuit : list of lists
            quantum circuit as a list of layers of gates
    """
    from qiskit import QuantumCircuit

    qiskit_circuit = QuantumCircuit.from_qasm_file(filename)
    return from_qiskit_circuit(qiskit_circuit)

def from_qiskit_circuit(qiskit_circuit):
    circuit_qubits = qiskit_circuit.qubits

    output_circuit = []
    indexed_circuit = []

    # creates the indexed circuit, which replaces
    # qubits with their indices
    for circuit_operation in qiskit_circuit:
        qubit_list = []
        for circuit_operation_qubits in circuit_operation[1]:
            qubit_list.append(circuit_qubits.index(circuit_operation_qubits))

        indexed_circuit.append((circuit_operation[0], qubit_list))

    # for each operation, append the resulting internal gate representation(s)
    for circuit_operation in indexed_circuit:
        operations = _qiskit_operation_to_internal_representation(circuit_operation)
        output_circuit.append(operations)

    # flatten the list. It may be of arbitrary depth due to nested gate definitions
    output_circuit = list(_flatten(output_circuit))

    return_circuit = []
    for gate in output_circuit:
        return_circuit.append([gate])

    qubit_count = len(qiskit_circuit.qubits)
    return qubit_count, return_circuit


def _qiskit_operation_to_internal_representation(circuit_operation):
    """
    Read a modified circuit operation and return
    an internal representation of it

    Parameters
    ----------
    circuit_operation : (qiskit_instruction, [int])
            tuple of a qiskit instruction and the
            indices of the qubits it acts on

    Returns
    -------
    output_circuit : list of gate operations in
            internal representation
    """
    from qiskit import circuit
    name_to_gate_dict = {
        'id': I,
        'u0': 'U_0',
        'u1': u1,
        'u2': u2,
        'u3': u3,
        'x': X,
        'y': Y,
        'z': Z,
        'h': H,
        's': S,
        'sdg': Sdag,
        't': T,
        'tdg': Tdag,
        'rx': rx,
        'ry': ry,
        'rz': rz,
        'reset': '\\left|0\\right\\rangle',
        'cx': cX,
        'CX': cX,
        'ccx': 'ccX',
        'cy': cY,
        'cz': cZ
    }

    instruction = circuit_operation[0]
    instruction_qubits = circuit_operation[1]

    output_circuit = []

    # if the operation is part of the qiskit standard gate set
    if (instruction.name in name_to_gate_dict) and (isinstance(instruction, circuit.gate.Gate)):
        if instruction.params:
            op = name_to_gate_dict[instruction.name](instruction.params, *instruction_qubits)
        else:
            op = name_to_gate_dict[instruction.name](*instruction_qubits)

        output_circuit.append(op)

    # if the operation is a custom-defined gate whose basic gate definition needs to be found
    else:
        if (isinstance(instruction, circuit.Instruction)) and instruction.definition:
            for definition in instruction.definition:
                # replaces the definition of which qubits the custom gate acts on with the actual qubit indices
                new_qubit_list = []
                for definition_qubits in definition[1]:
                    new_qubit_list.append(instruction_qubits[definition_qubits.index])

                defined_operation = (definition[0], new_qubit_list)

                # recursively call the function until all nested custom gates are defined
                op = _qiskit_operation_to_internal_representation(defined_operation)
                output_circuit.append(op)

    return output_circuit


class placeholder:
    """
    Class for placeholders. Placeholders are used to implement
    symbolic computation. This class is very similar to the
    Tensorflow's placeholder class.

    Attributes
    ----------
    name: str, default None
          Name of the placeholder (for clarity)
    shape: tuple, default None
          Shape of the tensor the placeholder represent
    """
    def __init__(self, name=None, shape=()):
        self._name = name
        self._shape = shape
        self._uuid = uuid.uuid4()

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def uuid(self):
        return self._uuid


def _flatten(l):
    """
    https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    Parameters
    ----------
    l: iterable
        arbitrarily nested list of lists

    Returns
    -------
    generator of a flattened list
    """
    import collections
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el
