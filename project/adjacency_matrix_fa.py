from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from typing import Any, Iterable
import networkx as nx
import numpy as np
from scipy import sparse
from regex_utils import regex_to_dfa
from graph_utils import graph_to_nfa


class AdjacencyMatrixFA:
    states_number: int
    boolean_decomposition_matrices: dict[Any, sparse.csr_matrix]
    start_states: set[int]
    final_states: set[int]
    nodes_to_ind: dict[State, int]
    ind_to_state: dict[int, State]

    def __init__(self, fa: NondeterministicFiniteAutomaton = None):
        self.boolean_decomposition_matrices = {}
        self.start_states = set()
        self.final_states = set()

        if fa is None:
            self.states_number = 0
            return

        self.states_number = len(fa.states)
        self.nodes_to_ind = {}

        for ind, state in enumerate(fa.states, 0):
            self.nodes_to_ind[state] = ind
            self.ind_to_state[ind] = state

        for start_state in fa.start_states:
            self.start_states.add(self.nodes_to_ind[start_state])

        for final_state in fa.final_states:
            self.final_states.add(self.nodes_to_ind[final_state])

        graph = fa.to_networkx()
        edges = graph.edges(data="label")

        nodes_connectivity = {}
        labels = set()

        for edge in edges:
            u = edge[0]
            v = edge[1]
            label = edge[2]

            if label is not None:
                nodes_connectivity.setdefault(label, []).append((u, v))
                labels.add(label)

        for label in labels:
            data = np.ones(len(nodes_connectivity[label]), dtype=bool)
            rows = list(
                map(lambda conn: self.nodes_to_ind[conn[0]], nodes_connectivity[label])
            )
            columns = list(
                map(lambda conn: self.nodes_to_ind[conn[1]], nodes_connectivity[label])
            )

            decomposed_matrix = sparse.csr_matrix(
                (data, (rows, columns)),
                shape=(self.states_number, self.states_number),
            )

            self.boolean_decomposition_matrices[label] = decomposed_matrix

    def accepts(self, word: Iterable[Symbol]) -> bool:
        states_vector = create_bool_vector(self.states_number, self.start_states)

        for symbol in word:
            states_vector = states_vector @ self.boolean_decomposition_matrices[symbol]

        final_states_vector = create_bool_vector(self.states_number, self.final_states)

        return np.any(states_vector & final_states_vector)

    def is_empty(self) -> bool:
        states_vector = create_bool_vector(self.states_number, self.start_states)

        transitive_matrix = sparse.csr_matrix(
            (
                np.ones(self.states_number, dtype=bool),
                (range(self.states_number), range(self.states_number)),
            ),
            shape=(self.states_number, self.states_number),
        )

        for matrix in self.boolean_decomposition_matrices.values():
            transitive_matrix = matrix + transitive_matrix

        transitive_matrix = transitive_matrix ** (self.states_number - 1)

        states_vector = states_vector @ transitive_matrix

        final_states_vector = create_bool_vector(self.states_number, self.final_states)

        return not np.any(states_vector & final_states_vector)


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    n = automaton1.states_number
    m = automaton2.states_number
    kron_matrix_size = n * m

    kron_boolean_decomposition_matrices = {}

    kron_labels = (
        automaton1.boolean_decomposition_matrices.keys()
        & automaton2.boolean_decomposition_matrices.keys()
    )

    for label in kron_labels:
        kron_boolean_decomposition_matrices[label] = sparse.kron(
            automaton1.boolean_decomposition_matrices[label],
            automaton2.boolean_decomposition_matrices[label],
        )

    start_states = set()
    final_states = set()

    for start_state1 in automaton1.start_states:
        for start_state2 in automaton2.start_states:
            start_states.add(start_state1 * m + start_state2)

    for final_state1 in automaton1.final_states:
        for final_state2 in automaton2.final_states:
            final_states.add(final_state1 * m + final_state2)

    adjacency_matrix_fa = AdjacencyMatrixFA()

    adjacency_matrix_fa.states_number = kron_matrix_size
    adjacency_matrix_fa.boolean_decomposition_matrices = (
        kron_boolean_decomposition_matrices
    )
    adjacency_matrix_fa.start_states = start_states
    adjacency_matrix_fa.final_states = final_states

    return adjacency_matrix_fa


def create_bool_vector(n, true_inds):
    vector = np.zeros(n, dtype=bool)

    for true_ind in true_inds:
        vector[true_ind] = True

    return vector


def tensor_based_rpq(
    regex: str, graph: nx.MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    regex_adjacency_matrix_fa = AdjacencyMatrixFA(regex_dfa)

    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_adjacency_matrix_fa = AdjacencyMatrixFA(graph_nfa)

    intersect = intersect_automata(graph_adjacency_matrix_fa, regex_adjacency_matrix_fa)

    transitive_matrix = sparse.csr_matrix(
        (
            np.ones(intersect.states_number, dtype=bool),
            (range(intersect.states_number), range(intersect.states_number)),
        ),
        shape=(intersect.states_number, intersect.states_number),
    )

    for matrix in intersect.boolean_decomposition_matrices.values():
        transitive_matrix = matrix + transitive_matrix

    transitive_matrix = transitive_matrix ** (intersect.states_number - 1)

    res = set()
    for graph_start_state in graph_adjacency_matrix_fa.start_states:
        for regex_start_state in regex_adjacency_matrix_fa.start_states:
            for graph_final_state in graph_adjacency_matrix_fa.final_states:
                for regex_final_state in regex_adjacency_matrix_fa.final_states:
                    intersect_start_state = (
                        graph_start_state * regex_adjacency_matrix_fa.states_number
                        + regex_start_state
                    )
                    intersect_final_state = (
                        graph_final_state * regex_adjacency_matrix_fa.states_number
                        + regex_final_state
                    )

                    if transitive_matrix[intersect_start_state][intersect_final_state]:
                        res_start_state = graph_adjacency_matrix_fa.ind_to_state[
                            graph_start_state
                        ]
                        res_final_state = graph_adjacency_matrix_fa.ind_to_state[
                            graph_final_state
                        ]

                        res.add((res_start_state, res_final_state))

    return res
