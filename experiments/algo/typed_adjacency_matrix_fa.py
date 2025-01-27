import numpy as np
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from typing import Any, Iterable
from scipy import sparse
from project.vector_utils import create_bool_vector


class AdjacencyMatrixFA:
    states_number: int
    boolean_decomposition_matrices: dict[Any, Any]
    start_states_ind: set[int]
    final_states_ind: set[int]
    state_to_ind: dict[State, int]
    ind_to_state: dict[int, State]

    def __init__(
        self,
        fa: NondeterministicFiniteAutomaton = None,
        sparse_matrix_type=sparse.csr_matrix,
    ):
        self.sparse_matrix_type = sparse_matrix_type
        self.boolean_decomposition_matrices = {}
        self.start_states_ind = set()
        self.final_states_ind = set()
        self.state_to_ind = {}
        self.ind_to_state = {}

        if fa is None:
            self.states_number = 0
            return

        self.states_number = len(fa.states)

        for ind, state in enumerate(fa.states, 0):
            self.state_to_ind[state] = ind
            self.ind_to_state[ind] = state

        for start_state in fa.start_states:
            self.start_states_ind.add(self.state_to_ind[start_state])

        for final_state in fa.final_states:
            self.final_states_ind.add(self.state_to_ind[final_state])

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
                map(lambda conn: self.state_to_ind[conn[0]], nodes_connectivity[label])
            )
            columns = list(
                map(lambda conn: self.state_to_ind[conn[1]], nodes_connectivity[label])
            )

            decomposed_matrix = self.sparse_matrix_type(
                (data, (rows, columns)),
                shape=(self.states_number, self.states_number),
            )

            self.boolean_decomposition_matrices[label] = decomposed_matrix

    def accepts(self, word: Iterable[Symbol]) -> bool:
        states_ind_vector = create_bool_vector(
            self.states_number, self.start_states_ind
        )

        for symbol in word:
            states_ind_vector = (
                states_ind_vector @ self.boolean_decomposition_matrices[symbol]
            )

        final_states_ind_vector = create_bool_vector(
            self.states_number, self.final_states_ind
        )

        return np.any(states_ind_vector & final_states_ind_vector)

    def is_empty(self) -> bool:
        states_ind_vector = create_bool_vector(
            self.states_number, self.start_states_ind
        )

        transitive_closure_matrix = self.get_transitive_closure_matrix()

        states_ind_vector = states_ind_vector @ transitive_closure_matrix

        final_states_ind_vector = create_bool_vector(
            self.states_number, self.final_states_ind
        )

        return not np.any(states_ind_vector & final_states_ind_vector)

    def get_transitive_closure_matrix(self):
        transitive_closure_matrix = self.sparse_matrix_type(
            (
                np.ones(self.states_number, dtype=bool),
                (range(self.states_number), range(self.states_number)),
            ),
            shape=(self.states_number, self.states_number),
        )

        for matrix in self.boolean_decomposition_matrices.values():
            transitive_closure_matrix = matrix + transitive_closure_matrix

        transitive_closure_matrix = transitive_closure_matrix ** (
            self.states_number - 1
        )

        return transitive_closure_matrix.asformat(
            self.sparse_matrix_type((0, 0)).format
        )


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    if automaton1.sparse_matrix_type != automaton2.sparse_matrix_type:
        raise ValueError(f"automata must have the same sparse_matrix_type")
    sparse_matrix_type = automaton1.sparse_matrix_type

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
        ).asformat(sparse_matrix_type((0, 0)).format)

    start_states_ind = set()
    final_states_ind = set()

    for start_state_ind1 in automaton1.start_states_ind:
        for start_state_ind2 in automaton2.start_states_ind:
            start_states_ind.add(start_state_ind1 * m + start_state_ind2)

    for final_state_ind1 in automaton1.final_states_ind:
        for final_state_ind2 in automaton2.final_states_ind:
            final_states_ind.add(final_state_ind1 * m + final_state_ind2)

    state_to_ind = {}
    ind_to_state = {}

    for i in range(0, kron_matrix_size):
        state_to_ind[i] = i
        ind_to_state[i] = i

    adjacency_matrix_fa = AdjacencyMatrixFA()

    adjacency_matrix_fa.sparse_matrix_type = sparse_matrix_type
    adjacency_matrix_fa.states_number = kron_matrix_size
    adjacency_matrix_fa.boolean_decomposition_matrices = (
        kron_boolean_decomposition_matrices
    )
    adjacency_matrix_fa.start_states_ind = start_states_ind
    adjacency_matrix_fa.final_states_ind = final_states_ind
    adjacency_matrix_fa.state_to_ind = state_to_ind
    adjacency_matrix_fa.ind_to_state = ind_to_state

    return adjacency_matrix_fa
