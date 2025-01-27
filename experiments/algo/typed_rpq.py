import networkx as nx
import numpy as np
from scipy import sparse

from experiments.algo.typed_adjacency_matrix_fa import (
    AdjacencyMatrixFA,
    intersect_automata,
)
from project.graph_utils import graph_to_nfa
from project.regex_utils import regex_to_dfa
from project.vector_utils import create_bool_vector


def tensor_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    sparse_matrix_type,
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    regex_adjacency_matrix_fa = AdjacencyMatrixFA(regex_dfa, sparse_matrix_type)

    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_adjacency_matrix_fa = AdjacencyMatrixFA(graph_nfa, sparse_matrix_type)

    intersect = intersect_automata(
        graph_adjacency_matrix_fa,
        regex_adjacency_matrix_fa,
    )

    transitive_closure_matrix = intersect.get_transitive_closure_matrix()
    if intersect.sparse_matrix_type == sparse.coo_matrix:
        return coo_matrix_res(
            graph_adjacency_matrix_fa,
            regex_adjacency_matrix_fa,
            transitive_closure_matrix,
        )
    else:
        return other_matrix_res(
            graph_adjacency_matrix_fa,
            regex_adjacency_matrix_fa,
            transitive_closure_matrix,
        )


def coo_matrix_res(
    graph_adjacency_matrix_fa, regex_adjacency_matrix_fa, transitive_closure_matrix
):
    res_indexes = {}
    for graph_start_state_ind in graph_adjacency_matrix_fa.start_states_ind:
        for regex_start_state_ind in regex_adjacency_matrix_fa.start_states_ind:
            for graph_final_state_ind in graph_adjacency_matrix_fa.final_states_ind:
                for regex_final_state_ind in regex_adjacency_matrix_fa.final_states_ind:
                    intersect_start_state_ind = (
                        graph_start_state_ind * regex_adjacency_matrix_fa.states_number
                        + regex_start_state_ind
                    )
                    intersect_final_state_ind = (
                        graph_final_state_ind * regex_adjacency_matrix_fa.states_number
                        + regex_final_state_ind
                    )

                    res_indexes[
                        (intersect_start_state_ind, intersect_final_state_ind)
                    ] = (graph_start_state_ind, graph_final_state_ind)

    res = set()
    for row, col, data in zip(
        transitive_closure_matrix.row,
        transitive_closure_matrix.col,
        transitive_closure_matrix.data,
    ):
        if (row, col) in res_indexes and data:
            graph_row, graph_col = res_indexes[(row, col)]
            res_start_state = graph_adjacency_matrix_fa.ind_to_state[graph_row]
            res_final_state = graph_adjacency_matrix_fa.ind_to_state[graph_col]

            res.add((res_start_state, res_final_state))

    return res


def other_matrix_res(
    graph_adjacency_matrix_fa, regex_adjacency_matrix_fa, transitive_closure_matrix
):
    res = set()

    for graph_start_state_ind in graph_adjacency_matrix_fa.start_states_ind:
        for regex_start_state_ind in regex_adjacency_matrix_fa.start_states_ind:
            for graph_final_state_ind in graph_adjacency_matrix_fa.final_states_ind:
                for regex_final_state_ind in regex_adjacency_matrix_fa.final_states_ind:
                    intersect_start_state_ind = (
                        graph_start_state_ind * regex_adjacency_matrix_fa.states_number
                        + regex_start_state_ind
                    )
                    intersect_final_state_ind = (
                        graph_final_state_ind * regex_adjacency_matrix_fa.states_number
                        + regex_final_state_ind
                    )

                    start = transitive_closure_matrix.indptr[intersect_start_state_ind]
                    end = transitive_closure_matrix.indptr[
                        intersect_start_state_ind + 1
                    ]
                    row_indexes = transitive_closure_matrix.indices[start:end]
                    data = transitive_closure_matrix.data[start:end]

                    if intersect_final_state_ind in row_indexes:
                        idx = row_indexes.tolist().index(intersect_final_state_ind)
                        if data[idx]:
                            res_start_state = graph_adjacency_matrix_fa.ind_to_state[
                                graph_start_state_ind
                            ]
                            res_final_state = graph_adjacency_matrix_fa.ind_to_state[
                                graph_final_state_ind
                            ]

                            res.add((res_start_state, res_final_state))

    return res


def ms_bfs_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    sparse_matrix_type,
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    dfa_adjacency_matrix_fa = AdjacencyMatrixFA(regex_dfa, sparse_matrix_type)

    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    nfa_adjacency_matrix_fa = AdjacencyMatrixFA(graph_nfa, sparse_matrix_type)

    k = dfa_adjacency_matrix_fa.states_number
    n = nfa_adjacency_matrix_fa.states_number

    dfa_start_state_ind = list(dfa_adjacency_matrix_fa.start_states_ind)[0]
    nfa_start_states_ind = nfa_adjacency_matrix_fa.start_states_ind
    nfa_start_states_number = len(nfa_start_states_ind)

    def create_start_front():
        data = np.ones(nfa_start_states_number, dtype=bool)
        rows = [dfa_start_state_ind + k * j for j in range(nfa_start_states_number)]
        columns = [start_state_ind for start_state_ind in nfa_start_states_ind]

        return sparse_matrix_type(
            (data, (rows, columns)), shape=(k * nfa_start_states_number, n), dtype=bool
        )

    labels = (
        dfa_adjacency_matrix_fa.boolean_decomposition_matrices.keys()
        & nfa_adjacency_matrix_fa.boolean_decomposition_matrices.keys()
    )

    dfa_decomposed_matrices = dfa_adjacency_matrix_fa.boolean_decomposition_matrices
    dfa_decomposed_transposed_matrices = {}

    for label in labels:
        dfa_decomposed_transposed_matrices[label] = dfa_decomposed_matrices[
            label
        ].transpose()

    nfa_decomposed_matrices = nfa_adjacency_matrix_fa.boolean_decomposition_matrices

    def update_front(front):
        if sparse_matrix_type == sparse.coo_matrix:
            return update_row_coo(
                k,
                n,
                labels,
                front,
                nfa_decomposed_matrices,
                dfa_decomposed_transposed_matrices,
                nfa_start_states_number,
            )
        else:
            return update_row_other(
                k,
                n,
                labels,
                front,
                nfa_decomposed_matrices,
                dfa_decomposed_transposed_matrices,
                sparse_matrix_type,
                nfa_start_states_number,
            )

    current_front = create_start_front()
    visited = sparse_matrix_type((k * nfa_start_states_number, n), dtype=bool)

    while current_front.count_nonzero() > 0:
        visited += current_front

        current_front = update_front(current_front)

        current_front = current_front > visited

    dfa_final_states_ind = dfa_adjacency_matrix_fa.final_states_ind

    nfa_final_states_ind = nfa_adjacency_matrix_fa.final_states_ind
    nfa_final_states_vector = create_bool_vector(
        nfa_adjacency_matrix_fa.states_number, nfa_final_states_ind
    )

    res = set()

    for i, nfa_start_state_ind in enumerate(nfa_start_states_ind, 0):
        for dfa_final_state_ind in dfa_final_states_ind:
            row = visited.getrow(i * k + dfa_final_state_ind)
            row_vector = create_bool_vector(n, row.indices)

            vector = row_vector & nfa_final_states_vector

            reached_nfa_final_states_ind = np.nonzero(vector)[0]

            for reached_nfa_final_state_ind in reached_nfa_final_states_ind:
                res.add(
                    (
                        nfa_adjacency_matrix_fa.ind_to_state[nfa_start_state_ind],
                        nfa_adjacency_matrix_fa.ind_to_state[
                            reached_nfa_final_state_ind
                        ],
                    )
                )

    return res


def update_row_coo(
    k,
    n,
    labels,
    front,
    nfa_decomposed_matrices,
    dfa_decomposed_transposed_matrices,
    nfa_start_states_number,
):
    decomposed_fronts = {}
    updated_decomposed_fronts = {}
    for label in labels:
        decomposed_fronts[label] = (front @ nfa_decomposed_matrices[label]).asformat(
            sparse.coo_matrix((0, 0)).format
        )
        updated_data = []
        updated_row = []
        updated_col = []
        row_indices = decomposed_fronts[label].row
        col_indices = decomposed_fronts[label].col
        data_values = decomposed_fronts[label].data

        for ind in range(nfa_start_states_number):
            rel_range = (row_indices >= ind * k) & (row_indices < (ind + 1) * k)

            relevant_rows = row_indices[rel_range] % k
            relevant_cols = col_indices[rel_range]
            relevant_data = data_values[rel_range]

            extracted_matrix = sparse.coo_matrix(
                (relevant_data, (relevant_rows, relevant_cols)),
                shape=(k, n),
                dtype=bool,
            )

            updated_extracted_matrix = (
                dfa_decomposed_transposed_matrices[label] @ extracted_matrix
            ).asformat(sparse.coo_matrix((0, 0)).format)

            updated_data += updated_extracted_matrix.data.tolist()
            updated_row += [
                elm + ind * k for elm in updated_extracted_matrix.row.tolist()
            ]
            updated_col += updated_extracted_matrix.col.tolist()

        updated_decomposed_fronts[label] = sparse.coo_matrix(
            (updated_data, (updated_row, updated_col)),
            shape=(k * nfa_start_states_number, n),
            dtype=bool,
        )

    new_front = sparse.coo_matrix((k * nfa_start_states_number, n), dtype=bool)

    for decomposed_front in updated_decomposed_fronts.values():
        new_front += decomposed_front

    return new_front


def update_row_other(
    k,
    n,
    labels,
    front,
    nfa_decomposed_matrices,
    dfa_decomposed_transposed_matrices,
    sparse_matrix_type,
    nfa_start_states_number,
):
    decomposed_fronts = {}
    for label in labels:
        decomposed_fronts[label] = (front @ nfa_decomposed_matrices[label]).asformat(
            sparse_matrix_type((0, 0)).format
        )

        for ind in range(nfa_start_states_number):
            decomposed_fronts[label][ind * k : (ind + 1) * k] = (
                dfa_decomposed_transposed_matrices[label]
                @ decomposed_fronts[label][ind * k : (ind + 1) * k]
            ).asformat(sparse_matrix_type((0, 0)).format)

    new_front = sparse_matrix_type((k * nfa_start_states_number, n), dtype=bool)

    for decomposed_front in decomposed_fronts.values():
        new_front += decomposed_front

    return new_front
