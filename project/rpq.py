import networkx as nx
from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.graph_utils import graph_to_nfa
from project.regex_utils import regex_to_dfa


def tensor_based_rpq(
    regex: str, graph: nx.MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    regex_adjacency_matrix_fa = AdjacencyMatrixFA(regex_dfa)

    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_adjacency_matrix_fa = AdjacencyMatrixFA(graph_nfa)

    intersect = intersect_automata(graph_adjacency_matrix_fa, regex_adjacency_matrix_fa)

    transitive_closure_matrix = intersect.get_transitive_closure_matrix()

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

                    if transitive_closure_matrix[
                        intersect_start_state_ind, intersect_final_state_ind
                    ]:
                        res_start_state = graph_adjacency_matrix_fa.ind_to_state[
                            graph_start_state_ind
                        ]
                        res_final_state = graph_adjacency_matrix_fa.ind_to_state[
                            graph_final_state_ind
                        ]

                        res.add((res_start_state, res_final_state))

    return res
