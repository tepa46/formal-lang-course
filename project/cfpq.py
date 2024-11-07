from pyformlang.cfg import CFG, Terminal
from pyformlang.rsa import RecursiveAutomaton

import networkx as nx
from scipy import sparse

from project.graph_utils import graph_to_nfa
from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.cfg_utils import cfg_to_weak_normal_form
from project.rsm_utils import rsm_to_nfa


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    wcnf = cfg_to_weak_normal_form(cfg)
    wcnf_productions = wcnf.productions
    wcnf_nullable_symbols = wcnf.get_nullable_symbols()

    m = set()
    r = set()

    edges = graph.edges(data="label")

    for u, v, label in edges:
        for production in wcnf_productions:
            if [Terminal(label)] == production.body:
                m.add((production.head, u, v))
                r.add((production.head, u, v))

    for node in graph.nodes:
        for sym in wcnf_nullable_symbols:
            m.add((sym, node, node))
            r.add((sym, node, node))

    while m:
        (N, a, b) = m.pop()
        for M, c, d in r.copy():
            if b == c:
                for production in wcnf_productions:
                    if [N, M] == production.body:
                        tr = (production.head, a, d)
                        if tr not in r:
                            m.add(tr)
                            r.add(tr)
            if a == d:
                for production in wcnf_productions:
                    if [M, N] == production.body:
                        tr = (production.head, c, b)
                        if tr not in r:
                            m.add(tr)
                            r.add(tr)

    result = set()
    for N, a, b in r:
        if (
            N == wcnf.start_symbol
            and (not start_nodes or a in start_nodes)
            and (not final_nodes or b in final_nodes)
        ):
            result.add((a, b))

    return result


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    node_to_ind = {}
    ind_to_node = {}

    for ind, node in enumerate(graph.nodes):
        node_to_ind[node] = ind
        ind_to_node[ind] = node

    wcnf = cfg_to_weak_normal_form(cfg)
    wcnf_productions = wcnf.productions
    wcnf_nullable_symbols = wcnf.get_nullable_symbols()

    matrix_size = graph.number_of_nodes()
    Nmatrices = {
        N: sparse.csr_matrix((matrix_size, matrix_size), dtype=bool)
        for N in wcnf.variables
    }

    edges = graph.edges(data="label")

    for u, v, label in edges:
        for production in wcnf_productions:
            if production.body == [Terminal(label)]:
                Nmatrices[production.head][node_to_ind[u], node_to_ind[v]] = True

    for sym in wcnf_nullable_symbols:
        for i in range(matrix_size):
            Nmatrices[sym][i, i] = True

    m: set = wcnf.variables

    while m:
        N = m.pop()
        for production in wcnf_productions:
            if N in production.body:
                updated_Nmatrix: sparse.csr_matrix = (
                    Nmatrices[production.body[0]] @ Nmatrices[production.body[1]]
                )

                if (updated_Nmatrix > Nmatrices[production.head]).count_nonzero() > 0:
                    Nmatrices[production.head] += updated_Nmatrix
                    m.add(production.head)

    result = set()

    for i in range(matrix_size):
        for j in range(matrix_size):
            if (
                Nmatrices[wcnf.start_symbol][i, j]
                and (not start_nodes or ind_to_node[i] in start_nodes)
                and (not final_nodes or ind_to_node[j] in final_nodes)
            ):
                result.add((ind_to_node[i], ind_to_node[j]))

    return result


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_states: set[int] = None,
    final_states: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_nfa = rsm_to_nfa(rsm)
    rsm_adjacency_matrix_fa = AdjacencyMatrixFA(rsm_nfa)

    graph_nfa = graph_to_nfa(graph, start_states, final_states)
    graph_adjacency_matrix_fa = AdjacencyMatrixFA(graph_nfa)
    m = graph_adjacency_matrix_fa.states_number

    updated = True
    while updated:
        updated = False

        intersect = intersect_automata(
            rsm_adjacency_matrix_fa, graph_adjacency_matrix_fa
        )

        transitive_closure_matrix = intersect.get_transitive_closure_matrix()

        for i, j in zip(*transitive_closure_matrix.nonzero()):
            if not transitive_closure_matrix[i, j]:
                continue

            rsm_nfa_start_state_ind = i // m
            _, rsm_nfa_start_label = rsm_adjacency_matrix_fa.ind_to_state[
                rsm_nfa_start_state_ind
            ].value

            graph_nfa_start_state_ind = i % m

            rsm_nfa_final_state_ind = j // m
            _, rsm_nfa_final_label = rsm_adjacency_matrix_fa.ind_to_state[
                rsm_nfa_final_state_ind
            ].value

            graph_nfa_final_state_ind = j % m

            if rsm_nfa_start_label != rsm_nfa_final_label:
                continue

            if (
                rsm_nfa_start_state_ind in rsm_adjacency_matrix_fa.start_states_ind
                and rsm_nfa_final_state_ind in rsm_adjacency_matrix_fa.final_states_ind
            ):
                graph_adjacency_matrix_fa.boolean_decomposition_matrices.setdefault(
                    rsm_nfa_start_label,
                    sparse.csr_matrix((m, m), dtype=bool),
                )

                if not graph_adjacency_matrix_fa.boolean_decomposition_matrices[
                    rsm_nfa_start_label
                ][graph_nfa_start_state_ind, graph_nfa_final_state_ind]:
                    graph_adjacency_matrix_fa.boolean_decomposition_matrices[
                        rsm_nfa_start_label
                    ][graph_nfa_start_state_ind, graph_nfa_final_state_ind] = True
                    updated = True

    if (
        rsm.initial_label
        not in graph_adjacency_matrix_fa.boolean_decomposition_matrices.keys()
    ):
        return set()

    result = set()
    for i in range(m):
        for j in range(m):
            i_state = graph_adjacency_matrix_fa.ind_to_state[i]
            j_state = graph_adjacency_matrix_fa.ind_to_state[j]

            if graph_adjacency_matrix_fa.boolean_decomposition_matrices[
                rsm.initial_label
            ][i, j]:
                if (not start_states or i_state in start_states) and (
                    not final_states or j_state in final_states
                ):
                    result.add((i_state, j_state))

    return result
