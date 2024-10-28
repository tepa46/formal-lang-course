from pyformlang.cfg import CFG, Terminal
import networkx as nx
from scipy import sparse

from project.cfg_utils import cfg_to_weak_normal_form


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
