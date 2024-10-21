from pyformlang.cfg import CFG, Terminal
import networkx as nx

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
