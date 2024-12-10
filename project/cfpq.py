from dataclasses import dataclass

import networkx as nx
from pyformlang.cfg import CFG, Terminal
from pyformlang.finite_automaton import State, Symbol
from pyformlang.rsa import RecursiveAutomaton
from scipy import sparse

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.cfg_utils import cfg_to_weak_normal_form
from project.graph_utils import graph_to_nfa
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


@dataclass(frozen=True)
class RsmState:
    dfa_label: Symbol
    dfa_state: State


def get_rsm_state_to_out_states(
    rsm: RecursiveAutomaton, rsm_state: RsmState
) -> dict[Symbol, RsmState]:
    dfa = rsm.boxes[rsm_state.dfa_label].dfa
    dfa_network = dfa.to_networkx()

    res = {}

    for u, v, label in dfa_network.edges(data="label"):
        if u == rsm_state.dfa_state:
            res.setdefault(label, set()).add(RsmState(rsm_state.dfa_label, State(v)))

    return res


class GraphState(int):
    def __new__(cls, val):
        return super().__new__(cls, val)


def get_graph_state_to_out_states(
    graph: nx.DiGraph, graph_state: GraphState
) -> dict[Symbol, GraphState]:
    res = {}

    for u, v, label in graph.edges(data="label"):
        if u == graph_state:
            res.setdefault(label, set()).add(GraphState(v))

    return res


class GSSGraph(nx.MultiDiGraph):
    def __init__(self):
        super().__init__()

    def find_node_in_graph(self, node_to_find):
        for node in list(self.nodes):
            if node == node_to_find:
                return node

        return None


class GSSNode:
    rsm_state: RsmState
    graph_state: GraphState
    return_values: set[GraphState]

    def __init__(self, rsm_state, graph_state):
        self.rsm_state = rsm_state
        self.graph_state = graph_state
        self.return_values = set()

    def __eq__(self, other):
        return (
            self.rsm_state == other.rsm_state and self.graph_state == other.graph_state
        )

    def __hash__(self):
        return hash((self.rsm_state, self.graph_state))


@dataclass(frozen=True)
class GLLConfiguration:
    rsm_state: RsmState
    graph_state: GraphState
    gss_node: GSSNode


class GLLSolver:
    _gss_graph: GSSGraph
    _rsm: RecursiveAutomaton
    _rsm_state_to_out_states: dict
    _graph: nx.DiGraph
    _graph_state_to_out_states: dict

    def __init__(self, rsm: RecursiveAutomaton, graph: nx.DiGraph):
        self._rsm = rsm
        self._rsm_state_to_out_states = {}

        for box in self._rsm.boxes.values():
            for state in box.dfa.states:
                rsm_state = RsmState(box.label, state)

                self._rsm_state_to_out_states.setdefault(rsm_state, {})

                state_to_out_states = get_rsm_state_to_out_states(self._rsm, rsm_state)

                for label in state_to_out_states.keys():
                    self._rsm_state_to_out_states[rsm_state].setdefault(label, set())

                    self._rsm_state_to_out_states[rsm_state][label] = (
                        state_to_out_states[label]
                    )

        self._graph = graph
        self._graph_state_to_out_states = {}

        for node in self._graph.nodes:
            graph_state = GraphState(node)

            self._graph_state_to_out_states.setdefault(graph_state, {})

            state_to_out_states = get_graph_state_to_out_states(
                self._graph, graph_state
            )

            for label in state_to_out_states.keys():
                self._graph_state_to_out_states[graph_state].setdefault(label, set())

                self._graph_state_to_out_states[graph_state][label] = (
                    state_to_out_states[label]
                )

    def get_init_configs(self, start_nodes: set[int]) -> set[GLLConfiguration]:
        rsm_start_label = self._rsm.initial_label
        rsm_start_dfa = self._rsm.boxes[rsm_start_label].dfa
        rsm_start_state = RsmState(rsm_start_label, rsm_start_dfa.start_state)

        init_configs = set()

        for start_node in start_nodes:
            graph_state = GraphState(start_node)
            gss_node = GSSNode(rsm_start_state, graph_state)
            gll_config = GLLConfiguration(rsm_start_state, graph_state, gss_node)

            self._gss_graph.add_node(gss_node)
            init_configs.add(gll_config)

        return init_configs

    def do_algo_step(self, config: GLLConfiguration) -> set[GLLConfiguration]:
        new_configs = set()

        new_configs |= self.do_term_algo_step(config)
        new_configs |= self.do_non_term_algo_step(config)
        new_configs |= self.do_return_algo_step(config)

        return new_configs

    def do_term_algo_step(self, config: GLLConfiguration) -> set[GLLConfiguration]:
        labels = (
            self._rsm_state_to_out_states[config.rsm_state].keys()
            & self._graph_state_to_out_states[config.graph_state].keys()
        )

        new_configs = set()

        for label in labels:
            for new_rsm_state in self._rsm_state_to_out_states[config.rsm_state][label]:
                for new_graph_state in self._graph_state_to_out_states[
                    config.graph_state
                ][label]:
                    new_configs.add(
                        GLLConfiguration(
                            new_rsm_state, new_graph_state, config.gss_node
                        )
                    )

        return new_configs

    def do_non_term_algo_step(self, config: GLLConfiguration) -> set[GLLConfiguration]:
        labels = (
            self._rsm.labels & self._rsm_state_to_out_states[config.rsm_state].keys()
        )

        new_configs = set()

        for label in labels:
            new_rsm = self._rsm.boxes[label].dfa
            new_rsm_start_state = RsmState(label, new_rsm.start_state)
            new_gss_node = GSSNode(new_rsm_start_state, config.graph_state)

            node_in_graph = self._gss_graph.find_node_in_graph(new_gss_node)

            for rsm_state in self._rsm_state_to_out_states[config.rsm_state][label]:
                if node_in_graph is not None:
                    for return_graph_state in node_in_graph.return_values:
                        new_configs.add(
                            GLLConfiguration(
                                rsm_state, return_graph_state, config.gss_node
                            )
                        )
                else:
                    node_in_graph = new_gss_node

                self._gss_graph.add_node(node_in_graph)
                self._gss_graph.add_edge(
                    node_in_graph, config.gss_node, label=rsm_state
                )
                new_configs.add(
                    GLLConfiguration(
                        new_rsm_start_state, config.graph_state, node_in_graph
                    )
                )

        return new_configs

    def do_return_algo_step(self, config: GLLConfiguration) -> set[GLLConfiguration]:
        if (
            config.rsm_state.dfa_state
            not in self._rsm.boxes[config.rsm_state.dfa_label].dfa.final_states
        ):
            return set()

        config.gss_node.return_values.add(config.graph_state)

        new_configs = set()

        for u, v, return_rsm_state in self._gss_graph.edges(data="label"):
            if u == config.gss_node:
                new_configs.add(
                    GLLConfiguration(return_rsm_state, config.graph_state, v)
                )

        return new_configs

    def solve_cfpq(
        self, start_nodes: set[int], final_nodes: set[int]
    ) -> set[tuple[int, int]]:
        self._gss_graph = GSSGraph()

        init_configs = self.get_init_configs(start_nodes)
        w = init_configs.copy()
        r = init_configs.copy()

        while w:
            cur_config = w.pop()

            new_configs = self.do_algo_step(cur_config)

            for new_config in new_configs:
                if new_config not in r:
                    w.add(new_config)
                    r.add(new_config)

        res = set()

        for init_config in init_configs:
            init_gss_node = init_config.gss_node

            for return_value in init_gss_node.return_values:
                if return_value in final_nodes:
                    res.add((init_config.graph_state, return_value))

        return res


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    start_nodes = start_nodes if start_nodes else graph.nodes
    final_nodes = final_nodes if final_nodes else graph.nodes

    gll_solver = GLLSolver(rsm, graph)
    return gll_solver.solve_cfpq(start_nodes, final_nodes)
