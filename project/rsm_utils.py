from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.cfg import CFG


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for label in rsm.labels:
        dfa = rsm.boxes[label].dfa

        nfa.add_start_state((dfa.start_state, label))

        for final_state in dfa.final_states:
            nfa.add_final_state((final_state, label))

        transitions = dfa.to_dict()

        for u in transitions:
            for sym, v in transitions[u].items():
                nfa.add_transition((u, label), sym, (v, label))

    return nfa
