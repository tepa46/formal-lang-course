from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import DeterministicFiniteAutomaton


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    eps_nfa = Regex(regex).to_epsilon_nfa()
    min_dfa = eps_nfa.to_deterministic().minimize()

    return min_dfa
