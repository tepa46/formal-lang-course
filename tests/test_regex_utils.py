from project.regex_utils import regex_to_dfa


def test_regex_to_dfa():
    res = regex_to_dfa("a.b.c*|d")

    assert res.is_equivalent_to(res.minimize())
    assert res.is_deterministic()
    assert res.accepts("abccc")
    assert res.accepts("d")
    assert not res.accepts("cd")
