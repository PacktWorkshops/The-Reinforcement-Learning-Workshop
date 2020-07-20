import import_ipynb
from Exercise5_02 import lcs_brute_force, lcs_tabular


def test_lcs_brute_force():
    a = "BBBABDABAA"
    b = "AAAABDABBAABB"
    assert 5 == lcs_tabular(a, b)


def test_lcs_tabular():
    a = "BBBABDABAA"
    b = "AAAABDABBAABB"
    assert 5 == lcs_tabular(a, b)
