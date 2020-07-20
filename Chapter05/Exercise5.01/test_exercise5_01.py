import import_ipynb
from Exercise5_01 import tribonacci_recursive, tribonacci_memo


def test_tribonacci_recursive():
    assert 7 == tribonacci_recursive(6)


def test_tribonacci_memoized():
    memo = {0: 0, 1: 0, 2: 1}
    assert 7 == tribonacci_memo(6, memo)
