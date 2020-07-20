import import_ipynb
from Exercise5_03 import count_changes


def test_count_changes():
    N = 5
    denominations = [1, 2]
    assert 3 == count_changes(N, denominations)
