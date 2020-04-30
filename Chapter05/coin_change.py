import numpy as np
import pandas as pd


def count_changes(N, denominations):
    """
    Counts the number of ways to add the coin denominations
    to N.
    Args:
        N: number to sum up to
        denominations: list of coins

    Returns:

    """
    print(f"Counting number of ways to get to {N} using coins: {denominations}")
    # table with dimension len(denomination) x (N + 1)
    # the number of columns is N + 1 since the index
    # includes zero as well
    table = np.ones((len(denominations), N + 1)).astype(int)
    # run the loop from 1 since the first row will always 1s
    for i in range(1, len(denominations)):
        for j in range(N + 1):
            if j < denominations[i]:
                # If the index is less than the denomination
                # then just copy the previous best
                table[i, j] = table[i - 1, j]
            else:
                # If not, the add two things:
                # 1. The number of ways to sum up to N *without* considering
                #    the existing denomination.
                # 2. And, the number of ways to sum up to N minus the
                #    value of the current denomination (by considering the
                #    current and the previous denominations)
                table[i, j] = table[i - 1, j] + table[i, j - denominations[i]]
    # pretty print the table
    print_table(table, denominations)


def print_table(table, denominations):
    """
    Pretty print a numpy table
    Args:
        table: table to print
        denominations: list of coins

    Returns:

    """
    df = pd.DataFrame(table)
    df = df.set_index(np.array(denominations))
    print(df)


if __name__ == '__main__':
    N = 5
    denominations = [1, 2]
    count_changes(N, denominations)


