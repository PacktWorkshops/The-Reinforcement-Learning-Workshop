import numpy as np


def tribonacci_recursive(n):
    """
    Uses recursion to calculate the nth tribonacci number
    Args:
        n: the number

    Returns:
        nth tribonacci number
    """
    if n <= 1:
        return 0
    elif n == 2:
        return 1
    else:
        return tribonacci_recursive(n - 1) + tribonacci_recursive(n - 2) + \
                tribonacci_recursive(n - 3)


def tribonacci_memo(n, memo):
    """
    Uses memoization to calculate the nth tribonacci number
    Args:
        n: the number
        memo: the dictionary that stores intermediate results
    Returns:
        nth tribonacci number
    """
    if n in memo:
        return memo[n]
    else:
        ans1 = tribonacci_memo(n - 1, memo)
        ans2 = tribonacci_memo(n - 2, memo)
        ans3 = tribonacci_memo(n - 3, memo)
        res = ans1 + ans2 + ans3
        memo[n] = res
        return res


def lcs_tabular(first, second):
    """
    Calculates the longest common substring using memoization.
    Args:
        first: the first string
        second: the second string

    Returns:
        the length of the longest common substring.
    """
    # initialize the table using numpy because it's convenient
    table = np.zeros((len(first), len(second)), dtype=int)
    for i in range(len(first)):
        for j in range(len(second)):
            if first[i] == second[j]:
                table[i][j] += 1 + table[i - 1][j - 1]
    print(table)
    return np.max(table)


def lcs_brute_force(first, second):
    """
    Use brute force to calculate the longest common substring of two strings
    Args:
        first: first string
        second: second string

    Returns:
        the length of the longest common substring
    """
    len_first = len(first)
    len_second = len(second)
    max_lcs = -1
    lcs_start, lcs_end = -1, -1
    # for every possible start in the first string
    for i1 in range(len_first):
        # for every possible end in the first string
        for j1 in range(i1, len_first):
            # for every possible start in the second string
            for i2 in range(len_second):
                # for every possible end in the second string
                for j2 in range(i2, len_second):
                    # start and end position of the current
                    # candidates
                    slice_first = slice(i1, j1)
                    slice_second = slice(i2, j2)
                    # if the strings match and the length is the
                    # highest so far
                    if first[slice_first] == second[slice_second] \
                            and j1 - i1 > max_lcs:
                        # save the lengths
                        max_lcs = j1 - i1
                        lcs_start = i1
                        lcs_end = j1

    print("LCS: ", first[lcs_start: lcs_end])
    return max_lcs


if __name__ == '__main__':
    a = "BBBABDABAA"
    b = "AAAABDABBAABB"
    lcs_tabular(a, b)
